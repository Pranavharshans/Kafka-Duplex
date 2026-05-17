"""Run small-sample Stage 1 checkpoint evaluation with task-aware generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafka_duplex.schema import SPECIAL_TOKEN_IDS
from models.stage1_model import Stage1CausalLM, Stage1ModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Stage 1 checkpoint on a few ASR/TTS examples.")
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint-step-XXXX.pt file.")
    parser.add_argument("--manifest", required=True, help="Path to a Stage 1 JSONL manifest.")
    parser.add_argument("--output", default="", help="Optional JSONL output path for generated sample records.")
    parser.add_argument("--task", choices=["ASR", "TTS", ""], default="", help="Optional task filter.")
    parser.add_argument("--example-id", default="", help="Optional single example id to evaluate.")
    parser.add_argument("--limit", type=int, default=4, help="Maximum number of examples to evaluate.")
    parser.add_argument("--device", default="cuda", help="Inference device.")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_model(checkpoint_payload: dict, device: torch.device) -> Stage1CausalLM:
    config = checkpoint_payload["config"]
    model_config = Stage1ModelConfig(
        vocab_size=int(config["model"]["vocab_size"]),
        context_length=int(config["model"]["context_length"]),
        hidden_size=int(config["model"].get("hidden_size", 768)),
        num_layers=int(config["model"].get("num_layers", 12)),
        num_heads=int(config["model"].get("num_heads", 12)),
        ffw_multiplier=int(config["model"].get("ffw_multiplier", 4)),
        dropout=float(config["model"].get("dropout", 0.1)),
    )
    model = Stage1CausalLM(model_config).to(device)
    model.load_state_dict(checkpoint_payload["model"])
    model.eval()
    return model


def find_prompt_length(task: str, sequence_token_ids: list[int]) -> int:
    if task == "ASR":
        marker = SPECIAL_TOKEN_IDS["SOT"]
    else:
        marker = SPECIAL_TOKEN_IDS["SOS"]
    for index, token_id in enumerate(sequence_token_ids):
        if token_id == marker:
            return index + 1
    raise ValueError(f"Could not find prompt marker for task {task}.")


def find_target_region(task: str, sequence_token_ids: list[int]) -> list[int]:
    if task == "ASR":
        start_marker = SPECIAL_TOKEN_IDS["SOT"]
        stop_marker = SPECIAL_TOKEN_IDS["EOT"]
    else:
        start_marker = SPECIAL_TOKEN_IDS["SOS"]
        stop_marker = SPECIAL_TOKEN_IDS["EOS"]

    start_index = None
    for index, token_id in enumerate(sequence_token_ids):
        if token_id == start_marker:
            start_index = index + 1
            break
    if start_index is None:
        raise ValueError(f"Missing start marker for task {task}.")

    stop_index = None
    for index in range(start_index, len(sequence_token_ids)):
        if sequence_token_ids[index] == stop_marker:
            stop_index = index
            break
    if stop_index is None:
        raise ValueError(f"Missing stop marker for task {task}.")

    return sequence_token_ids[start_index:stop_index]


def stop_token_for_task(task: str) -> int:
    return SPECIAL_TOKEN_IDS["EOT"] if task == "ASR" else SPECIAL_TOKEN_IDS["EOS"]


def greedy_generate(
    model: Stage1CausalLM,
    *,
    prompt_ids: list[int],
    stop_token_id: int,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[list[int], str]:
    generated = list(prompt_ids)
    for _ in range(max_new_tokens):
        input_ids = torch.tensor([generated], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        with torch.no_grad():
            logits, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
        next_token_id = int(logits[0, -1].argmax().item())
        generated.append(next_token_id)
        if next_token_id == stop_token_id:
            return generated, "stop_token"
    return generated, "max_new_tokens"


def evaluate_row(model: Stage1CausalLM, row: dict[str, object], device: torch.device) -> dict[str, object]:
    task = str(row["task"])
    sequence_token_ids = list(row["sequence_token_ids"])
    prompt_length = find_prompt_length(task, sequence_token_ids)
    prompt_ids = sequence_token_ids[:prompt_length]
    target_region = find_target_region(task, sequence_token_ids)
    generated_ids, stop_reason = greedy_generate(
        model,
        prompt_ids=prompt_ids,
        stop_token_id=stop_token_for_task(task),
        max_new_tokens=max(1, len(target_region) + 1),
        device=device,
    )
    generated_region = generated_ids[prompt_length:]
    if generated_region and generated_region[-1] == stop_token_for_task(task):
        generated_region = generated_region[:-1]

    compared = min(len(generated_region), len(target_region))
    exact_matches = sum(
        1 for predicted_token_id, target_token_id in zip(generated_region[:compared], target_region[:compared], strict=True)
        if predicted_token_id == target_token_id
    )
    match_rate = (exact_matches / compared) if compared > 0 else 0.0

    return {
        "example_id": row["example_id"],
        "task": task,
        "transcript": row["transcript"],
        "prompt_token_ids": prompt_ids,
        "generated_token_ids": generated_region,
        "target_token_ids": target_region,
        "generated_length": len(generated_region),
        "target_length": len(target_region),
        "exact_matches": exact_matches,
        "compared_tokens": compared,
        "exact_match_rate": match_rate,
        "stop_reason": stop_reason,
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else None

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    checkpoint_payload = torch.load(checkpoint_path, map_location=device)
    model = build_model(checkpoint_payload, device)

    rows = load_rows(manifest_path)
    filtered_rows = []
    for row in rows:
        if args.task and row["task"] != args.task:
            continue
        if args.example_id and row["example_id"] != args.example_id:
            continue
        filtered_rows.append(row)
        if len(filtered_rows) >= args.limit:
            break

    if not filtered_rows:
        raise RuntimeError("No manifest rows matched the requested filters.")

    payloads = [evaluate_row(model, row, device) for row in filtered_rows]
    for payload in payloads:
        print(json.dumps(payload, ensure_ascii=True), flush=True)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "\n".join(json.dumps(payload, ensure_ascii=True) for payload in payloads) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
