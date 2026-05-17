"""Final Stage 1 checkpoint evaluation with decoded ASR, WER, and optional TTS audio dumps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafka_duplex.audio import concatenate, save_wav_mono_pcm16
from kafka_duplex.codec import create_codec
from kafka_duplex.token_interface import (
    Stage1TokenInterface,
    build_hf_stage1_token_interface,
    legacy_stage1_token_interface,
)
from models.stage1_model import Stage1CausalLM, Stage1ModelConfig
from scripts.eval_stage1_checkpoint import (
    find_prompt_length,
    find_target_region,
    greedy_generate,
    resolve_special_token_ids,
    stop_token_for_task,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run decoded final evaluation for a Stage 1 checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint-step-XXXX.pt or checkpoint-final.pt.")
    parser.add_argument("--manifest", required=True, help="Path to a Stage 1 JSONL manifest.")
    parser.add_argument("--output-dir", required=True, help="Directory for summary JSON, samples JSONL, and optional WAVs.")
    parser.add_argument("--device", default="cuda", help="Inference device.")
    parser.add_argument("--max-asr-samples", type=int, default=32, help="Number of ASR examples to evaluate.")
    parser.add_argument("--max-tts-samples", type=int, default=8, help="Number of TTS examples to evaluate.")
    parser.add_argument("--codec", default="cosyvoice", help="Codec used to optionally decode TTS outputs.")
    parser.add_argument("--write-tts-audio", action="store_true", help="Decode and write TTS WAVs if the codec supports it.")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_model(checkpoint_payload: dict[str, Any], device: torch.device) -> Stage1CausalLM:
    config = checkpoint_payload["config"]
    model_config = Stage1ModelConfig(
        vocab_size=int(config["model"]["vocab_size"]),
        context_length=int(config["model"]["context_length"]),
        backbone=str(config["model"].get("backbone", "Stage1BootstrapLM")),
        hf_model_name=str(config["model"].get("hf_model_name", "")),
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


def build_token_interface(config: dict[str, Any]) -> Stage1TokenInterface:
    token_interface = config.get("token_interface")
    dataset = config.get("dataset", {})
    text_tokenizer_name = str(dataset.get("text_tokenizer", "mock"))
    if not isinstance(token_interface, dict) or not token_interface:
        return legacy_stage1_token_interface() if text_tokenizer_name == "mock" else build_hf_stage1_token_interface(text_tokenizer_name)

    return Stage1TokenInterface(
        text_tokenizer_name=str(token_interface.get("text_tokenizer_name", text_tokenizer_name)),
        text_vocab_size=int(token_interface.get("text_vocab_size", 49_152)),
        special_token_ids={str(key): int(value) for key, value in dict(token_interface.get("special_token_ids", {})).items()},
        speech_vocab_offset=int(token_interface.get("speech_vocab_offset", 49_152)),
        speech_vocab_size=int(token_interface.get("speech_vocab_size", 4_096)),
        total_vocab_size=int(token_interface.get("total_vocab_size", 53_260)),
    )


def normalize_text_for_wer(text: str) -> list[str]:
    return [token for token in text.upper().strip().split() if token]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = normalize_text_for_wer(reference)
    hyp = normalize_text_for_wer(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0

    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1] / len(ref)


def decode_asr_text(token_interface: Stage1TokenInterface, generated_region: list[int]) -> str:
    text_token_ids = [token_id for token_id in generated_region if token_id < token_interface.text_vocab_size]
    return token_interface.decode_text(text_token_ids)


def generate_region(
    model: Stage1CausalLM,
    row: dict[str, Any],
    *,
    device: torch.device,
    special_token_ids: dict[str, int],
) -> tuple[list[int], list[int], str]:
    task = str(row["task"])
    sequence_token_ids = list(row["sequence_token_ids"])
    prompt_length = find_prompt_length(task, sequence_token_ids, special_token_ids)
    prompt_ids = sequence_token_ids[:prompt_length]
    target_region = find_target_region(task, sequence_token_ids, special_token_ids)
    generated_ids, stop_reason = greedy_generate(
        model,
        prompt_ids=prompt_ids,
        stop_token_id=stop_token_for_task(task, special_token_ids),
        max_new_tokens=max(1, len(target_region) + 1),
        device=device,
    )
    generated_region = generated_ids[prompt_length:]
    if generated_region and generated_region[-1] == stop_token_for_task(task, special_token_ids):
        generated_region = generated_region[:-1]
    return generated_region, target_region, stop_reason


def decode_tts_audio(token_interface: Stage1TokenInterface, codec_name: str, generated_region: list[int]):
    codec = create_codec(codec_name)
    if not codec.supports_decode:
        return None
    raw_speech_ids = token_interface.vocab_to_raw_speech_ids(generated_region)
    if not raw_speech_ids:
        return None

    chunk_size = 10
    decoded_chunks = []
    for start in range(0, len(raw_speech_ids), chunk_size):
        chunk = raw_speech_ids[start:start + chunk_size]
        if len(chunk) < chunk_size:
            chunk = chunk + [0] * (chunk_size - len(chunk))
        decoded_chunks.append(codec.decode_chunk(chunk))
    return concatenate(decoded_chunks)


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    checkpoint_payload = torch.load(checkpoint_path, map_location=device)
    model = build_model(checkpoint_payload, device)
    config = checkpoint_payload["config"]
    special_token_ids = resolve_special_token_ids(config)
    token_interface = build_token_interface(config)
    rows = load_rows(manifest_path)

    asr_rows = [row for row in rows if row["task"] == "ASR"][: args.max_asr_samples]
    tts_rows = [row for row in rows if row["task"] == "TTS"][: args.max_tts_samples]

    asr_payloads: list[dict[str, Any]] = []
    tts_payloads: list[dict[str, Any]] = []
    wer_values: list[float] = []
    exact_match_rates_asr: list[float] = []
    exact_match_rates_tts: list[float] = []

    for row in asr_rows:
        generated_region, target_region, stop_reason = generate_region(
            model, row, device=device, special_token_ids=special_token_ids
        )
        hypothesis = decode_asr_text(token_interface, generated_region)
        wer = word_error_rate(str(row["transcript"]), hypothesis)
        compared = min(len(generated_region), len(target_region))
        exact_matches = sum(
            1
            for predicted_token_id, target_token_id in zip(generated_region[:compared], target_region[:compared], strict=True)
            if predicted_token_id == target_token_id
        )
        exact_rate = exact_matches / compared if compared else 0.0
        wer_values.append(wer)
        exact_match_rates_asr.append(exact_rate)
        asr_payloads.append(
            {
                "example_id": row["example_id"],
                "task": "ASR",
                "transcript": row["transcript"],
                "predicted_text": hypothesis,
                "generated_token_ids": generated_region,
                "target_token_ids": target_region,
                "stop_reason": stop_reason,
                "exact_match_rate": exact_rate,
                "wer": wer,
            }
        )

    for index, row in enumerate(tts_rows):
        generated_region, target_region, stop_reason = generate_region(
            model, row, device=device, special_token_ids=special_token_ids
        )
        compared = min(len(generated_region), len(target_region))
        exact_matches = sum(
            1
            for predicted_token_id, target_token_id in zip(generated_region[:compared], target_region[:compared], strict=True)
            if predicted_token_id == target_token_id
        )
        exact_rate = exact_matches / compared if compared else 0.0
        exact_match_rates_tts.append(exact_rate)
        payload: dict[str, Any] = {
            "example_id": row["example_id"],
            "task": "TTS",
            "transcript": row["transcript"],
            "generated_token_ids": generated_region,
            "target_token_ids": target_region,
            "stop_reason": stop_reason,
            "exact_match_rate": exact_rate,
        }
        audio = decode_tts_audio(token_interface, args.codec, generated_region) if args.write_tts_audio else None
        if audio is not None:
            wav_path = output_dir / f"tts_sample_{index:02d}.wav"
            save_wav_mono_pcm16(str(wav_path), audio)
            payload["decoded_wav_path"] = str(wav_path)
        tts_payloads.append(payload)

    summary = {
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "num_asr_samples": len(asr_payloads),
        "num_tts_samples": len(tts_payloads),
        "mean_wer": statistics.mean(wer_values) if wer_values else None,
        "median_wer": statistics.median(wer_values) if wer_values else None,
        "mean_asr_exact_match_rate": statistics.mean(exact_match_rates_asr) if exact_match_rates_asr else None,
        "mean_tts_exact_match_rate": statistics.mean(exact_match_rates_tts) if exact_match_rates_tts else None,
        "codec": args.codec,
        "tts_audio_written": bool(args.write_tts_audio and tts_payloads and any("decoded_wav_path" in payload for payload in tts_payloads)),
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "asr_samples.jsonl").write_text(
        "\n".join(json.dumps(payload, ensure_ascii=True) for payload in asr_payloads) + ("\n" if asr_payloads else ""),
        encoding="utf-8",
    )
    (output_dir / "tts_samples.jsonl").write_text(
        "\n".join(json.dumps(payload, ensure_ascii=True) for payload in tts_payloads) + ("\n" if tts_payloads else ""),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
