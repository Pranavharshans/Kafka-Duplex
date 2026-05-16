"""Stage 1 training loop."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from models.stage1_model import Stage1CausalLM, Stage1ModelConfig
from training.stage1_dataset import Stage1JsonlDataset, collate_stage1_batch


@dataclass(slots=True)
class Stage1RunConfig:
    config_path: str
    output_dir: str
    device: str = "cuda"


def load_config(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def lr_for_step(base_lr: float, warmup_steps: int, step: int) -> float:
    if warmup_steps <= 0:
        return base_lr
    if step >= warmup_steps:
        return base_lr
    return base_lr * (step / warmup_steps)


def greedy_decode(logits: torch.Tensor) -> list[int]:
    return logits.argmax(dim=-1).tolist()


def run_stage1_training(run_config: Stage1RunConfig) -> None:
    config = load_config(run_config.config_path)
    output_dir = Path(run_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = Path(config["dataset"]["train_manifest"])
    val_manifest = Path(config["dataset"]["val_manifest"])
    context_length = int(config["model"]["context_length"])

    train_dataset = Stage1JsonlDataset(str(train_manifest), context_length=context_length)
    val_dataset = Stage1JsonlDataset(str(val_manifest), context_length=context_length)

    micro_batch_size = int(config["optimization"]["micro_batch_size"])
    train_loader = DataLoader(train_dataset, batch_size=micro_batch_size, shuffle=True, collate_fn=collate_stage1_batch)
    val_loader = DataLoader(val_dataset, batch_size=micro_batch_size, shuffle=False, collate_fn=collate_stage1_batch)

    model_config = Stage1ModelConfig(
        vocab_size=int(config["model"]["vocab_size"]),
        context_length=context_length,
        hidden_size=int(config["model"].get("hidden_size", 768)),
        num_layers=int(config["model"].get("num_layers", 12)),
        num_heads=int(config["model"].get("num_heads", 12)),
        ffw_multiplier=int(config["model"].get("ffw_multiplier", 4)),
        dropout=float(config["model"].get("dropout", 0.1)),
    )
    device = torch.device(run_config.device if torch.cuda.is_available() else "cpu")
    model = Stage1CausalLM(model_config).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=float(config["optimization"]["learning_rate"]),
        weight_decay=float(config["optimization"]["weight_decay"]),
    )

    effective_batch = int(config["optimization"]["effective_batch_size"])
    grad_accum_steps = max(1, effective_batch // micro_batch_size)
    epochs = int(config["optimization"]["epochs"])
    eval_every = int(config["evaluation"]["eval_every_steps"])
    checkpoint_every = int(config["evaluation"]["checkpoint_every_steps"])
    warmup_steps = int(config["optimization"]["warmup_steps"])
    max_grad_norm = float(config["optimization"]["max_grad_norm"])

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    train_iter = 0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            train_iter += 1
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits, loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            (loss / grad_accum_steps).backward()

            if train_iter % grad_accum_steps == 0:
                global_step += 1
                lr = lr_for_step(float(config["optimization"]["learning_rate"]), warmup_steps, global_step)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if global_step % 20 == 0:
                    print(f"train step={global_step} epoch={epoch + 1} loss={loss.item():.4f} lr={lr:.6f}", flush=True)

                if global_step % eval_every == 0:
                    evaluate(model, val_loader, device, output_dir, global_step)

                if global_step % checkpoint_every == 0:
                    checkpoint_path = output_dir / f"checkpoint-step-{global_step}.pt"
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "global_step": global_step,
                            "config": config,
                        },
                        checkpoint_path,
                    )
                    print(f"checkpoint path={checkpoint_path}", flush=True)

    final_path = output_dir / "checkpoint-final.pt"
    torch.save({"model": model.state_dict(), "config": config, "global_step": global_step}, final_path)
    print(f"training_complete steps={global_step} final_checkpoint={final_path}", flush=True)


def evaluate(model: Stage1CausalLM, val_loader: DataLoader, device: torch.device, output_dir: Path, step: int) -> None:
    model.eval()
    losses: list[float] = []
    sample_payloads: list[str] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits, loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            losses.append(loss.item())

            if batch_index == 0:
                greedy = greedy_decode(logits[0, :-1])
                sample_payloads.append(
                    json.dumps(
                        {
                            "example_id": batch["example_ids"][0],
                            "task": batch["tasks"][0],
                            "transcript": batch["transcripts"][0],
                            "predicted_token_ids": greedy[:64],
                        },
                        ensure_ascii=True,
                    )
                )

            if batch_index >= 7:
                break

    average_loss = sum(losses) / len(losses)
    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    sample_path = eval_dir / f"step-{step}.samples.jsonl"
    sample_path.write_text("\n".join(sample_payloads) + ("\n" if sample_payloads else ""), encoding="utf-8")
    print(f"eval step={step} val_loss={average_loss:.4f} samples={sample_path}", flush=True)
    model.train()
