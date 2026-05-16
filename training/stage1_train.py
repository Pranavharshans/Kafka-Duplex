"""Stage 1 training loop."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - optional at runtime
    SummaryWriter = None

from models.stage1_model import Stage1CausalLM, Stage1ModelConfig
from training.stage1_dataset import Stage1JsonlDataset, collate_stage1_batch


@dataclass(slots=True)
class Stage1RunConfig:
    config_path: str
    output_dir: str
    device: str = "cuda"
    allow_cpu_fallback: bool = False


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


def per_example_losses(logits: torch.Tensor, labels: torch.Tensor) -> list[float]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    token_losses = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.size())
    valid_mask = shift_labels.ne(-100)
    losses: list[float] = []
    for row_loss, row_mask in zip(token_losses, valid_mask, strict=True):
        valid_count = int(row_mask.sum().item())
        if valid_count == 0:
            losses.append(0.0)
            continue
        losses.append(float(row_loss[row_mask].mean().item()))
    return losses


def run_stage1_training(run_config: Stage1RunConfig) -> None:
    config = load_config(run_config.config_path)
    output_dir = Path(run_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = output_dir / "tb"
    writer = SummaryWriter(log_dir=str(tb_dir)) if SummaryWriter is not None else None

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
    requested_device = run_config.device
    cuda_available = torch.cuda.is_available()
    if requested_device == "cuda" and not cuda_available and not run_config.allow_cpu_fallback:
        raise RuntimeError(
            "CUDA was requested for Stage 1 training, but torch.cuda.is_available() returned False. "
            "Refusing to silently fall back to CPU."
        )
    device = torch.device(requested_device if cuda_available else "cpu")
    model = Stage1CausalLM(model_config).to(device)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not use_bf16)

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

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"):
                logits, loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            scaler.scale(loss / grad_accum_steps).backward()

            if train_iter % grad_accum_steps == 0:
                global_step += 1
                lr = lr_for_step(float(config["optimization"]["learning_rate"]), warmup_steps, global_step)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if global_step % 20 == 0:
                    print(f"train step={global_step} epoch={epoch + 1} loss={loss.item():.4f} lr={lr:.6f}", flush=True)
                    if writer is not None:
                        writer.add_scalar("train/loss", float(loss.item()), global_step)
                        writer.add_scalar("train/lr", float(lr), global_step)

                if global_step % eval_every == 0:
                    evaluate(model, val_loader, device, output_dir, global_step, writer)

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
    if writer is not None:
        writer.close()


def evaluate(
    model: Stage1CausalLM,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    step: int,
    writer: SummaryWriter | None = None,
) -> None:
    model.eval()
    losses: list[float] = []
    task_losses: dict[str, list[float]] = {"ASR": [], "TTS": []}
    asr_sample_payloads: list[str] = []
    tts_sample_payloads: list[str] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16, enabled=device.type == "cuda"):
                logits, loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            losses.append(loss.item())
            example_losses = per_example_losses(logits.float(), labels)
            for idx, task in enumerate(batch["tasks"]):
                task_losses.setdefault(task, []).append(example_losses[idx])
                greedy = greedy_decode(logits[idx, :-1])
                payload = json.dumps(
                    {
                        "example_id": batch["example_ids"][idx],
                        "task": task,
                        "transcript": batch["transcripts"][idx],
                        "predicted_token_ids": greedy[:64],
                        "target_token_ids": input_ids[idx, :64].tolist(),
                    },
                    ensure_ascii=True,
                )
                if task == "ASR" and not asr_sample_payloads:
                    asr_sample_payloads.append(payload)
                if task == "TTS" and not tts_sample_payloads:
                    tts_sample_payloads.append(payload)

            if batch_index >= 7:
                break

    average_loss = sum(losses) / len(losses)
    asr_loss = sum(task_losses["ASR"]) / len(task_losses["ASR"]) if task_losses["ASR"] else average_loss
    tts_loss = sum(task_losses["TTS"]) / len(task_losses["TTS"]) if task_losses["TTS"] else average_loss
    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    asr_sample_path = eval_dir / f"step-{step}.asr.samples.jsonl"
    asr_sample_path.write_text("\n".join(asr_sample_payloads) + ("\n" if asr_sample_payloads else ""), encoding="utf-8")
    tts_sample_path = eval_dir / f"step-{step}.tts.samples.jsonl"
    tts_sample_path.write_text("\n".join(tts_sample_payloads) + ("\n" if tts_sample_payloads else ""), encoding="utf-8")
    print(
        " ".join(
            [
                f"eval step={step}",
                f"val_loss={average_loss:.4f}",
                f"asr_val_loss={asr_loss:.4f}",
                f"tts_val_loss={tts_loss:.4f}",
                f"asr_samples={asr_sample_path}",
                f"tts_samples={tts_sample_path}",
            ]
        ),
        flush=True,
    )
    if writer is not None:
        writer.add_scalar("eval/loss", float(average_loss), step)
        writer.add_scalar("eval/asr_loss", float(asr_loss), step)
        writer.add_scalar("eval/tts_loss", float(tts_loss), step)
    model.train()
