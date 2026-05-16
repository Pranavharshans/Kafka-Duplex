"""Stage 1 JSONL dataset and collation."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class Stage1JsonlDataset(Dataset[dict[str, object]]):
    def __init__(self, path: str, *, context_length: int) -> None:
        self.path = Path(path)
        self.context_length = context_length
        self.rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        token_ids = row["sequence_token_ids"][: self.context_length]
        return {
            "example_id": row["example_id"],
            "task": row["task"],
            "transcript": row["transcript"],
            "input_ids": token_ids,
        }


def collate_stage1_batch(batch: list[dict[str, object]]) -> dict[str, torch.Tensor | list[str]]:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    labels = []
    attention_mask = []
    transcripts: list[str] = []
    tasks: list[str] = []
    example_ids: list[str] = []

    for item in batch:
        seq = list(item["input_ids"])
        pad = max_len - len(seq)
        input_ids.append(seq + [0] * pad)
        labels.append(seq + [-100] * pad)
        attention_mask.append([1] * len(seq) + [0] * pad)
        transcripts.append(str(item["transcript"]))
        tasks.append(str(item["task"]))
        example_ids.append(str(item["example_id"]))

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
        "transcripts": transcripts,
        "tasks": tasks,
        "example_ids": example_ids,
    }
