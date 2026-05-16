"""Stage 1 alignment data structures and serialization helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
from typing import Iterable

from .schema import SPECIAL_TOKEN_IDS, SPECIAL_TOKENS, SPEECH_VOCAB_OFFSET


@dataclass(slots=True)
class Stage1AlignmentExample:
    """One Stage 1 ASR or TTS training example."""

    task: str
    example_id: str
    transcript: str
    text_token_ids: list[int]
    speech_token_ids: list[int]
    source_audio_path: str
    speaker_id: str
    chapter_id: str
    utterance_id: str
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.task not in {"ASR", "TTS"}:
            raise ValueError("Stage1AlignmentExample.task must be 'ASR' or 'TTS'.")
        if not self.text_token_ids:
            raise ValueError("Stage1AlignmentExample requires non-empty text_token_ids.")
        if not self.speech_token_ids:
            raise ValueError("Stage1AlignmentExample requires non-empty speech_token_ids.")

    @property
    def prompt_token(self) -> str:
        return SPECIAL_TOKENS[self.task]

    def to_training_sequence(self) -> list[str | int]:
        """Serialize to the canonical Stage 1 flat sequence."""

        if self.task == "ASR":
            return [
                SPECIAL_TOKENS["ASR"],
                SPECIAL_TOKENS["SOS"],
                *self.speech_token_ids,
                SPECIAL_TOKENS["EOS"],
                SPECIAL_TOKENS["SOT"],
                *self.text_token_ids,
                SPECIAL_TOKENS["EOT"],
            ]

        return [
            SPECIAL_TOKENS["TTS"],
            SPECIAL_TOKENS["SOT"],
            *self.text_token_ids,
            SPECIAL_TOKENS["EOT"],
            SPECIAL_TOKENS["SOS"],
            *self.speech_token_ids,
            SPECIAL_TOKENS["EOS"],
        ]

    def to_training_token_ids(self) -> list[int]:
        """Serialize to a pure integer sequence for training."""

        if self.task == "ASR":
            return [
                SPECIAL_TOKEN_IDS["ASR"],
                SPECIAL_TOKEN_IDS["SOS"],
                *self.speech_token_ids,
                SPECIAL_TOKEN_IDS["EOS"],
                SPECIAL_TOKEN_IDS["SOT"],
                *self.text_token_ids,
                SPECIAL_TOKEN_IDS["EOT"],
            ]

        return [
            SPECIAL_TOKEN_IDS["TTS"],
            SPECIAL_TOKEN_IDS["SOT"],
            *self.text_token_ids,
            SPECIAL_TOKEN_IDS["EOT"],
            SPECIAL_TOKEN_IDS["SOS"],
            *self.speech_token_ids,
            SPECIAL_TOKEN_IDS["EOS"],
        ]

    def to_json(self) -> str:
        payload = asdict(self)
        payload["sequence"] = self.to_training_sequence()
        payload["sequence_token_ids"] = self.to_training_token_ids()
        return json.dumps(payload, ensure_ascii=True)


@dataclass(slots=True)
class LibriSpeechUtterance:
    """Minimal normalized LibriSpeech utterance metadata."""

    utterance_id: str
    speaker_id: str
    chapter_id: str
    transcript: str
    audio_path: str


def text_to_mock_ids(text: str, *, vocab_offset: int = 10_000) -> list[int]:
    """Cheap deterministic text-token stand-in until the real text tokenizer is wired."""

    words = [word for word in text.strip().split() if word]
    if not words:
        return [vocab_offset]
    return [vocab_offset + (sum(ord(ch) for ch in word) % 2048) for word in words]


def speech_to_vocab_ids(raw_speech_token_ids: list[int]) -> list[int]:
    """Map codec-local speech codes into the global model vocabulary."""

    return [SPEECH_VOCAB_OFFSET + token_id for token_id in raw_speech_token_ids]


def write_jsonl(path: str | Path, examples: Iterable[Stage1AlignmentExample]) -> int:
    """Write examples as JSONL and return the number of rows written."""

    count = 0
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(example.to_json())
            handle.write("\n")
            count += 1
    return count


def deterministic_split(items: list[LibriSpeechUtterance], *, val_ratio: float, seed: int) -> tuple[list[LibriSpeechUtterance], list[LibriSpeechUtterance]]:
    """Split normalized utterances into train/val partitions."""

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio))
    val_items = shuffled[:val_count]
    train_items = shuffled[val_count:]
    return train_items, val_items
