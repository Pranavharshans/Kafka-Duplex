"""Tokenizer-backed Stage 1 vocabulary interface."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .schema import SPECIAL_TOKEN_ORDER, SPECIAL_TOKENS, SPEECH_VOCAB_OFFSET, SPEECH_VOCAB_SIZE, SPECIAL_TOKEN_IDS


@dataclass(slots=True)
class Stage1TokenInterface:
    text_tokenizer_name: str
    text_vocab_size: int
    special_token_ids: dict[str, int]
    speech_vocab_offset: int
    speech_vocab_size: int
    total_vocab_size: int

    def encode_text(self, text: str) -> list[int]:
        tokenizer = get_hf_tokenizer(self.text_tokenizer_name)
        return list(tokenizer.encode(text, add_special_tokens=False))

    def decode_text(self, token_ids: list[int]) -> str:
        if self.text_tokenizer_name == "mock_text_ids_until_real_tokenizer_is_wired":
            return " ".join(str(token_id) for token_id in token_ids)
        tokenizer = get_hf_tokenizer(self.text_tokenizer_name)
        return str(tokenizer.decode(token_ids, skip_special_tokens=True)).strip()

    def speech_to_vocab_ids(self, raw_speech_token_ids: list[int]) -> list[int]:
        return [self.speech_vocab_offset + token_id for token_id in raw_speech_token_ids]

    def vocab_to_raw_speech_ids(self, vocab_speech_token_ids: list[int]) -> list[int]:
        return [token_id - self.speech_vocab_offset for token_id in vocab_speech_token_ids]

    def to_metadata(self) -> dict[str, object]:
        return {
            "text_tokenizer_name": self.text_tokenizer_name,
            "text_vocab_size": self.text_vocab_size,
            "special_token_ids": self.special_token_ids,
            "speech_vocab_offset": self.speech_vocab_offset,
            "speech_vocab_size": self.speech_vocab_size,
            "total_vocab_size": self.total_vocab_size,
        }


def legacy_stage1_token_interface() -> Stage1TokenInterface:
    return Stage1TokenInterface(
        text_tokenizer_name="mock_text_ids_until_real_tokenizer_is_wired",
        text_vocab_size=SPEECH_VOCAB_OFFSET,
        special_token_ids=dict(SPECIAL_TOKEN_IDS),
        speech_vocab_offset=SPEECH_VOCAB_OFFSET,
        speech_vocab_size=SPEECH_VOCAB_SIZE,
        total_vocab_size=max(SPECIAL_TOKEN_IDS.values()) + 1,
    )


def build_hf_stage1_token_interface(model_name: str, *, speech_vocab_size: int = SPEECH_VOCAB_SIZE) -> Stage1TokenInterface:
    tokenizer = get_hf_tokenizer(model_name)
    text_vocab_size = int(len(tokenizer))
    special_start = text_vocab_size
    special_token_ids = {
        token_name: special_start + index for index, token_name in enumerate(SPECIAL_TOKEN_ORDER)
    }
    speech_vocab_offset = special_start + len(SPECIAL_TOKEN_ORDER)
    total_vocab_size = speech_vocab_offset + speech_vocab_size
    return Stage1TokenInterface(
        text_tokenizer_name=model_name,
        text_vocab_size=text_vocab_size,
        special_token_ids=special_token_ids,
        speech_vocab_offset=speech_vocab_offset,
        speech_vocab_size=speech_vocab_size,
        total_vocab_size=total_vocab_size,
    )


def special_tokens_in_order() -> list[str]:
    return [SPECIAL_TOKENS[name] for name in SPECIAL_TOKEN_ORDER]


@lru_cache(maxsize=8)
def get_hf_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("transformers is required to use the Stage 1 HF tokenizer interface.") from exc

    return AutoTokenizer.from_pretrained(model_name, use_fast=True)
