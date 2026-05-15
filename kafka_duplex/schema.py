"""Canonical token and chunk schema for Kafka-Duplex.

This module is intentionally small and dependency-light so it can be reused by
data preparation, training, inference, and evaluation code without drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DuplexAction(str, Enum):
    """Chunk-level action predicted after user speech is consumed."""

    LISTEN = "LISTEN"
    SPEAK = "SPEAK"
    BACKCHANNEL = "BACKCHANNEL"


SPECIAL_TOKENS = {
    "CHUNK": "[CHUNK]",
    "ASR": "[ASR]",
    "TTS": "[TTS]",
    "SOS": "[SOS]",
    "EOS": "[EOS]",
    "SOT": "[SOT]",
    "EOT": "[EOT]",
    "SIL_SPEECH": "<sil_sp>",
    "SIL_TEXT": "<sil_txt>",
    "LISTEN": "[LISTEN]",
    "SPEAK": "[SPEAK]",
    "BACKCHANNEL": "[BACKCHANNEL]",
}

ACTION_TOKEN_ORDER = {
    DuplexAction.LISTEN: SPECIAL_TOKENS["LISTEN"],
    DuplexAction.SPEAK: SPECIAL_TOKENS["SPEAK"],
    DuplexAction.BACKCHANNEL: SPECIAL_TOKENS["BACKCHANNEL"],
}


@dataclass(slots=True)
class UserInputChunk:
    """User-side 200ms input chunk represented as codec tokens."""

    speech_tokens: list[int]

    def __post_init__(self) -> None:
        if len(self.speech_tokens) != 10:
            raise ValueError("UserInputChunk requires exactly 10 speech tokens.")


@dataclass(slots=True)
class AgentTargetChunk:
    """Agent-side target tokens for one 200ms chunk."""

    action: DuplexAction
    text_tokens: list[int] = field(default_factory=list)
    speech_tokens: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.action == DuplexAction.LISTEN:
            if self.text_tokens:
                raise ValueError("LISTEN chunks must not contain text tokens.")
            if len(self.speech_tokens) not in (0, 10):
                raise ValueError("LISTEN chunks must contain 0 or 10 silence speech tokens.")
            return

        if self.action == DuplexAction.SPEAK:
            if len(self.text_tokens) != 2:
                raise ValueError("SPEAK chunks require exactly 2 text tokens.")
            if len(self.speech_tokens) != 10:
                raise ValueError("SPEAK chunks require exactly 10 speech tokens.")
            return

        if self.action == DuplexAction.BACKCHANNEL:
            if self.text_tokens:
                raise ValueError("BACKCHANNEL chunks must not contain text tokens.")
            if len(self.speech_tokens) != 10:
                raise ValueError("BACKCHANNEL chunks require exactly 10 speech tokens.")


@dataclass(slots=True)
class DuplexChunkRecord:
    """Canonical chunk-level record shared across pipeline stages."""

    user: UserInputChunk
    agent: AgentTargetChunk
    user_is_speaking: bool = True
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    @property
    def action_token(self) -> str:
        return ACTION_TOKEN_ORDER[self.agent.action]
