"""Chunk serialization and parsing utilities for Kafka-Duplex."""

from __future__ import annotations

from typing import TypeAlias

from .schema import (
    ACTION_TOKEN_ORDER,
    SPECIAL_TOKENS,
    AgentTargetChunk,
    DuplexAction,
    DuplexChunkRecord,
    UserInputChunk,
)

Token: TypeAlias = int | str

USER_SPEECH_TOKENS_PER_CHUNK = 10
AGENT_TEXT_TOKENS_PER_SPEAK_CHUNK = 2
AGENT_SPEECH_TOKENS_PER_CHUNK = 10
CANONICAL_CHUNK_SIZE = 24

_ACTION_FROM_TOKEN = {value: key for key, value in ACTION_TOKEN_ORDER.items()}


def build_listen_chunk(
    user_tokens: list[int],
    *,
    user_is_speaking: bool = True,
    metadata: dict[str, str | int | float | bool] | None = None,
) -> DuplexChunkRecord:
    """Create a semantic LISTEN chunk record."""

    return DuplexChunkRecord(
        user=UserInputChunk(user_tokens),
        agent=AgentTargetChunk(action=DuplexAction.LISTEN, speech_tokens=[]),
        user_is_speaking=user_is_speaking,
        metadata=dict(metadata or {}),
    )


def build_speak_chunk(
    user_tokens: list[int],
    *,
    text_tokens: list[int],
    speech_tokens: list[int],
    user_is_speaking: bool = False,
    metadata: dict[str, str | int | float | bool] | None = None,
) -> DuplexChunkRecord:
    """Create a semantic SPEAK chunk record."""

    return DuplexChunkRecord(
        user=UserInputChunk(user_tokens),
        agent=AgentTargetChunk(
            action=DuplexAction.SPEAK,
            text_tokens=text_tokens,
            speech_tokens=speech_tokens,
        ),
        user_is_speaking=user_is_speaking,
        metadata=dict(metadata or {}),
    )


def build_backchannel_chunk(
    user_tokens: list[int],
    *,
    speech_tokens: list[int],
    user_is_speaking: bool = True,
    metadata: dict[str, str | int | float | bool] | None = None,
) -> DuplexChunkRecord:
    """Create a semantic BACKCHANNEL chunk record."""

    return DuplexChunkRecord(
        user=UserInputChunk(user_tokens),
        agent=AgentTargetChunk(
            action=DuplexAction.BACKCHANNEL,
            speech_tokens=speech_tokens,
        ),
        user_is_speaking=user_is_speaking,
        metadata=dict(metadata or {}),
    )


def build_chunk(record: DuplexChunkRecord) -> list[Token]:
    """Serialize a chunk record into the canonical 14-token layout."""

    if record.agent.action == DuplexAction.LISTEN:
        return (
            [SPECIAL_TOKENS["CHUNK"]]
            + record.user.speech_tokens
            + [SPECIAL_TOKENS["LISTEN"]]
            + _silence_text_tokens()
            + _silence_speech_tokens()
        )

    if record.agent.action == DuplexAction.SPEAK:
        return (
            [SPECIAL_TOKENS["CHUNK"]]
            + record.user.speech_tokens
            + [SPECIAL_TOKENS["SPEAK"]]
            + record.agent.text_tokens
            + record.agent.speech_tokens
        )

    return (
        [SPECIAL_TOKENS["CHUNK"]]
        + record.user.speech_tokens
        + [SPECIAL_TOKENS["BACKCHANNEL"]]
        + _silence_text_tokens()
        + record.agent.speech_tokens
    )


def parse_chunk(tokens: list[Token]) -> DuplexChunkRecord:
    """Parse a canonical chunk token layout back into a semantic record."""

    if len(tokens) != CANONICAL_CHUNK_SIZE:
        raise ValueError(
            f"Chunk must contain exactly {CANONICAL_CHUNK_SIZE} tokens, got {len(tokens)}."
        )
    if tokens[0] != SPECIAL_TOKENS["CHUNK"]:
        raise ValueError("Chunk must start with the [CHUNK] token.")

    user_end = 1 + USER_SPEECH_TOKENS_PER_CHUNK
    action_index = user_end
    payload_start = action_index + 1

    user_tokens = _expect_int_tokens(tokens[1:user_end], name="user speech")
    action_token = tokens[action_index]
    try:
        action = _ACTION_FROM_TOKEN[action_token]
    except KeyError as exc:
        raise ValueError(f"Unknown action token: {action_token!r}") from exc

    payload = tokens[payload_start:]
    if action == DuplexAction.LISTEN:
        _validate_listen_payload(payload)
        return build_listen_chunk(user_tokens)

    if action == DuplexAction.SPEAK:
        text_tokens = _expect_int_tokens(payload[:2], name="agent text")
        speech_tokens = _expect_int_tokens(payload[2:], name="agent speech")
        return build_speak_chunk(
            user_tokens,
            text_tokens=text_tokens,
            speech_tokens=speech_tokens,
        )

    _validate_backchannel_payload(payload[:2])
    speech_tokens = _expect_int_tokens(payload[2:], name="backchannel speech")
    return build_backchannel_chunk(user_tokens, speech_tokens=speech_tokens)


def render_chunk(tokens: list[Token]) -> str:
    """Render a chunk as a compact log-friendly string."""

    return " ".join(str(token) for token in tokens)


def _silence_text_tokens() -> list[str]:
    return [SPECIAL_TOKENS["SIL_TEXT"]] * AGENT_TEXT_TOKENS_PER_SPEAK_CHUNK


def _silence_speech_tokens() -> list[str]:
    return [SPECIAL_TOKENS["SIL_SPEECH"]] * AGENT_SPEECH_TOKENS_PER_CHUNK


def _expect_int_tokens(tokens: list[Token], *, name: str) -> list[int]:
    if not all(isinstance(token, int) for token in tokens):
        raise ValueError(f"{name} tokens must all be integers.")
    return [int(token) for token in tokens]


def _validate_listen_payload(payload: list[Token]) -> None:
    if payload[:2] != _silence_text_tokens():
        raise ValueError("LISTEN chunks must contain <sil_txt> placeholders.")
    if payload[2:] != _silence_speech_tokens():
        raise ValueError("LISTEN chunks must contain <sil_sp> placeholders.")


def _validate_backchannel_payload(payload: list[Token]) -> None:
    if payload != _silence_text_tokens():
        raise ValueError("BACKCHANNEL chunks must use <sil_txt> placeholders for text slots.")
