"""Kafka-Duplex core package."""

from .chunks import (
    CANONICAL_CHUNK_SIZE,
    Token,
    build_backchannel_chunk,
    build_chunk,
    build_listen_chunk,
    build_speak_chunk,
    parse_chunk,
)
from .schema import (
    ACTION_TOKEN_ORDER,
    SPECIAL_TOKENS,
    AgentTargetChunk,
    DuplexAction,
    DuplexChunkRecord,
    UserInputChunk,
)

__all__ = [
    "CANONICAL_CHUNK_SIZE",
    "Token",
    "ACTION_TOKEN_ORDER",
    "SPECIAL_TOKENS",
    "AgentTargetChunk",
    "DuplexAction",
    "DuplexChunkRecord",
    "UserInputChunk",
    "build_backchannel_chunk",
    "build_chunk",
    "build_listen_chunk",
    "build_speak_chunk",
    "parse_chunk",
]
