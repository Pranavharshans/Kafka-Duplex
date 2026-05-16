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
from .audio import AudioBuffer, AudioChunk, chunk_audio, concatenate, generate_sine_wave, load_wav_mono_pcm16, save_wav_mono_pcm16
from .codec import CodecStats, CosyVoiceCodec, MockSpeechCodec, create_codec, timed_roundtrip
from .codec import CosyVoiceRuntimeConfig, timed_encode
from .schema import (
    ACTION_TOKEN_ORDER,
    SPECIAL_TOKENS,
    AgentTargetChunk,
    DuplexAction,
    DuplexChunkRecord,
    UserInputChunk,
)
from .stage1 import LibriSpeechUtterance, Stage1AlignmentExample, deterministic_split, text_to_mock_ids, write_jsonl

__all__ = [
    "CANONICAL_CHUNK_SIZE",
    "Token",
    "ACTION_TOKEN_ORDER",
    "SPECIAL_TOKENS",
    "AudioBuffer",
    "AudioChunk",
    "AgentTargetChunk",
    "CodecStats",
    "CosyVoiceRuntimeConfig",
    "CosyVoiceCodec",
    "DuplexAction",
    "DuplexChunkRecord",
    "LibriSpeechUtterance",
    "MockSpeechCodec",
    "Stage1AlignmentExample",
    "UserInputChunk",
    "build_backchannel_chunk",
    "build_chunk",
    "build_listen_chunk",
    "build_speak_chunk",
    "chunk_audio",
    "concatenate",
    "create_codec",
    "deterministic_split",
    "generate_sine_wave",
    "load_wav_mono_pcm16",
    "parse_chunk",
    "save_wav_mono_pcm16",
    "text_to_mock_ids",
    "timed_roundtrip",
    "timed_encode",
    "write_jsonl",
]
