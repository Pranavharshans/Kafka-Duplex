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
    "AudioBuffer",
    "AudioChunk",
    "AgentTargetChunk",
    "CodecStats",
    "CosyVoiceCodec",
    "DuplexAction",
    "DuplexChunkRecord",
    "MockSpeechCodec",
    "UserInputChunk",
    "build_backchannel_chunk",
    "build_chunk",
    "build_listen_chunk",
    "build_speak_chunk",
    "chunk_audio",
    "concatenate",
    "create_codec",
    "generate_sine_wave",
    "load_wav_mono_pcm16",
    "parse_chunk",
    "save_wav_mono_pcm16",
    "timed_roundtrip",
]
