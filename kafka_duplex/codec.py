"""Codec abstraction layer for Kafka-Duplex Phase 2."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Protocol

from .audio import AudioBuffer, AudioChunk, DEFAULT_CHUNK_MS, DEFAULT_SAMPLE_RATE, generate_sine_wave


TOKENS_PER_CHUNK = 5


@dataclass(slots=True)
class CodecStats:
    """Per-operation latency and shape information."""

    encode_ms: float
    decode_ms: float
    tokens_per_chunk: int


class SpeechCodec(Protocol):
    """Minimal codec adapter interface."""

    name: str

    def encode_chunk(self, chunk: AudioChunk) -> list[int]:
        """Encode one 200ms chunk into speech tokens."""

    def decode_chunk(self, tokens: list[int], *, sample_rate: int = DEFAULT_SAMPLE_RATE) -> AudioBuffer:
        """Decode speech tokens back into audio."""


class MockSpeechCodec:
    """Lightweight deterministic codec used until the real stack is available."""

    name = "mock"

    def encode_chunk(self, chunk: AudioChunk) -> list[int]:
        if not chunk.samples:
            return [0] * TOKENS_PER_CHUNK

        window = max(1, len(chunk.samples) // TOKENS_PER_CHUNK)
        tokens: list[int] = []
        for index in range(TOKENS_PER_CHUNK):
            start = index * window
            end = len(chunk.samples) if index == TOKENS_PER_CHUNK - 1 else min(len(chunk.samples), start + window)
            slice_samples = chunk.samples[start:end]
            avg_abs = sum(abs(sample) for sample in slice_samples) / max(1, len(slice_samples))
            tokens.append(int(avg_abs) % 4096)
        return tokens

    def decode_chunk(self, tokens: list[int], *, sample_rate: int = DEFAULT_SAMPLE_RATE) -> AudioBuffer:
        if len(tokens) != TOKENS_PER_CHUNK:
            raise ValueError(f"MockSpeechCodec expects exactly {TOKENS_PER_CHUNK} tokens.")

        # Map token values to simple sine frequencies to produce an audible artifact.
        base_duration = DEFAULT_CHUNK_MS // TOKENS_PER_CHUNK
        buffers = []
        for token in tokens:
            frequency = 220.0 + float(token % 440)
            buffers.append(
                generate_sine_wave(
                    frequency_hz=frequency,
                    duration_ms=base_duration,
                    sample_rate=sample_rate,
                    amplitude=0.12,
                )
            )
        samples: list[int] = []
        for buffer in buffers:
            samples.extend(buffer.samples)
        return AudioBuffer(samples=samples, sample_rate=sample_rate)


class CosyVoiceCodec:
    """Deferred adapter for the real CosyVoice stack.

    This class intentionally fails fast with clear setup requirements until the
    actual dependency is installed and wired in.
    """

    name = "cosyvoice"

    def __init__(self) -> None:
        raise RuntimeError(
            "CosyVoiceCodec is not available yet. Install the CosyVoice runtime and "
            "replace this stub with the actual model loading and encode/decode calls."
        )

    def encode_chunk(self, chunk: AudioChunk) -> list[int]:
        raise NotImplementedError

    def decode_chunk(self, tokens: list[int], *, sample_rate: int = DEFAULT_SAMPLE_RATE) -> AudioBuffer:
        raise NotImplementedError


def create_codec(name: str) -> SpeechCodec:
    """Instantiate a codec adapter by name."""

    normalized = name.lower().strip()
    if normalized == "mock":
        return MockSpeechCodec()
    if normalized == "cosyvoice":
        return CosyVoiceCodec()
    raise ValueError(f"Unsupported codec: {name!r}")


def timed_roundtrip(codec: SpeechCodec, chunk: AudioChunk) -> tuple[list[int], AudioBuffer, CodecStats]:
    """Encode and decode one chunk while measuring latency."""

    encode_start = time.perf_counter()
    tokens = codec.encode_chunk(chunk)
    encode_ms = (time.perf_counter() - encode_start) * 1000.0

    decode_start = time.perf_counter()
    decoded = codec.decode_chunk(tokens, sample_rate=chunk.sample_rate)
    decode_ms = (time.perf_counter() - decode_start) * 1000.0

    stats = CodecStats(
        encode_ms=encode_ms,
        decode_ms=decode_ms,
        tokens_per_chunk=len(tokens),
    )
    return tokens, decoded, stats
