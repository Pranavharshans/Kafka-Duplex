"""Codec abstraction layer for Kafka-Duplex Phase 2."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any, Protocol

from .audio import AudioBuffer, AudioChunk, DEFAULT_CHUNK_MS, DEFAULT_SAMPLE_RATE, generate_sine_wave


TOKENS_PER_CHUNK = 10
DEFAULT_COSYVOICE_SAMPLE_RATE = 16_000


@dataclass(slots=True)
class CosyVoiceRuntimeConfig:
    """Runtime configuration for the optional CosyVoice adapter."""

    model_dir: str
    repo_dir: str | None = None
    load_jit: bool = False
    load_trt: bool = False
    load_vllm: bool = False
    fp16: bool = False
    sample_rate: int = DEFAULT_COSYVOICE_SAMPLE_RATE

    @classmethod
    def from_env(cls) -> "CosyVoiceRuntimeConfig":
        model_dir = os.environ.get("KAFKA_DUPLEX_COSYVOICE_MODEL_DIR", "").strip()
        if not model_dir:
            raise RuntimeError(
                "CosyVoice model path is not configured. Set "
                "KAFKA_DUPLEX_COSYVOICE_MODEL_DIR to the pretrained model directory."
            )

        repo_dir = os.environ.get("KAFKA_DUPLEX_COSYVOICE_REPO_DIR", "").strip() or None
        return cls(
            model_dir=model_dir,
            repo_dir=repo_dir,
            load_jit=os.environ.get("KAFKA_DUPLEX_COSYVOICE_LOAD_JIT", "0") == "1",
            load_trt=os.environ.get("KAFKA_DUPLEX_COSYVOICE_LOAD_TRT", "0") == "1",
            load_vllm=os.environ.get("KAFKA_DUPLEX_COSYVOICE_LOAD_VLLM", "0") == "1",
            fp16=os.environ.get("KAFKA_DUPLEX_COSYVOICE_FP16", "0") == "1",
            sample_rate=int(os.environ.get("KAFKA_DUPLEX_COSYVOICE_SAMPLE_RATE", str(DEFAULT_COSYVOICE_SAMPLE_RATE))),
        )


@dataclass(slots=True)
class CodecStats:
    """Per-operation latency and shape information."""

    encode_ms: float
    decode_ms: float
    tokens_per_chunk: int


class SpeechCodec(Protocol):
    """Minimal codec adapter interface."""

    name: str
    supports_decode: bool

    def encode_chunk(self, chunk: AudioChunk) -> list[int]:
        """Encode one 200ms chunk into speech tokens."""

    def decode_chunk(self, tokens: list[int], *, sample_rate: int = DEFAULT_SAMPLE_RATE) -> AudioBuffer:
        """Decode speech tokens back into audio."""


class MockSpeechCodec:
    """Lightweight deterministic codec used until the real stack is available."""

    name = "mock"
    supports_decode = True

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
    """Lazy adapter for the real CosyVoice speech-token extractor."""

    name = "cosyvoice"
    supports_decode = False

    def __init__(self, config: CosyVoiceRuntimeConfig | None = None) -> None:
        self.config = config or CosyVoiceRuntimeConfig.from_env()
        self._frontend: Any | None = None

    def _ensure_frontend(self) -> Any:
        if self._frontend is not None:
            return self._frontend

        if self.config.repo_dir:
            repo_path = Path(self.config.repo_dir).expanduser().resolve()
            if not repo_path.exists():
                raise RuntimeError(f"Configured CosyVoice repo does not exist: {repo_path}")

            import sys

            repo_str = str(repo_path)
            if repo_str not in sys.path:
                sys.path.insert(0, repo_str)

        try:
            from cosyvoice.cli.cosyvoice import CosyVoice  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "CosyVoice imports failed. Install the CosyVoice runtime and, if needed, "
                "set KAFKA_DUPLEX_COSYVOICE_REPO_DIR to the local CosyVoice checkout."
            ) from exc

        model_dir = Path(self.config.model_dir).expanduser().resolve()
        if not model_dir.exists():
            raise RuntimeError(f"Configured CosyVoice model directory does not exist: {model_dir}")

        model = CosyVoice(
            str(model_dir),
            load_jit=self.config.load_jit,
            load_trt=self.config.load_trt,
            load_vllm=self.config.load_vllm,
            fp16=self.config.fp16,
        )
        self._frontend = model.frontend
        return self._frontend

    def _chunk_to_tensor(self, chunk: AudioChunk) -> Any:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("CosyVoiceCodec requires torch to be installed.") from exc

        samples = chunk.samples
        if chunk.sample_rate != self.config.sample_rate:
            try:
                import torchaudio.functional as F
            except ImportError as exc:
                raise RuntimeError(
                    "Sample rate conversion requires torchaudio when the chunk sample rate "
                    f"({chunk.sample_rate}) differs from the configured CosyVoice rate "
                    f"({self.config.sample_rate})."
                ) from exc

            waveform = torch.tensor(samples, dtype=torch.float32).unsqueeze(0) / 32768.0
            waveform = F.resample(waveform, orig_freq=chunk.sample_rate, new_freq=self.config.sample_rate)
            return waveform

        return torch.tensor(samples, dtype=torch.float32).unsqueeze(0) / 32768.0

    def encode_chunk(self, chunk: AudioChunk) -> list[int]:
        frontend = self._ensure_frontend()
        speech = self._chunk_to_tensor(chunk)
        token_tensor = frontend._extract_speech_token(speech)

        try:
            flattened = token_tensor.detach().cpu().reshape(-1).tolist()
        except AttributeError:
            flattened = list(token_tensor.reshape(-1))

        return [int(token) for token in flattened]

    def decode_chunk(self, tokens: list[int], *, sample_rate: int = DEFAULT_SAMPLE_RATE) -> AudioBuffer:
        raise RuntimeError(
            "CosyVoiceCodec decode_chunk is not wired yet. The current adapter supports real "
            "speech-token extraction only; token-to-wave decode should be validated on the GPU box."
        )


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


def timed_encode(codec: SpeechCodec, chunk: AudioChunk) -> tuple[list[int], float]:
    """Encode one chunk while measuring latency."""

    encode_start = time.perf_counter()
    tokens = codec.encode_chunk(chunk)
    encode_ms = (time.perf_counter() - encode_start) * 1000.0
    return tokens, encode_ms
