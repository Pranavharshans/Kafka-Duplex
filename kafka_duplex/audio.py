"""Audio chunking utilities for Kafka-Duplex.

This module uses only the Python standard library so Phase 2 scaffolding can
run before heavyweight audio dependencies are installed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
import wave


DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_CHUNK_MS = 200
PCM16_MAX = 32_767


@dataclass(slots=True)
class AudioBuffer:
    """In-memory mono PCM16 audio buffer."""

    samples: list[int]
    sample_rate: int = DEFAULT_SAMPLE_RATE

    @property
    def duration_ms(self) -> float:
        return (len(self.samples) / self.sample_rate) * 1000.0


@dataclass(slots=True)
class AudioChunk:
    """Fixed-size audio chunk."""

    index: int
    samples: list[int]
    sample_rate: int
    start_ms: float
    duration_ms: float

    @property
    def rms(self) -> float:
        if not self.samples:
            return 0.0
        mean_square = sum(sample * sample for sample in self.samples) / len(self.samples)
        return math.sqrt(mean_square)

    @property
    def is_silent(self) -> bool:
        return self.rms < 500.0


def load_wav_mono_pcm16(path: str) -> AudioBuffer:
    """Load a mono PCM16 WAV file using the stdlib `wave` module."""

    with wave.open(path, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.getnframes()

        if channels != 1:
            raise ValueError("Only mono WAV files are supported in Phase 2 validation.")
        if sample_width != 2:
            raise ValueError("Only PCM16 WAV files are supported in Phase 2 validation.")

        raw = wav_file.readframes(frames)
        samples = list(struct.unpack(f"<{frames}h", raw))
        return AudioBuffer(samples=samples, sample_rate=sample_rate)


def save_wav_mono_pcm16(path: str, buffer: AudioBuffer) -> None:
    """Save a mono PCM16 WAV file."""

    clipped = [max(-PCM16_MAX, min(PCM16_MAX, sample)) for sample in buffer.samples]
    raw = struct.pack(f"<{len(clipped)}h", *clipped)

    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(buffer.sample_rate)
        wav_file.writeframes(raw)


def chunk_audio(
    buffer: AudioBuffer,
    *,
    chunk_ms: int = DEFAULT_CHUNK_MS,
    pad_final_chunk: bool = True,
) -> list[AudioChunk]:
    """Slice an audio buffer into fixed-size chunks."""

    samples_per_chunk = int(buffer.sample_rate * (chunk_ms / 1000.0))
    if samples_per_chunk <= 0:
        raise ValueError("chunk_ms must produce at least one sample per chunk.")

    chunks: list[AudioChunk] = []
    for index, start in enumerate(range(0, len(buffer.samples), samples_per_chunk)):
        end = start + samples_per_chunk
        chunk_samples = buffer.samples[start:end]
        if len(chunk_samples) < samples_per_chunk and pad_final_chunk:
            chunk_samples = chunk_samples + [0] * (samples_per_chunk - len(chunk_samples))
        duration_ms = (len(chunk_samples) / buffer.sample_rate) * 1000.0
        chunks.append(
            AudioChunk(
                index=index,
                samples=chunk_samples,
                sample_rate=buffer.sample_rate,
                start_ms=(start / buffer.sample_rate) * 1000.0,
                duration_ms=duration_ms,
            )
        )
    return chunks


def generate_sine_wave(
    *,
    frequency_hz: float,
    duration_ms: int,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    amplitude: float = 0.25,
) -> AudioBuffer:
    """Generate a simple mono sine wave."""

    total_samples = int(sample_rate * (duration_ms / 1000.0))
    samples = []
    for index in range(total_samples):
        sample = math.sin(2.0 * math.pi * frequency_hz * (index / sample_rate))
        samples.append(int(sample * amplitude * PCM16_MAX))
    return AudioBuffer(samples=samples, sample_rate=sample_rate)


def concatenate(buffers: list[AudioBuffer]) -> AudioBuffer:
    """Concatenate buffers with the same sample rate."""

    if not buffers:
        return AudioBuffer(samples=[])

    sample_rate = buffers[0].sample_rate
    if any(buffer.sample_rate != sample_rate for buffer in buffers):
        raise ValueError("All buffers must have the same sample rate to concatenate.")

    samples: list[int] = []
    for buffer in buffers:
        samples.extend(buffer.samples)
    return AudioBuffer(samples=samples, sample_rate=sample_rate)
