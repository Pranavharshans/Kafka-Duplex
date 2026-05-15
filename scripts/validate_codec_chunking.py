"""Offline Phase 2 codec validation script."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafka_duplex import chunk_audio, create_codec, generate_sine_wave, load_wav_mono_pcm16, save_wav_mono_pcm16, timed_roundtrip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate chunked codec behavior offline.")
    parser.add_argument("--codec", default="mock", help="Codec adapter to use: mock or cosyvoice.")
    parser.add_argument("--input-wav", help="Path to a mono PCM16 WAV file.")
    parser.add_argument("--chunk-ms", type=int, default=200, help="Chunk size in milliseconds.")
    parser.add_argument("--max-chunks", type=int, default=8, help="Maximum chunks to validate.")
    parser.add_argument(
        "--write-decoded",
        help="Optional path to write one decoded chunk WAV for spot-checking.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    codec = create_codec(args.codec)

    if args.input_wav:
        audio = load_wav_mono_pcm16(args.input_wav)
        source = args.input_wav
    else:
        audio = generate_sine_wave(frequency_hz=440.0, duration_ms=max(args.chunk_ms * args.max_chunks, 800))
        source = "generated_sine_wave"

    chunks = chunk_audio(audio, chunk_ms=args.chunk_ms)
    if not chunks:
        raise RuntimeError("No chunks generated from the provided audio.")

    encode_latencies: list[float] = []
    decode_latencies: list[float] = []
    token_lengths: list[int] = []
    decoded_example = None

    print(f"codec={codec.name} source={source} sample_rate={audio.sample_rate} total_chunks={len(chunks)}")
    for chunk in chunks[: args.max_chunks]:
        tokens, decoded, stats = timed_roundtrip(codec, chunk)
        encode_latencies.append(stats.encode_ms)
        decode_latencies.append(stats.decode_ms)
        token_lengths.append(stats.tokens_per_chunk)
        if decoded_example is None:
            decoded_example = decoded

        print(
            " ".join(
                [
                    f"chunk={chunk.index:02d}",
                    f"start_ms={chunk.start_ms:.1f}",
                    f"duration_ms={chunk.duration_ms:.1f}",
                    f"rms={chunk.rms:.1f}",
                    f"silent={chunk.is_silent}",
                    f"tokens={tokens}",
                    f"encode_ms={stats.encode_ms:.2f}",
                    f"decode_ms={stats.decode_ms:.2f}",
                ]
            )
        )

    print(
        "summary",
        f"avg_encode_ms={sum(encode_latencies) / len(encode_latencies):.2f}",
        f"avg_decode_ms={sum(decode_latencies) / len(decode_latencies):.2f}",
        f"tokens_per_chunk={token_lengths}",
    )

    if args.write_decoded and decoded_example is not None:
        save_wav_mono_pcm16(args.write_decoded, decoded_example)
        print(f"wrote_decoded_chunk={args.write_decoded}")


if __name__ == "__main__":
    main()
