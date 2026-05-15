"""Probe the real CosyVoice speech-token extractor on a WAV file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafka_duplex import chunk_audio, create_codec, load_wav_mono_pcm16, timed_encode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe CosyVoice token counts on chunked audio.")
    parser.add_argument("--input-wav", required=True, help="Path to a mono PCM16 WAV file.")
    parser.add_argument("--chunk-ms", type=int, default=200, help="Chunk size in milliseconds.")
    parser.add_argument("--max-chunks", type=int, default=0, help="Optional limit on inspected chunks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    codec = create_codec("cosyvoice")
    audio = load_wav_mono_pcm16(args.input_wav)
    chunks = chunk_audio(audio, chunk_ms=args.chunk_ms)
    inspected = chunks if args.max_chunks <= 0 else chunks[: args.max_chunks]

    token_lengths: list[int] = []
    encode_latencies: list[float] = []

    print(
        " ".join(
            [
                "probe",
                f"codec={codec.name}",
                f"source={args.input_wav}",
                f"sample_rate={audio.sample_rate}",
                f"duration_ms={audio.duration_ms:.1f}",
                f"total_chunks={len(chunks)}",
            ]
        )
    )

    for chunk in inspected:
        tokens, encode_ms = timed_encode(codec, chunk)
        token_lengths.append(len(tokens))
        encode_latencies.append(encode_ms)
        print(
            " ".join(
                [
                    f"chunk={chunk.index:02d}",
                    f"start_ms={chunk.start_ms:.1f}",
                    f"duration_ms={chunk.duration_ms:.1f}",
                    f"token_count={len(tokens)}",
                    f"tokens={tokens}",
                    f"encode_ms={encode_ms:.2f}",
                ]
            )
        )

    avg_tokens = sum(token_lengths) / len(token_lengths)
    avg_encode_ms = sum(encode_latencies) / len(encode_latencies)
    tokens_per_sec = avg_tokens / (args.chunk_ms / 1000.0)
    print(
        " ".join(
            [
                "summary",
                f"avg_tokens_per_chunk={avg_tokens:.2f}",
                f"tokens_per_second={tokens_per_sec:.2f}",
                f"avg_encode_ms={avg_encode_ms:.2f}",
                f"chunk_ms={args.chunk_ms}",
            ]
        )
    )


if __name__ == "__main__":
    main()
