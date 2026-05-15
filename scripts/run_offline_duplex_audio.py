"""Run the Phase 2 offline prerecorded-audio duplex harness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.offline_duplex_audio import OfflineDuplexRunner
from kafka_duplex import create_codec, generate_sine_wave, load_wav_mono_pcm16, save_wav_mono_pcm16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline duplex control flow over prerecorded audio.")
    parser.add_argument("--codec", default="mock", help="Codec adapter to use: mock or cosyvoice.")
    parser.add_argument("--input-wav", help="Path to mono PCM16 WAV input.")
    parser.add_argument("--chunk-ms", type=int, default=200, help="Chunk size in milliseconds.")
    parser.add_argument("--write-agent-wav", help="Optional path for decoded agent output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    codec = create_codec(args.codec)
    if args.input_wav:
        audio = load_wav_mono_pcm16(args.input_wav)
        source = args.input_wav
    else:
        audio = generate_sine_wave(frequency_hz=330.0, duration_ms=1600)
        source = "generated_sine_wave"

    result = OfflineDuplexRunner(codec, chunk_ms=args.chunk_ms).run(audio)
    print(f"codec={codec.name} source={source} chunks={len(result.events)} context_tokens={len(result.rolling_context)}")
    for event in result.events:
        print(
            " ".join(
                [
                    f"chunk={event.chunk_index:02d}",
                    f"rms={event.user_rms:.1f}",
                    f"action={event.action.value}",
                    f"user_tokens={event.user_tokens}",
                    f"agent_tokens={event.agent_tokens}",
                ]
            )
        )

    if args.write_agent_wav and result.decoded_agent_audio is not None:
        save_wav_mono_pcm16(args.write_agent_wav, result.decoded_agent_audio)
        print(f"wrote_agent_wav={args.write_agent_wav}")


if __name__ == "__main__":
    main()
