"""CLI entrypoint for the mocked Kafka-Duplex Phase 1 loop."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.mock_duplex import MockDuplexConfig, MockDuplexRunner, format_event


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the mocked Kafka-Duplex loop.")
    parser.add_argument("--chunks", type=int, default=12, help="Number of 200ms chunks to simulate.")
    parser.add_argument(
        "--backchannel-every",
        type=int,
        default=4,
        help="Emit a BACKCHANNEL on every Nth user-speaking chunk.",
    )
    parser.add_argument(
        "--user-pause-every",
        type=int,
        default=3,
        help="Make the user pause on every Nth chunk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MockDuplexConfig(
        total_chunks=args.chunks,
        backchannel_every=args.backchannel_every,
        user_pause_every=args.user_pause_every,
    )
    result = MockDuplexRunner(config).run()

    for event in result.events:
        print(format_event(event))
    print(f"context_tokens={len(result.rolling_context)}")


if __name__ == "__main__":
    main()
