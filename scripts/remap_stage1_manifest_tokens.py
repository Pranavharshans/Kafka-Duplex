"""Remap Stage 1 manifests from the bootstrap token layout to a tokenizer-backed interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafka_duplex.schema import SPEECH_VOCAB_OFFSET
from kafka_duplex.stage1 import Stage1AlignmentExample
from kafka_duplex.token_interface import build_hf_stage1_token_interface


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remap bootstrap Stage 1 manifests to a real tokenizer/vocab interface.")
    parser.add_argument("--input-root", required=True, help="Directory containing train.stage1.jsonl and val.stage1.jsonl.")
    parser.add_argument("--output-root", required=True, help="Output directory for remapped manifests.")
    parser.add_argument("--hf-model-name", default="LiquidAI/LFM2.5-350M", help="HF tokenizer/model name.")
    return parser.parse_args()


def remap_manifest(input_path: Path, output_path: Path, hf_model_name: str) -> int:
    token_interface = build_hf_stage1_token_interface(hf_model_name)
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            legacy_speech_ids = [int(token_id) for token_id in row["speech_token_ids"]]
            raw_speech_ids = [token_id - SPEECH_VOCAB_OFFSET for token_id in legacy_speech_ids]
            example = Stage1AlignmentExample(
                task=str(row["task"]),
                example_id=str(row["example_id"]),
                transcript=str(row["transcript"]),
                text_token_ids=token_interface.encode_text(str(row["transcript"])),
                speech_token_ids=token_interface.speech_to_vocab_ids(raw_speech_ids),
                source_audio_path=str(row["source_audio_path"]),
                speaker_id=str(row["speaker_id"]),
                chapter_id=str(row["chapter_id"]),
                utterance_id=str(row["utterance_id"]),
                metadata={
                    **dict(row.get("metadata", {})),
                    "token_interface": token_interface.to_metadata(),
                    "remapped_from": str(input_path),
                },
            )
            handle.write(example.to_json(special_token_ids=token_interface.special_token_ids))
            handle.write("\n")
            written += 1
    return written


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    for name in ("train.stage1.jsonl", "val.stage1.jsonl"):
        count = remap_manifest(input_root / name, output_root / name, args.hf_model_name)
        print(
            " ".join(
                [
                    "stage1_manifest_remap",
                    f"file={name}",
                    f"examples={count}",
                    f"hf_model_name={args.hf_model_name}",
                    f"output_path={output_root / name}",
                ]
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
