"""Download a built Stage 1 dataset snapshot from Hugging Face into /workspace paths."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch a Stage 1 manifest dataset from Hugging Face.")
    parser.add_argument(
        "--repo-id",
        default="Praha-Labs/kafka-duplex-stage1-trainclean460",
        help="Hugging Face dataset repo containing the built Stage 1 manifests.",
    )
    parser.add_argument(
        "--output-root",
        default="/workspace/kafka_duplex_data/stage1_trainclean460",
        help="Directory where the downloaded manifest files should be placed.",
    )
    parser.add_argument(
        "--token",
        default="",
        help="Optional Hugging Face token. Falls back to HF_TOKEN if omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to fetch the Stage 1 dataset.") from exc

    token = args.token or None
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    files = [
        "train.stage1.jsonl",
        "val.stage1.jsonl",
        "stage1_trainclean460.json",
        "build.log",
        "README.md",
    ]

    for filename in files:
        downloaded = Path(
            hf_hub_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                filename=filename,
                token=token,
            )
        )
        target = output_root / filename
        shutil.copy2(downloaded, target)
        print(f"hf_dataset_file repo={args.repo_id} file={filename} target={target}", flush=True)

    print(f"hf_dataset_ready repo={args.repo_id} output_root={output_root}", flush=True)


if __name__ == "__main__":
    main()
