"""Upload a built Stage 1 dataset directory to a Hugging Face dataset repo."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a built Stage 1 dataset directory to Hugging Face.")
    parser.add_argument("--repo-id", required=True, help="Target Hugging Face dataset repo, e.g. Praha-Labs/kafka-duplex-stage1-trainclean460-real.")
    parser.add_argument("--dataset-root", required=True, help="Directory containing train.stage1.jsonl and val.stage1.jsonl.")
    parser.add_argument("--token", default="", help="Optional Hugging Face token. Falls back to HF_TOKEN if omitted.")
    parser.add_argument("--commit-message", default="Upload real Stage 1 dataset", help="HF commit message.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to upload the Stage 1 dataset.") from exc

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    required = ["train.stage1.jsonl", "val.stage1.jsonl"]
    missing = [name for name in required if not (dataset_root / name).exists()]
    if missing:
        raise RuntimeError(f"Dataset root is missing required files: {', '.join(missing)}")

    token = args.token or None
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(dataset_root),
        commit_message=args.commit_message,
    )
    print(f"hf_dataset_upload_complete repo={args.repo_id} dataset_root={dataset_root}", flush=True)


if __name__ == "__main__":
    main()
