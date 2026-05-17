"""Upload a Stage 1 baseline checkpoint package to Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a Stage 1 baseline checkpoint package to Hugging Face.")
    parser.add_argument("--repo-id", required=True, help="Target Hugging Face model repo, e.g. Praha-Labs/kafka-duplex-stage1-lfm.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file to upload.")
    parser.add_argument("--config", required=True, help="Runtime/config JSON used for the checkpoint.")
    parser.add_argument("--train-log", default="", help="Optional train.log path.")
    parser.add_argument("--eval-dir", default="", help="Optional eval directory to include.")
    parser.add_argument("--token", default="", help="Optional Hugging Face token. Falls back to HF_TOKEN if omitted.")
    parser.add_argument("--commit-message", default="Upload Stage 1 baseline checkpoint", help="HF commit message.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to upload the Stage 1 baseline.") from exc

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    config = Path(args.config).expanduser().resolve()
    train_log = Path(args.train_log).expanduser().resolve() if args.train_log else None
    eval_dir = Path(args.eval_dir).expanduser().resolve() if args.eval_dir else None
    token = args.token or None

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=False, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="stage1_hf_upload_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        shutil.copy2(checkpoint, temp_dir / checkpoint.name)
        shutil.copy2(config, temp_dir / "runtime_config.json")
        if train_log is not None and train_log.exists():
            shutil.copy2(train_log, temp_dir / "train.log")
        if eval_dir is not None and eval_dir.exists():
            target_eval_dir = temp_dir / "eval"
            shutil.copytree(eval_dir, target_eval_dir, dirs_exist_ok=True)

        readme = temp_dir / "README.md"
        readme.write_text(
            "\n".join(
                [
                    f"# {args.repo_id}",
                    "",
                    "Stage 1 baseline checkpoint package for Kafka-Duplex.",
                    "",
                    f"- checkpoint: `{checkpoint.name}`",
                    f"- config: `runtime_config.json`",
                    "- includes eval artifacts when provided",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=str(temp_dir),
            commit_message=args.commit_message,
        )

    print(f"hf_model_upload_complete repo={args.repo_id} checkpoint={checkpoint}", flush=True)


if __name__ == "__main__":
    main()
