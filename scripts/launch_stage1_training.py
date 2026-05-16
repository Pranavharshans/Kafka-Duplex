"""Stage 1 training launcher scaffold.

This script records the intended training parameters and validates that the
expected manifests exist. Actual model training should be wired to the future
training implementation once the Stage 1 model class lands.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and print Stage 1 training launch parameters.")
    parser.add_argument("--config", default="configs/stage1_alignment.json", help="Path to Stage 1 config JSON.")
    parser.add_argument(
        "--output-dir",
        default="/workspace/kafka_duplex_runs/stage1",
        help="Directory for checkpoints, eval outputs, and TensorBoard event files.",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Allow Stage 1 training to continue on CPU when CUDA is unavailable.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print parameters and dataset checks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    import json

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    dataset = config["dataset"]
    train_manifest = Path(dataset["train_manifest"]).resolve()
    val_manifest = Path(dataset["val_manifest"]).resolve()

    print(
        " ".join(
            [
                "stage1_train_config",
                f"epochs={config['optimization']['epochs']}",
                f"lr={config['optimization']['learning_rate']}",
                f"warmup_steps={config['optimization']['warmup_steps']}",
                f"batch={config['optimization']['effective_batch_size']}",
                f"eval_every={config['evaluation']['eval_every_steps']}",
                f"checkpoint_every={config['evaluation']['checkpoint_every_steps']}",
                f"train_manifest_exists={train_manifest.exists()}",
                f"val_manifest_exists={val_manifest.exists()}",
            ]
        )
    )

    if args.dry_run:
        return

    from training.stage1_train import Stage1RunConfig, run_stage1_training

    run_stage1_training(
        Stage1RunConfig(
            config_path=str(config_path),
            output_dir=args.output_dir,
            allow_cpu_fallback=args.allow_cpu_fallback,
        )
    )


if __name__ == "__main__":
    main()
