#!/usr/bin/env bash
set -euo pipefail

if [[ -d "/workspace" ]]; then
  DEFAULT_ROOT="/workspace"
else
  DEFAULT_ROOT="${HOME}"
fi

REPO_DIR="${REPO_DIR:-${DEFAULT_ROOT}/Kafka-Duplex}"
VENV_DIR="${VENV_DIR:-${DEFAULT_ROOT}/venv-kafka-duplex}"
HF_REPO_ID="${HF_REPO_ID:-Praha-Labs/kafka-duplex-stage1-trainclean460}"
DATASET_ROOT="${DATASET_ROOT:-${DEFAULT_ROOT}/kafka_duplex_data/stage1_trainclean460}"
RUN_ROOT="${RUN_ROOT:-${DEFAULT_ROOT}/kafka_duplex_runs/stage1_trainclean460}"
CONFIG_PATH="${CONFIG_PATH:-configs/stage1_trainclean460_40gb.json}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN must be set to fetch the Stage 1 dataset." >&2
  exit 1
fi

mkdir -p "$(dirname "$REPO_DIR")" "$(dirname "$VENV_DIR")" "$(dirname "$DATASET_ROOT")" "$(dirname "$RUN_ROOT")"
cd "$(dirname "$REPO_DIR")"
if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone https://github.com/Pranavharshans/Kafka-Duplex.git "$REPO_DIR"
fi

cd "$REPO_DIR"
git pull --ff-only

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install soundfile tensorboard huggingface_hub

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA preflight failed: torch.cuda.is_available() is False")
print(torch.__version__)
print(torch.cuda.get_device_name(0))
print(torch.tensor([1.0], device="cuda"))
PY

mkdir -p "$DATASET_ROOT" "$RUN_ROOT"
python scripts/fetch_stage1_hf_dataset.py --repo-id "$HF_REPO_ID" --output-root "$DATASET_ROOT"

nohup bash -lc "source '$VENV_DIR/bin/activate' && cd '$REPO_DIR' && python scripts/launch_stage1_training.py --config '$CONFIG_PATH' --output-dir '$RUN_ROOT' > '$RUN_ROOT/train.log' 2>&1" > "$RUN_ROOT/driver.log" 2>&1 < /dev/null &

echo "stage1_run_started config=$CONFIG_PATH run_root=$RUN_ROOT dataset_root=$DATASET_ROOT"
echo "tail -f $RUN_ROOT/train.log"
