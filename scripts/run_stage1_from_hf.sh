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

USE_SYSTEM_PYTHON="${USE_SYSTEM_PYTHON:-0}"

mkdir -p "$(dirname "$REPO_DIR")" "$(dirname "$VENV_DIR")" "$(dirname "$DATASET_ROOT")" "$(dirname "$RUN_ROOT")"
cd "$(dirname "$REPO_DIR")"
if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone https://github.com/Pranavharshans/Kafka-Duplex.git "$REPO_DIR"
fi

cd "$REPO_DIR"
git pull --ff-only

if [[ "$USE_SYSTEM_PYTHON" != "1" && ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

if [[ "$USE_SYSTEM_PYTHON" == "1" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-python3}"
else
  source "$VENV_DIR/bin/activate"
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
"$PYTHON_BIN" -m pip install soundfile tensorboard huggingface_hub

"$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA preflight failed: torch.cuda.is_available() is False")
print(torch.__version__)
print(torch.cuda.get_device_name(0))
print(torch.tensor([1.0], device="cuda"))
PY

mkdir -p "$DATASET_ROOT" "$RUN_ROOT"
HF_FETCH_ARGS=(scripts/fetch_stage1_hf_dataset.py --repo-id "$HF_REPO_ID" --output-root "$DATASET_ROOT")
if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_FETCH_ARGS+=(--token "$HF_TOKEN")
fi
"$PYTHON_BIN" "${HF_FETCH_ARGS[@]}"

if [[ "$USE_SYSTEM_PYTHON" == "1" ]]; then
  NOHUP_CMD="cd '$REPO_DIR' && '$PYTHON_BIN' scripts/launch_stage1_training.py --config '$CONFIG_PATH' --output-dir '$RUN_ROOT' > '$RUN_ROOT/train.log' 2>&1"
else
  NOHUP_CMD="source '$VENV_DIR/bin/activate' && cd '$REPO_DIR' && '$PYTHON_BIN' scripts/launch_stage1_training.py --config '$CONFIG_PATH' --output-dir '$RUN_ROOT' > '$RUN_ROOT/train.log' 2>&1"
fi

nohup bash -lc "$NOHUP_CMD" > "$RUN_ROOT/driver.log" 2>&1 < /dev/null &

echo "stage1_run_started config=$CONFIG_PATH run_root=$RUN_ROOT dataset_root=$DATASET_ROOT python_bin=$PYTHON_BIN use_system_python=$USE_SYSTEM_PYTHON"
echo "tail -f $RUN_ROOT/train.log"
