#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/datasets/rice}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" src/train.py --model_name resnet50 --save_name resnet50_rice_imbal --phase train --data_path "$DATA_PATH"
"$PYTHON_BIN" src/train.py --model_name resnet50 --save_name resnet50_rice_bal --phase train_bal --data_path "$DATA_PATH"