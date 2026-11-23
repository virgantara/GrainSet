#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# Point DATA_ROOT to the folder containing train/ and test/ image directories.
DATA_ROOT="${DATA_ROOT:-/datasets/rice}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python src/train_al.py \
  --data-root "$DATA_ROOT" \
  --output runs/active_learning \
  --methods random least_confidence entropy margin hybrid \
  --seeds 42 123 509 \
  --start-pct 2 --step-pct 2 --max-pct 50 \
  --batch-size 32 --pool-batch-size 128 \
  --epochs 50 --patience 10 --lr 0.0002 \
  --backend timm --pretrained --include-full
python src/plot_al.py --input runs/active_learning
