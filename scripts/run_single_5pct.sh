#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DATA_ROOT="${DATA_ROOT:-/datasets/rice}"
METHODS="${METHODS:-random}"
SEED="${SEED:-42}"
# METHODS accepts a space-separated list, e.g.: METHODS="entropy hybrid"
# shellcheck disable=SC2086
python src/train_al.py --data-root "$DATA_ROOT" \
  --output runs/active_learning --single-pct 5 \
  --single-initial-pct 2 --methods $METHODS --seeds "$SEED" \
  --backend timm --pretrained --epochs 50 --patience 10 --batch-size 32