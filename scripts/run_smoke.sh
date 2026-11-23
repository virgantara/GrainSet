#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DATA_ROOT="${DATA_ROOT:-/datasets/rice}"
python src/train_al.py --data-root "$DATA_ROOT" \
  --output runs/al_smoke --methods random entropy hybrid \
  --seeds 42 --start-pct 2 --step-pct 2 --max-pct 6 \
  --epochs 2 --patience 2 --batch-size 32 --workers 2 \
  --backend timm --pretrained
python src/plot_al.py --input runs/al_smoke
