#!/usr/bin/env bash
# Example: bash scripts/run_one_al.sh entropy 2
#          bash scripts/run_one_al.sh entropy 5 2
#          bash scripts/run_one_al.sh entropy 10 5
#          bash scripts/run_one_al.sh random 5
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
if (( $# < 2 || $# > 3 )); then
  echo "Usage: bash scripts/run_one_al.sh METHOD TARGET_PCT [PREVIOUS_PCT]" >&2
  echo "METHOD: random | least_confidence | entropy | margin | hybrid" >&2
  exit 2
fi
METHOD="$1"
TARGET_PCT="$2"
PREVIOUS_PCT="${3:-}"
DATA_ROOT="${DATA_ROOT:-datasets/rice}"
OUTPUT="${OUTPUT:-runs/active_learning}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WORKERS="${WORKERS:-0}"
ARGS=(
  --data-root "$DATA_ROOT" --output "$OUTPUT"
  --single-pct "$TARGET_PCT" --method "$METHOD" --seed "$SEED"
  --epochs "$EPOCHS" --patience "$PATIENCE"
  --batch-size "$BATCH_SIZE" --workers "$WORKERS"
  --backend timm --pretrained
)
if [[ -n "$PREVIOUS_PCT" ]]; then
  ARGS+=(--continue-from-pct "$PREVIOUS_PCT")
fi
if [[ "${EVAL_TEST:-0}" == "1" ]]; then
  ARGS+=(--single-eval-test)
fi
exec python src/train_al.py "${ARGS[@]}"