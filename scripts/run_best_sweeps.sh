#!/usr/bin/env bash
set -euo pipefail

CFG_DIR=./configs/reports

CONFIGS=(
  "${CFG_DIR}/report-resnet-flair.yaml"
  "${CFG_DIR}/report-adamw-flair.yaml"
  "${CFG_DIR}/report-resnet-none.yaml"
  "${CFG_DIR}/report-adamw-none.yaml"
  "${CFG_DIR}/report-bg-resnet-flair.yaml"
  "${CFG_DIR}/report-bg-adamw-flair.yaml"
  "${CFG_DIR}/report-bg-resnet-none.yaml"
  "${CFG_DIR}/report-bg-adamw-none.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  echo "[sweep] === ${cfg} ==="
  python -m src.engine.sweep \
    --config "$cfg" \
    --two_stage \
    --auto-summary \
    --analysis \
    --eval-winners \
    --test_split_file AUTO \
    --skip-existing
done
