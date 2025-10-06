#!/usr/bin/env bash
set -Euo pipefail

CFG_DIR=${CFG_DIR:-./configs/results}
LOG_DIR=${LOG_DIR:-./logs/sweeps}

CONFIGS=(
  "${CFG_DIR}/bg2.yaml"
  "${CFG_DIR}/no-bg2.yaml"
)

mkdir -p "$LOG_DIR"

ok=0
fail=0

for cfg in "${CONFIGS[@]}"; do
  ts="$(date '+%Y%m%d-%H%M%S')"
  name="$(basename "$cfg" .yaml)"
  log="${LOG_DIR}/${name}-${ts}.log"

  echo "[sweep] === ${cfg} ===" | tee -a "$log"

  if [[ ! -f "$cfg" ]]; then
    echo "[sweep][warn] Config not found: $cfg — skipping." | tee -a "$log"
    ((fail++))
    continue
  fi

  cmd=(python -u -m src.engine.sweep
       --config "$cfg"
       --only_full
       --auto-summary
       --analysis
       --eval-winners
       --test_split_file AUTO
       --skip-existing)

  # Show the exact command
  printf '[sweep] cmd: %q ' "${cmd[@]}" | tee -a "$log"
  echo | tee -a "$log"

  # Run; tee both stdout and stderr to the log, preserve Python's exit code
  if "${cmd[@]}" 2>&1 | tee -a "$log"; then
    rc=0
  else
    rc=${PIPESTATUS[0]}
  fi

  if (( rc == 0 )); then
    ((ok++))
    echo "[sweep] OK: ${cfg}" | tee -a "$log"
  else
    ((fail++))
    echo "[sweep][warn] FAILED for ${cfg} (exit ${rc}). Continuing..." | tee -a "$log"
  fi

  echo | tee -a "$log"
done

echo "[sweep] Completed. success=${ok} failure=${fail}"

# Exit non-zero if any failed; override with: EXIT_NONZERO_ON_FAIL=0 ./run_sweeps.sh
EXIT_NONZERO_ON_FAIL=${EXIT_NONZERO_ON_FAIL:-1}
if (( fail > 0 )) && (( EXIT_NONZERO_ON_FAIL )); then
  exit 1
fi
