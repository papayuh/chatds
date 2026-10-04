#!/usr/bin/env bash
# D13a full-run driver: submit (resumable, picks up wherever results/full.jsonl
# left off) -> postprocess + contamination filter -> one summary file.
# Meant to be launched detached (nohup ./run_full.sh &) so it survives the
# launching session exiting. set -u only, not -e: postprocess/summary must
# still run and report even if submit_deepseek.py exits non-zero (the
# reasoning-guard or cost-ceiling abort).
set -u
cd "$(dirname "$0")"

CONCURRENCY="${CONCURRENCY:-16}"
LOG=results/full-run.log

echo "=== submit start $(date -u +%FT%TZ) concurrency=$CONCURRENCY ===" >>"$LOG"
python3 submit_deepseek.py --all --out results/full.jsonl --concurrency "$CONCURRENCY" >>"$LOG" 2>&1
SUBMIT_RC=$?
echo "=== submit exit code $SUBMIT_RC $(date -u +%FT%TZ) ===" >>"$LOG"

echo "=== postprocess start $(date -u +%FT%TZ) ===" >>"$LOG"
python3 postprocess.py --raw-dir results --requests-dir requests \
  --out results/full-train.jsonl --stats results/full-stats.json >>"$LOG" 2>&1
POSTPROCESS_RC=$?
echo "=== postprocess exit code $POSTPROCESS_RC $(date -u +%FT%TZ) ===" >>"$LOG"

{
  echo "D13a full run summary -- $(date -u +%FT%TZ)"
  echo "submit_deepseek.py exit code: $SUBMIT_RC"
  echo "postprocess.py exit code: $POSTPROCESS_RC"
  if [ -f results/ABORTED.txt ]; then
    echo
    echo "*** RUN WAS ABORTED BY A SAFETY GUARD ***"
    cat results/ABORTED.txt
  fi
  echo
  echo "--- postprocess stats (results/full-stats.json) ---"
  cat results/full-stats.json 2>/dev/null || echo "(missing)"
  echo
  echo "--- spend tracking (batches.json) ---"
  cat batches.json 2>/dev/null || echo "(missing)"
} >results/FULL-RUN-SUMMARY.txt

echo "=== driver done $(date -u +%FT%TZ) ===" >>"$LOG"
