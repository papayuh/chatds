#!/usr/bin/env bash
# Evaluate an exported DSQ8 model with the clean host product.
# Usage: run_chatds_suites.sh RUN_DIR TAG [suite ...]
set -euo pipefail
cd "$(dirname "$0")/.."
R=${1:?run directory}; TAG=${2:?output tag}; shift 2
if (($# == 0)); then set -- chatds-acceptance-30 suite-v1.1; fi
make host
for suite in "$@"; do
    args=()
    [[ ! -f corpus/kb/out/simplewiki.kb ]] || args+=(--chatds-kb corpus/kb/out/simplewiki.kb)
    python3 eval/run_eval.py --suite "eval/$suite.jsonl" \
        --tokenizer "$R/tokenizer.bin" --model "$R/${MODEL:-model_dsq8.tie}" \
        --out "$R/eval-$suite-$TAG.jsonl" "${args[@]}"
done
