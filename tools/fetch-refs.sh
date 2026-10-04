#!/bin/bash
# Clone reference repos (not vendored; gitignored). Re-run anytime.
set -e
cd "$(dirname "$0")/../reference" 2>/dev/null || { mkdir -p "$(dirname "$0")/../reference"; cd "$(dirname "$0")/../reference"; }
# Optional MIT training dependency only; never fetch an unlicensed runtime.
for r in RileyGreiff/ds-llm; do
  d=$(basename "$r"); [ -d "$d" ] || git clone -q --depth 1 "https://github.com/$r"
done
ls
