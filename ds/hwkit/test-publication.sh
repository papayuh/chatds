#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARGS=("${1:?usage: test-publication.sh KIT_DIR [ROM_OVERRIDE]}" --one-token)
[[ -z "${2:-}" ]] || ARGS+=(--rom "$2")
exec python3 "$ROOT/tools/product_smoke.py" "${ARGS[@]}"
