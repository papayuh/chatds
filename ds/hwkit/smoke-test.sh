#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Isolated, offscreen clean host/device smoke; no shared emulator config or lock.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python3 "$ROOT/tools/product_smoke.py" "${1:-$ROOT/ds/hwkit/sdcard}"
