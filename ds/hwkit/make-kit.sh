#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Build the clean product and stage files; never write to a physical card.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE=assistant MODEL="" TOK="" KB="" KB_ATTRIBUTION="" STEPS="" PROMPT=""
OUT="$ROOT/ds/hwkit/sdcard"
while (($#)); do
    case "$1" in
        --mode|--model|--tokenizer|--kb|--kb-attribution|--steps|--prompt|--out)
            (($# >= 2)) || { echo "Missing value: $1" >&2; exit 2; }
            case "$1" in
                --mode) MODE=$2;; --model) MODEL=$2;; --tokenizer) TOK=$2;; --kb) KB=$2;;
                --kb-attribution) KB_ATTRIBUTION=$2;;
                --steps) STEPS=$2;; --prompt) PROMPT=$2;; --out) OUT=$2;;
            esac; shift 2;;
        *) echo 'Usage: make-kit.sh --model PATH --tokenizer PATH [--mode assistant|story] [--kb PATH --kb-attribution PATH] [--steps N] [--prompt TEXT] [--out DIR]' >&2; exit 2;;
    esac
done
[[ -n "$MODEL" && -n "$TOK" ]] || { echo 'Supply both --model and --tokenizer; no implicit model fallback.' >&2; exit 2; }
case "$MODE" in
    story) STEPS=${STEPS:-200}; PROMPT=${PROMPT:-Once upon a time}; INSTRUCT=0;;
    assistant) STEPS=${STEPS:-64}; PROMPT=${PROMPT:-whats 38 times 47}; INSTRUCT=1;;
    *) echo 'Mode must be story or assistant' >&2; exit 2;;
esac
[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] && ((STEPS <= 256)) || { echo 'steps must be 1..256' >&2; exit 2; }
MODEL=$(realpath -e "$MODEL"); TOK=$(realpath -e "$TOK")
if [[ -n "$KB" ]]; then
    ((INSTRUCT)) || { echo '--kb only makes sense in assistant mode' >&2; exit 2; }
    KB=$(realpath -e "$KB"); [[ -f "$KB" ]] || exit 2
fi
if [[ -n "$KB_ATTRIBUTION" ]]; then
    [[ -n "$KB" ]] || { echo '--kb-attribution requires --kb' >&2; exit 2; }
    KB_ATTRIBUTION=$(realpath -e "$KB_ATTRIBUTION")
fi
[[ -f "$MODEL" && -f "$TOK" ]] || exit 2
python3 - "$PROMPT" <<'PY'
import sys
p = sys.argv[1]
if not p or not p.isascii() or len(p) > 127 or any(c in p for c in '\n\r\0'):
    raise SystemExit('prompt must be one nonempty ASCII line, at most 127 bytes')
PY
OUT=$(realpath -m "$OUT")
mkdir -p "$(dirname "$OUT")"
[[ ! -e "$OUT.previous" ]] || { echo "Move $OUT.previous aside first; refusing to overwrite backup" >&2; exit 2; }
STAGE=$(mktemp -d "$(dirname "$OUT")/.kit-stage.XXXXXX")
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/chatds" "$STAGE/.host"
source "$ROOT/tools/env.sh"
make -C "$ROOT/clean/product" all host
cp "$ROOT/clean/product/build/chatds.nds" "$STAGE/chatds.nds"
cp "$MODEL" "$STAGE/chatds/model.bin"
cp "$TOK" "$STAGE/chatds/tok.bin"
[[ -z "$KB" ]] || cp "$KB" "$STAGE/chatds/kb.bin"
[[ -z "$KB_ATTRIBUTION" ]] || cp "$KB_ATTRIBUTION" "$STAGE/chatds/kb-ATTRIBUTION.txt"
cp "$ROOT/LICENSE" "$STAGE/LICENSE"
cp "$ROOT/THIRD_PARTY_NOTICES.md" "$STAGE/THIRD_PARTY_NOTICES.md"
printf 'steps=%s\nkbd=1\ninstruct=%s\nprompt=%s\n' "$STEPS" "$INSTRUCT" "$PROMPT" > "$STAGE/chatds/run.txt"
# Clean host/device smoke expectation, distinct from the frozen legacy goldens.
"$ROOT/clean/product/build/chatds-host" "$MODEL" "$TOK" "${KB:--}" \
    "$STAGE/chatds/run.txt" "$STAGE/.host/"
cp "$STAGE/.host/ids.txt" "$STAGE/.host-golden.ids"
cp "$STAGE/.host/out.txt" "$STAGE/.host-golden.txt"
tail -n 6 "$STAGE/.host/out.txt" > "$STAGE/.host-golden.ids.meta"
rm -r "$STAGE/.host"
python3 - "$STAGE" "$MODE" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
files = sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file())
manifest = {'mode': sys.argv[2], 'engine': 'clean-dsq8-kv16',
            'quality': 'experimental; not a capability certification',
            'sha256': {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in files}}
(root / 'kit.json').write_text(json.dumps(manifest, indent=2) + '\n')
PY
if [[ -e "$OUT" ]]; then mv "$OUT" "$OUT.previous"; fi
mv "$STAGE" "$OUT"
trap - EXIT
printf '\nStaged clean %s kit: %s\nPrevious kit preserved at %s.previous (if present).\nNo physical SD card touched.\n' "$MODE" "$OUT" "$OUT"
