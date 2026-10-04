#!/usr/bin/env bash
# Headless melonDS test of the DS ROM. Plays a UI script (default
# tests/scenarios/walk.txt) inside the emulator, pulls the screen dumps off the
# emulated SD card and compares every one, byte for byte, with what the host
# simulator renders for the same script.
#
# Input is injected at the UI's frame input (the script driver), not through
# melonDS's mouse, because a headless melonDS has no pointer. Everything below
# that line runs for real: ROM, libnds video setup, VRAM writes, DLDI/FAT.
#
# melonDS only runs with QT_QPA_PLATFORM=offscreen (never a window), under a
# timeout, with its own config dir so the shared melonDS.toml is untouched.
# Usage: tests/ds-ui-melon.sh [script] [timeout_s]
set -euo pipefail
cd "$(dirname "$0")/.."
HERE=$PWD
SCRIPT=$(realpath "${1:-tests/scenarios/walk.txt}")
TIMEOUT=${2:-180}
REPO=$(cd ../.. && pwd)
WORK=$HERE/build/melon
mkdir -p "$WORK"/{config/melonDS,home,cache,host,device}

# Serialize only this checkout's build/artifacts; no shared home-specific lock.
exec 9>"$WORK/melon.lock"
flock -w 300 9 || { echo 'Emulator busy' >&2; exit 2; }
# shellcheck source=../../../tools/melon-lib.sh
source "$REPO/tools/melon-lib.sh"
source "$REPO/tools/env.sh"

make -s -f ds/Makefile >"$WORK/build.log" 2>&1 || { cat "$WORK/build.log" >&2; exit 2; }
make -s sim >>"$WORK/build.log" 2>&1 || { cat "$WORK/build.log" >&2; exit 2; }

ROM=$WORK/chatds-ui.nds
IMG=$WORK/sd.img
cp chatds-ui.nds "$ROM"
rm -rf "$WORK/host" "$WORK/device"; mkdir -p "$WORK/host" "$WORK/device"
./build/ui_sim "$SCRIPT" "$WORK/host"

melon_prepare_image_io
melon_build_image "$IMG" "$SCRIPT:chatds-ui/script.txt"
cat > "$WORK/config/melonDS/melonDS.toml" <<TOML
LimitFPS = false

[Emu]
DirectBoot = true
ExternalBIOSEnable = false

[JIT]
Enable = true

[DLDI]
Enable = true
ImagePath = "$IMG"
ImageSize = 0
ReadOnly = false
FolderSync = false
FolderPath = ""
TOML

trap 'melon_kill "$ROM"' EXIT
melon_kill "$ROM"
echo "[ds-ui-melon] launching melonDS offscreen: $(basename "$SCRIPT")"
flatpak run --filesystem="$WORK" --filesystem="$(dirname "$MELON_IMAGE_IO")" \
    --env=QT_QPA_PLATFORM=offscreen --env=LD_PRELOAD="$MELON_IMAGE_IO" \
    --env=MELON_UNBUFFERED_IMAGE="$IMG" --command=sh "$MELON_APPID" \
    -c 'export XDG_CONFIG_HOME="$1" HOME="$2" XDG_CACHE_HOME="$3" QT_QPA_PLATFORM=offscreen; exec melonDS "$4"' \
    sh "$WORK/config" "$WORK/home" "$WORK/cache" "$ROM" >"$WORK/emu.log" 2>&1 9>&- &

got=0
for ((t = 0; t < TIMEOUT; t++)); do
    sleep 1
    if mcopy -o -i "$IMG" ::/chatds-ui/result.txt "$WORK/result.txt" 2>/dev/null &&
        [[ $(tail -n 1 "$WORK/result.txt") == status=* ]]; then
        got=1
        break
    fi
    [[ -n "$(melon_pids "$ROM")" ]] || { echo '[ds-ui-melon] melonDS exited early' >&2; break; }
done
melon_kill "$ROM"
if ((!got)) || [[ $(tail -n 1 "$WORK/result.txt") != status=OK ]]; then
    echo "FAIL: no passing result.txt within ${TIMEOUT}s (see $WORK/emu.log)" >&2
    tail -n 20 "$WORK/emu.log" >&2 || true
    exit 1
fi
grep -Fq "[melon-image] unbuffered: $IMG" "$WORK/emu.log" ||
    { echo "FAIL: image I/O shim was not active (see $WORK/emu.log)" >&2; exit 1; }

mcopy -s -o -i "$IMG" ::/chatds-ui "$WORK/device/"
cat "$WORK/result.txt"
fail=0 n=0
for h in "$WORK"/host/*; do
    f=$(basename "$h")
    [[ $f == *.ppm ]] && continue
    n=$((n + 1))
    if ! cmp -s "$h" "$WORK/device/chatds-ui/$f"; then
        echo "MISMATCH: $f" >&2
        fail=1
    fi
done
((n > 0)) || { echo 'FAIL: host produced no dumps' >&2; exit 1; }
((fail == 0)) || { echo "FAIL: device and host screens differ (host $WORK/host, device $WORK/device)" >&2; exit 1; }
echo "PASS: $n files identical between melonDS and host ($(basename "$SCRIPT"))"
