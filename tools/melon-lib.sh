# tools/melon-lib.sh — shared melonDS-via-flatpak DLDI harness mechanics.
# Source this; it defines functions, doesn't run anything on its own.
#
# Shared image-build / launch / poll / kill helpers.
# melonDS 1.1 buffers the final FATStorage write in stdio.
# Our process-local fdopen shim makes ONLY the DLDI image unbuffered so
# mtools can observe completion before a forced emulator exit;
# DLDI.ImageSize in melonDS.toml is an *index*, not bytes — always write 0
# ("auto") so a prebuilt image is never reformatted; flatpak execs into
# bwrap, so kill by matching the ROM's absolute path in the process argv,
# not by app id.

MELON_APPID="net.kuribo64.melonDS"
MELON_CONFIG_DIR="$HOME/.var/app/$MELON_APPID/config/melonDS"
MELON_CONFIG="$MELON_CONFIG_DIR/melonDS.toml"
MELON_TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MELON_IMAGE_IO="$MELON_TOOLS_DIR/build/melon-image-io.so"
MELON_EMU_LOG="${MELON_EMU_LOG:-${TMPDIR:-/tmp}/melon-emu.log}"
MELON_MCOPY_LOG="${MELON_MCOPY_LOG:-${TMPDIR:-/tmp}/melon-mcopy.log}"
MELON_MKFS_LOG="${MELON_MKFS_LOG:-${TMPDIR:-/tmp}/melon-mkfs.log}"

melon_prepare_image_io() {
    mkdir -p "$(dirname "$MELON_IMAGE_IO")" || return 1
    if [[ ! -f "$MELON_IMAGE_IO" || "$MELON_TOOLS_DIR/melon-image-io.c" -nt "$MELON_IMAGE_IO" ]]; then
        cc -std=c99 -O2 -Wall -Wextra -Werror -shared -fPIC \
            "$MELON_TOOLS_DIR/melon-image-io.c" -ldl -o "$MELON_IMAGE_IO" || return 1
    fi
    flatpak run --filesystem="$(dirname "$MELON_IMAGE_IO")" --command=sh \
        "$MELON_APPID" -c 'test -r "$1"' sh "$MELON_IMAGE_IO"
}

melon_log() { echo "[melon-lib] $*" >&2; }

# Match actual emulator processes and an EXACT argv element, not shell command
# substrings: `pkill -f ROM` can kill a harness whose own argv names that ROM.
melon_pids() {
    local rom="$1" pid
    for pid in $(pgrep -u "$(id -u)" -x melonDS 2>/dev/null || true); do
        if grep -Fzxq -- "$rom" "/proc/$pid/cmdline" 2>/dev/null; then
            printf '%s\n' "$pid"
        fi
    done
    return 0
}

# melon_kill ROM
melon_kill() {
    local rom="$1" pids
    pids=$(melon_pids "$rom")
    [[ -n "$pids" ]] || return 0
    kill -TERM -- $pids >/dev/null 2>&1 || true
    for _ in 1 2 3 4 5; do
        [[ -n "$(melon_pids "$rom")" ]] || return 0
        sleep 1
    done
    pids=$(melon_pids "$rom")
    [[ -z "$pids" ]] || kill -KILL -- $pids >/dev/null 2>&1 || true
}

# melon_build_image SD_IMAGE "host_path:image_path" [...]
# image_path is the mtools path relative to the FAT root (e.g. chatds/model.bin).
melon_build_image() {
    local sd_image="$1"; shift
    local total=0 pair host img dir
    for pair in "$@"; do
        host="${pair%%:*}"
        [ -f "$host" ] || { melon_log "missing file: $host"; return 1; }
        total=$((total + $(stat -c%s "$host")))
    done
    local img_kib=$(( (total + 16*1024*1024 + 1023) / 1024 ))
    # FAT32 needs >= ~33 MiB of clusters; mkfs.vfat -F 32 refuses smaller. Floor at 64 MiB.
    [ "$img_kib" -lt 65536 ] && img_kib=65536
    rm -f "$sd_image"
    mkfs.vfat -F 32 -n SDCARD -C "$sd_image" "$img_kib" >"$MELON_MKFS_LOG" 2>&1 \
        || { melon_log "mkfs.vfat failed (see $MELON_MKFS_LOG)"; return 1; }
    for pair in "$@"; do
        host="${pair%%:*}"
        img="${pair#*:}"
        dir="$(dirname "$img")"
        if [ "$dir" != "." ]; then
            mmd -i "$sd_image" "::/$dir" >/dev/null 2>&1 || true # may pre-exist
        fi
        mcopy -i "$sd_image" "$host" "::/$img" \
            || { melon_log "mcopy $host -> $img failed"; return 1; }
    done
}

# melon_write_config SD_IMAGE
melon_write_config() {
    local sd_image="$1"
    mkdir -p "$MELON_CONFIG_DIR"
    cat > "$MELON_CONFIG" <<EOF
LimitFPS = false

[Emu]
DirectBoot = true
ExternalBIOSEnable = false

[JIT]
Enable = true

[DLDI]
Enable = true
ImagePath = "$sd_image"
ImageSize = 0
ReadOnly = false
FolderSync = false
FolderPath = ""
EOF
}

# melon_launch_and_poll ROM SD_IMAGE RESULT_IMAGE_PATH RESULT_HOST_PATH [timeout_s] [interval_s] [expected_status]
# Polls via mcopy for the last-line status (OK by default), process exit or
# timeout. Fault tests may request FAIL, ignoring a previous result's OK. Leaves
# the pulled file at RESULT_HOST_PATH. Echoes 1 (found) or 0 (not found) and
# always kills melonDS before returning.
melon_launch_and_poll() {
    local rom="$1" sd_image="$2" result_img="$3" result_host="$4"
    local timeout="${5:-180}" interval="${6:-3}" expected="${7:-OK}"
    local elapsed=0 got=0
    [[ "$expected" == OK || "$expected" == FAIL ]] || { melon_log 'expected status must be OK or FAIL'; echo 0; return; }

    if ! melon_prepare_image_io; then
        melon_log "image I/O shim build/access failed (needs host cc and a home/repo path)"
        echo 0
        return
    fi
    sd_image=$(realpath -e "$sd_image") || { echo 0; return; }
    melon_kill "$rom"
    sleep 1
    melon_log "launching melonDS: $rom"
    flatpak run --env=LD_PRELOAD="$MELON_IMAGE_IO" --env=MELON_UNBUFFERED_IMAGE="$sd_image" \
        "$MELON_APPID" "$rom" >"$MELON_EMU_LOG" 2>&1 &

    while [ "$elapsed" -lt "$timeout" ]; do
        sleep "$interval"
        elapsed=$((elapsed + interval))
        # A direct writer can expose a partial file before its final marker.
        # Overwrite the previous poll's copy, and never mistake answer text
        # containing "status=OK" for the trailing protocol status.
        if mcopy -o -i "$sd_image" "::/$result_img" "$result_host" 2>"$MELON_MCOPY_LOG"; then
            local status
            status=$(tail -n 1 "$result_host")
            if [[ "$status" == "status=$expected" ]]; then
                got=1
                melon_log "$result_img completed after ${elapsed}s"
                break
            elif [[ "$status" == "status=FAIL" ]]; then
                melon_log "$result_img reported FAIL after ${elapsed}s (see $result_host)"
                break
            fi
        fi
        if [[ -z "$(melon_pids "$rom")" ]]; then
            melon_log "melonDS process exited early after ${elapsed}s (crash?)"
            break
        fi
    done

    melon_kill "$rom"
    if ! grep -Fq "[melon-image] unbuffered: $sd_image" "$MELON_EMU_LOG"; then
        melon_log "image I/O shim was not activated; see $MELON_EMU_LOG"
        got=0
    fi
    echo "$got"
}
