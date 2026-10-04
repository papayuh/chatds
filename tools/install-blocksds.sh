#!/bin/bash
# Install Wonderful toolchain + BlocksDS into a user-writable prefix (no sudo).
set -euo pipefail
export WONDERFUL_TOOLCHAIN="${WONDERFUL_TOOLCHAIN:-$HOME/opt/wonderful}"
export PATH="$WONDERFUL_TOOLCHAIN/bin:$PATH"
mkdir -p "$WONDERFUL_TOOLCHAIN"
if [ ! -x "$WONDERFUL_TOOLCHAIN/bin/wf-pacman" ]; then
  tmp=$(mktemp)
  curl -L -o "$tmp" https://wonderful.asie.pl/bootstrap/wf-bootstrap-x86_64.tar.gz
  tar -xzf "$tmp" -C "$WONDERFUL_TOOLCHAIN"/
  rm -f "$tmp"
fi
wf-pacman -Syu --noconfirm wf-tools
wf-config repo enable blocksds
wf-pacman -Syu --noconfirm
wf-pacman -S --noconfirm blocksds-toolchain
echo "BLOCKSDS=$WONDERFUL_TOOLCHAIN/thirdparty/blocksds/core"
ls "$WONDERFUL_TOOLCHAIN/thirdparty/blocksds/core" && echo INSTALL_OK
