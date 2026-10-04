# DS lite hardware boundaries

The target ARM946E-S runs at 67 MHz (ARMv5TE), without FPU or SIMD. Main RAM is
4 MiB, shared by application code, runtime, heap and reservations. ITCM is
32 KiB, DTCM 16 KiB; video owns part of VRAM. The clean core uses ARM halfword
MACs and software float only for initialization/normalization coefficients.
[GBATEK](https://mgba-emu.github.io/gbatek/) documents the CPU/memory map.

The clean product reserves 10 KiB DTCM for bounded session workspace and keeps
KV on the heap. Heap break is only a footprint proxy: it does not certify stack
headroom. See [engine arithmetic/memory contract](../clean/ARITHMETIC.md).

Historical DS lite + DSpico observations in `ds/hw-results/` are data, not
calibration source. File-level sequential bandwidth was about 1826 KiB/s;
random 512-byte file reads were around 1 ms. Old burst-read/maximum-malloc
probes were misleading. Do not treat those old model timings or projections as
measurements of the current product. The old calibration source was removed
because its timer had been copied from an unlicensed upstream runtime.

## current product hardware gate

Build the exact release kit, then test on DS lite + DSpico:

- type/backspace/send, repeated questions and scrolling;
- cancellation between inference steps, clean result publication and card removal;
- actual generated IDs, calculator result and fact retrieval;
- boot/model-load latency separately from first-token and total inference time;
- heap/stack headroom, including stack canary, on the full-context path.

This gate is pending. DSi, 3DS and other carts are untested.

## emulator boundary

melonDS is a functional test environment, not physical timing evidence.
`DLDI.ImageSize = 0` selects the existing image size; other values can reformat
an image. `tools/engine_golden.py` uses private image/config/ROM copies, offscreen
Qt and exact-argv process cleanup. `tools/melon-image-io.c` disables buffering
only for the selected FAT image so completion is observable before shutdown.
It is a host shim, not a firmware workaround.
