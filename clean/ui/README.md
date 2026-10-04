# chatds touch ui

the ds-side ui for the clean engine: on-screen keyboard on the bottom screen, wrapped and scrollable chat log on the top screen, text streaming in as it's produced, a busy spinner with a stop button, and an about screen with the third-party licenses.

it knows nothing about models. it pulls text from a generator (`include/chatds_ui_gen.h`) once per frame. cancel works between steps, since the engine has no cancel of its own.

## layout

```
include/   chatds_ui.h (the ui), chatds_ui_gen.h (answer source), chatds_ui_engine.h (adapter),
           chatds_ui_demo.h (canned generator + scripted input)
src/       ui core. pure c, no heap, no libnds. renders into two 256x192 argb1555 buffers
shell/     demo generator and script driver. shared by the host sim and the rom
adapter/   chatds_ui_engine_gen: answers through ../plumbing (cds_ask_begin + cds_gen_step), one forward per step
host/      ui_sim: runs a script, writes the screens as raw + ppm
ds/        blocksds rom shell
tests/     unit tests (stub engine), scenarios, headless melonDS test
```

the core keeps about 8 KB of state and no framebuffers. on the ds it draws straight into vram with 16-bit stores.

## use

for the real clean-engine product, run root `make` or `ds/hwkit/make-kit.sh`.
that shell is `../product/`; the commands below still build the isolated UI demo,
not the shipped application. the product owns its clock and asset paths.

```
make test       # unit tests, host cc. SAN=1 adds asan/ubsan
make sim        # build/ui_sim SCRIPT OUTDIR
make rom        # chatds-ui.nds, needs blocksds (tools/install-blocksds.sh)
make rom-test   # play tests/scenarios/walk.txt in headless melonDS and compare
```

per frame: fill a `chatds_ui_input` (pen down + x,y, held keys), call `chatds_ui_frame()`, then `chatds_ui_render(ui, top, bottom)`. it only redraws a screen that changed. the busy header samples milliseconds at render time: 100 ms per spinner phase, 1000 ms per elapsed second, even when generation blocks. the default clock uses hardware timers 0/1 (cascaded, bus / 1024) on ARM9 and `CLOCK_MONOTONIC` on host. call `chatds_ui_set_clock(ui, clock_ms, ctx)` before sending to inject a test clock. the DS counter must be sampled at least once per 36 hours to track hardware wrap.

controls: tap keys and buttons (a key fires when the pen lifts on the key it went down on). up/down buttons scroll and repeat when held. d-pad up/down scroll, B stops, start sends (not while about is open), select opens/closes about. a view scrolled up stays put while new text arrives; sending jumps back to the bottom.

## wiring the engine

```c
chatds_ui_engine_gen g;
chatds_ui_engine_gen_init(&g, session, &tokenizer, kb_or_null, instruct, prompt_ids, prompt_ids_cap, max_new_tokens);
chatds_ui_init(&ui, &g.gen);
```

it's `cds_ask` from `clean/plumbing`, split into steps: same `cds_tokenizer`, same kb retrieval and `C:`/`Q:`/`A:` prompt, same greedy loop and stops (eos, bos, budget, full context), same `calc()`. the adapter only streams each piece to the ui as it comes out and appends the calc result (`calc(6*7)` shows ` = 42`). cancel goes through the plumbing's cancel poll. the rom in `ds/` uses the canned generator until the engine and tokenizer land.

## tests

- `tests/test_ui.c`: fake-clock spinner phases and timer text without input frames (including clock wrap and skipped phases), wrap, log eviction, keyboard geometry (every ascii char reachable), typing, streaming, cancel, scroll, about, rendering bounds and dirty tracking, and the engine adapter (with the real plumbing) against `tests/stub_engine.c`.
- `tests/ds-ui-melon.sh`: builds the rom, puts a script on a fat image, runs melonDS with `QT_QPA_PLATFORM=offscreen` under a timeout and its own config dir, pulls the screens off the card and compares them byte for byte with `ui_sim` on the same script.

script playback injects a deterministic millisecond clock (60 Hz script time) on both platforms so byte-for-byte comparisons do not depend on emulator speed or file I/O latency. interactive runs use the hardware clock.

the melonDS run injects touches at the ui's frame input, not through the emulator's mouse: a headless melonDS has no pointer, and `tools/ds_ui.py` needs a real x11 window. everything under that line is real: rom, libnds video setup, vram writes, dldi/fat. what it doesn't cover is `touchRead()` itself and real pen jitter.

reused from the repo: `tools/melon-lib.sh` (image build, shim, process kill), `tools/melon-image-io.c`, `tools/env.sh`, the shared emulator lock.

## limits

- the log keeps `CHATDS_UI_LOG_BYTES` (6 KB) and drops the oldest messages first.
- typed question is capped at `CHATDS_UI_INPUT_MAX` (127) bytes, the plumbing's `CDS_QUESTION_MAX`. ascii only.
- the rom draws into the visible vram, so a big redraw can tear for a frame. add a second page if it shows on hardware.
- no physical ds run yet.
