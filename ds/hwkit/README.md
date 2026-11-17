# clean ChatDS card kit

`make-kit.sh` builds `clean/product/` with int16 KV and stages `chatds.nds` plus
`chatds/` assets. no legacy engine, planner, converter or host oracle participates.
model shape is parsed at load time. the model must be resident DSQ8 and match its
tokenizer; passing these checks does not establish answer quality.

```sh
source tools/env.sh
bash ds/hwkit/make-kit.sh --mode assistant \
  --model /path/to/model.bin --tokenizer /path/to/tok.bin \
  --kb4 /path/to/kb4.bin --kb-attribution /path/to/ATTRIBUTION.txt \
  --prompt 'what is the capital of france' --out build/chatds-kit
```

Alternatively pass `--asset-dir DIR`, containing `model.bin`, `tok.bin`, and
`SHA256SUMS` (standard `sha256sum` format). Both checksums are required and
verified before building or staging; duplicate entries and mismatches fail.
This option is mutually exclusive with `--model`/`--tokenizer`. Assets remain
external build inputs, never compiled into the ROM or committed to source.
The v0.2.0 release ZIP's `chatds/` holds that pair; write a `SHA256SUMS`
for `model.bin` and `tok.bin` beside them. The model and its rebuilt tokenizer
always travel together. Example:

```sh
bash ds/hwkit/make-kit.sh --asset-dir path/to/assets \
  --kb4 path/to/kb4.bin --kb-attribution path/to/kb-ATTRIBUTION.txt \
  --prompt 'whats 38 times 47' --out build/chatds-kit
```

Both model and tokenizer are mandatory. `--mode story` uses the question as raw
input, without retrieval or calculator handling. defaults: assistant, 64 new
tokens; story, 200 new tokens. `--steps` accepts 1..256.
`--kb4` stages ranked KB4 instead of KB3 (`--kb`). either index needs
`--kb-attribution`; `corpus/kb/build_kb4.py` writes `ATTRIBUTION.txt` beside `kb4.bin`.
Assistant kits also stage the three pinned round3 CDSR heads from
`campaign/router/models/`. Missing/corrupt heads fail closed, not back to c14.
Story kits do not need them. Heads load lazily after eligibility checks and
remain validated across deterministic turns; bad files fail when needed.
Fallbacks borrow 49,448 bytes of idle main-RAM KV values; insufficient workspace
fails startup. Dispatch invalidates cached views before generator reset clears
those bytes for prefill. Context and KV precision are unchanged; no additional
primary model/scratch heap allocation. The host product reads the heads from the
model's directory, or from `CHATDS_ROUND3_DIR` (trailing slash) when set.
KB4 takes precedence on the card. Only an absent `kb4.bin` permits KB3 fallback;
a corrupt or unreadable KB4 fails closed. The prompt is one ASCII
line, at most 127 bytes. the keyboard edits ASCII only.

an existing output is moved to `OUTPUT.previous`. if that backup already exists,
the build refuses to overwrite it. no script formats or writes a physical card.
`kit.json` identifies the engine and hashes staged files. `.host-golden.*` are
clean host smoke expectations for this kit, not replacements for the frozen
legacy fixtures.

## testing

C14 and C24 release validation used a private validation harness that is not shipped
here; results are in [PRODUCT-C14](../../docs/PRODUCT-C14.md) and
[PRODUCT-C24](../../docs/PRODUCT-C24.md). Its funnel uses private
FAT images, config, home/cache and ROM copies. It verifies denied X11/Wayland sockets before launching Qt offscreen, denies GPU
access, and compares generated IDs, stop, context, text, calculator output and
path against the clean host product. It screens 60 fixed dev cases, including
multi-turn sessions; for c24 it also compares learned head IDs/margins and keeps
every dev learned-text capture plus learned acts from both splits. Its memory run additionally checks cancellation/reset, budget
stops, maximum editor length, stack paint and the conservative allocator gate.
Both need a KB4 kit and 64-token assistant settings.

The older `smoke-test.sh` and `test-publication.sh` enforce a transformer-only
result schema (nonzero prompt tokens); they are not C14/C24 validators because
C14/C24 deterministic paths correctly use zero tokens and `stop=none`.

`test-publication.sh` uses one token to check the save path. it accepts an optional
second argument naming a ROM override. `tools/benchmark_product.py` separately
checks the frozen compatibility suite and records emulated timing/RAM. see
[the benchmark method](../../docs/ENGINE-BENCHMARK.md).

for real touch input and frame capture on the product ROM:

```sh
python3 tools/record_product_demo.py --kit build/chatds-kit \
  --work build/demo-recording --output docs/demo.gif
```

this recorder requires the release KB3 assets. it checks both answers on the
host first, then types the questions through the DS keyboard. `ui-script.txt`
on the card enables this optional test mode; normal kits never contain it.
scripted captures use at most 64 periodic two-screen frames plus explicit dumps.
leave at least 16 MiB free for the supplied demo script. remove `ui-script.txt`
to return to interactive use.

## card layout and controls

follow [card setup and storage selection](../../README.md#run), preserving any
flashcart bootloader files. staged layout:

```text
chatds.nds
chatds/model.bin
chatds/tok.bin
chatds/kb4.bin                # optional; preferred
chatds/round3-act.cdsr        # required for assistant mode
chatds/round3-op.cdsr         # required for assistant mode
chatds/round3-tag.cdsr        # required for assistant mode
chatds/kb.bin                 # optional legacy fallback when KB4 is absent
chatds/kb-ATTRIBUTION.txt      # staged with any index; keep it when redistributing
chatds/run.txt
LICENSE
THIRD_PARTY_NOTICES.md
```

boot `chatds.nds`. **Send**, return, or **START** submits. **Bksp** deletes.
**Stop** or **B** cancels between engine steps. **Up/Down** scroll the log;
**About/SELECT** opens the credits. Each transformer question resets KV.
Assistant mode runs the v0.2.0 product: condition 24 plus NARROWFIX (explicit
capability phrases route before help disambiguation; explicit literal-length
requests bypass factual retrieval, bare entity-length questions stay factual)
and the owner-style rule fixes (`clean/product/ownerfix.c`, casual math in
`clean/plumbing/casualmath.inc`). The editor starts empty; `prompt=` in
`run.txt` is used only by batch runs. The c14 style/rules/dialogue get first refusal,
then eligible learned conversation acts; guarded regex textops and calculator
precede learned op/literal-span execution; ranked KB4/direct fields/templates
and transformer fallthrough remain c14. Act/op/tag Q8 thresholds: 1793/386/0.
Learned spans only index original user bytes, never state-substituted queries.
A bounded 420-byte dialogue record persists across questions, separate from KV
and visible history. Repeat replays the capped last answer; pushback rechecks its
source. Pending clarification/candidates live one turn. Calc/text/code topic
changes clear entity state. Startup, errors, publication failure, Stop/B, and
"never mind"/"start over" reset dialogue. "Normal case" clears only style; a
session style request expires after eight styled answers. Path tags go only to
`chatds/debug.log`, never the UI. `kbd=0` runs once for unattended tests.
See [PRODUCT-C24](../../docs/PRODUCT-C24.md) for parity, learned fallback
screening and the unchanged >=64 KiB allocator-margin gate.

Results are `chatds/{out,ids,ctx,calc,heap}.txt`, plus append-only `debug.log`.
Optional test-only `session.txt` accepts R (reset), S (bounded permitted state),
Q (question), and C (cancel after N incremental steps); output is
`session-results.tsv`. Neither test input file is included in normal kits.
Optional UI scripting allocates the script's byte length plus a terminator
(up to 8192 bytes) only while it runs; that instrumented mode has a separate
memory budget from normal product use.
Each TXT result is written through a temporary file and renamed; `out.txt` is published last, with `status=` as its final line.
The TXT files describe the latest run, not a session archive. A failed publication is
not reported as a successful answer by the host runner or UI.

`heap.txt` includes model/session/KV allocations, bus-clock timing, and platform
heap boundaries on DS. the heap break is a high-water proxy, not a stack-canary
measurement. emulator timing is not physical DS timing. the old calibration and
hardware experiments remain separate from the release kit.
