# clean ChatDS card kit

`make-kit.sh` builds `clean/product/` with int16 KV and stages `chatds.nds` plus
`chatds/` assets. no legacy engine, planner, converter or host oracle participates.
model shape is parsed at load time. the model must be resident DSQ8 and match its
tokenizer; passing these checks does not establish answer quality.

```sh
source tools/env.sh
bash ds/hwkit/make-kit.sh --mode assistant \
  --model /path/to/model.bin --tokenizer /path/to/tok.bin \
  --kb /path/to/kb.bin --kb-attribution /path/to/ATTRIBUTION.txt \
  --prompt 'what is the capital of france' --out build/chatds-kit
bash ds/hwkit/smoke-test.sh "$PWD/build/chatds-kit"
bash ds/hwkit/test-publication.sh "$PWD/build/chatds-kit"
```

both model and tokenizer are mandatory. `--mode story` uses the question as raw
input, without retrieval or calculator handling. defaults: assistant, 64 new
tokens; story, 200 new tokens. `--steps` accepts 1..256. the prompt is one ASCII
line, at most 127 bytes. the keyboard edits ASCII only.

an existing output is moved to `OUTPUT.previous`. if that backup already exists,
the build refuses to overwrite it. no script formats or writes a physical card.
`kit.json` identifies the engine and hashes staged files. `.host-golden.*` are
clean host smoke expectations for this kit, not replacements for the frozen
legacy fixtures.

## testing

the smoke test uses private FAT images, config, home/cache and ROM copies under
`build/product-smoke/`. melonDS is always offscreen. no shared config or emulator
lock is needed. with a KB it checks fact, calculator and python questions;
otherwise it checks the configured prompt. generated IDs, stop, context, text
and calculator output must match the clean host runtime exactly.

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

copy to the flashcart card root, preserving its bootloader files:

```text
chatds.nds
chatds/model.bin
chatds/tok.bin
chatds/kb.bin                 # optional
chatds/kb-ATTRIBUTION.txt      # required when redistributing the index
chatds/run.txt
LICENSE
THIRD_PARTY_NOTICES.md
```

boot `chatds.nds`. **Send**, return, or **START** submits. **Bksp** deletes.
**Stop** or **B** cancels between engine steps. **Up/Down** scroll the log;
**About/SELECT** opens the credits. each question resets KV; visible history is
not conversation context. `kbd=0` runs once for unattended tests.

results are `chatds/{out,ids,ctx,calc,heap}.txt`. each is written through a temporary
file and renamed; `out.txt` is published last, with `status=` as its final line.
these describe the latest run, not a session archive. a failed publication is
not reported as a successful answer by the host runner or UI.

`heap.txt` includes model/session/KV allocations, bus-clock timing, and platform
heap boundaries on DS. the heap break is a high-water proxy, not a stack-canary
measurement. emulator timing is not physical DS timing. the old calibration and
hardware experiments remain separate from the release kit.
