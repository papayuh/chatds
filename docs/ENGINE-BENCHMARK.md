# clean product measurement: 2026-10-03

## exact builds and inputs

[raw observations](benchmarks/clean-product-2026-10-03.json) contain all 33 cases
for all three ROMs, complete normalized outputs, stop decisions, heap trailers,
raw clean timer ticks, source hashes and toolchain identities.

- clean default, int16 Q8.8:
  `394a455ba5e3605dd1734844f21fe31b9b9e4809d9bc7c363407c5328738bff2`
- experimental int8 Q4.4:
  `99b3949cbe4dd19c0abe6ee43316eb63318afa7f7ed4db9e5900b1a3f3426d76`
- frozen legacy black-box ROM:
  `bfdd06e783d7d7f6b0e6f8e8406295b0ee5d2cca319e163d46b0b3aa9404f9b1`

all runs use the fixture's pinned model, tokenizer and **KB2** index, not the
release/demo KB3 index. shape: dim 192, hidden 512, six layers, six query heads,
two KV heads, vocab 2048, context 256, group size 32. the product parses this
shape from the header. no dimensions are specialized into its kernels.

the clean ROM includes the real tokenizer, retrieval, calculator, touch UI and
result writer, even in unattended `kbd=0` mode. it is not the earlier scout ROM.
no legacy source was opened, compiled or linked for the comparison. the baseline
is a prebuilt binary used only through its public card protocol.

## method and boundaries

melonDS 1.1, Flatpak, DS mode, direct boot, JIT enabled, FPS limit disabled.
`QT_QPA_PLATFORM=offscreen` is set on the host, Flatpak invocation and sandbox
shell. each boot has its own ROM copy, FAT image, config, home and cache.
cleanup matches that exact ROM argv, never another lane's emulator. startup,
mtools and inference polling are bounded. no visible window is used.

one complete boot per case/build, plus a separate one-token baseline boot for
each of its 28 successful inputs. this is not a distribution or a median of
repetitions. the headline numbers come only from this run's three exact smoke
sequences. other prompts and the two accepted divergent outputs remain in the
raw record; shorter wrong outputs are never called a speed improvement.

- clean timing uses the DS bus timer, **33,513,982 ticks/s**, extended across
  samples so runs longer than a single 32-bit counter period remain valid.
- clean TTFT starts before question setup and ends at the first piece callback.
  it includes retrieval/tokenization/prefill, excludes loading the model, and
  is not the later screen-refresh time. when no piece is emitted, the record
  explicitly falls back to first-logit time.
- legacy TTFT is its public one-token run `ms`, a **proxy**. internal timer
  boundaries are unknown; the comparison does not pretend otherwise.
- total clean time ends before result-file publication, after generation and
  calculator handling. there is no host-wall-time speedup claim.
- decode steps/s = post-first forward count / `(total - first)`.
  EOS/BOS runs use `generated` forwards (the final one predicts the stop);
  budget runs use `generated - 1`. zero-step rows have no throughput value.
- model-load/boot time and physical DS performance are not measured here.

## results

| prompt | prompt / generated IDs | legacy TTFT proxy | clean TTFT | legacy total | clean total | total speedup |
|---|---:|---:|---:|---:|---:|---:|
| who was george washington | 51 / 18 | 35.414 s | 12.314 s | 48.199 s | 17.599 s | 2.739x |
| whats 38 times 47 | 15 / 11 | 10.292 s | 3.522 s | 17.905 s | 6.551 s | 2.733x |
| sort scores biggest first | 15 / 17 | 10.292 s | 3.522 s | 22.081 s | 8.227 s | 2.684x |

post-first decode steps/s: legacy **1.408 / 1.445 / 1.442**, clean
**3.405 / 3.631 / 3.613**, in the same question order.

### memory and KV decision

largest reported main-RAM base-to-heap-break over all successful cases:

- legacy: **3,436,544 B**.
- clean int16: **3,653,632 B**, with **393,216 B KV**, **403,704 B session heap**,
  and **9,472 B active session TCM workspace**.
- clean int8: **3,457,024 B**, with **196,608 B KV**.

these are **heap-break high-water proxies**, not allocator live peaks or measured
stack peaks. stack and system reservations above the heap limit, ITCM/DTCM and
VRAM are not included in the base-to-break value. the clean ELF reserves
**10,240 B DTCM** and **2,808 B ITCM**. the scripted two-question UI demo reaches
the same 3,653,632 B break, with the release KB3 assets. baseline KV bytes are
not exposed by this ROM's heap trailer, so no historical estimate is substituted.

int8 saves **196,608 B** of cache/main-RAM break, but only **26/33** fixture
behaviors match. on the same math sequence it takes **6.596 s** versus int16's
**6.551 s**, and on the same python sequence **8.295 s** versus **8.227 s**.
its different, shorter fact output is not a comparable speed test. **int16 stays
the default.** fixed Q4.4 is one experiment, not proof that every possible int8
scheme would fail. the int8 option was measured from an earlier build of this
branch and then removed from the engine source; its ROM hash and raw results
remain in the archive. the default int16 ROM rebuilt after that removal is
byte-identical to the measured one.

### parity

- baseline: 28 exact successful-input fixtures plus five observed clean failures.
- clean int16: **26 exact successful-input fixtures**, **five clean oversized-input
  rejections**, **two captain-accepted numerical divergences**.
- clean int8: 21 exact successful-input fixtures, five safety passes, seven
  divergences. rejected as a product option.

`near-tie-uppercase` and `raw-length-0` have separately pinned clean outputs in
[`clean-known-divergences.json`](../ds/engine-golden/clean-known-divergences.json).
successful frozen observations remain unchanged. public preparation normalized
safety-class labels and diagnostic paths only, retaining the original fixture
hash in provenance. strict `engine_golden.py check` still fails
those two; the product measurement command accepts only their recorded outputs,
not arbitrary future changes on the same prompts. no prompt-dependent branch
was added to inference. see the [investigation](ENGINE-SWAP-INVESTIGATION.md).

## reproduce

private model/tokenizer/KB assets are required. `GOLDEN_KIT` must contain the
frozen KB2 assets whose hashes match the fixture, not the newer release index.
the historical baseline executable lane has been removed; its recorded
measurements remain in the archive. card-path metadata in that archive was
anonymized, not remeasured.

```sh
source tools/env.sh
make
GOLDEN_KIT=/path/to/frozen-kb2-kit
MODEL="$GOLDEN_KIT/model.bin"
TOK="$GOLDEN_KIT/tok.bin"
KB="$GOLDEN_KIT/kb.bin"

python3 tools/benchmark_product.py --rom clean/product/build/chatds.nds \
  --model "$MODEL" --tokenizer "$TOK" --kb "$KB" \
  --known-divergences ds/engine-golden/clean-known-divergences.json --out build/measure-int16
```

use new/empty output directories. successful boots retain small raw text/logs
but discard their FAT images. failures retain artifacts. the archive command
refuses to identify a different current ROM as the measured build.
