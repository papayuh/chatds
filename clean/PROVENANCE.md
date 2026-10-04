# Clean-core provenance

This directory is independently written MIT-licensed ChatDS code. It is not a
renamed, translated or mechanically modified copy of an upstream runtime. Root `LICENSE`
applies to these sources; BlocksDS/libnds/libc/libm retain their own licenses.
This does not clear rights for model weights, training data, tokenizer assets,
or the Wikipedia-derived knowledge base.

## Hard boundary for maintainers and automated reviewers

**Do not open, read, grep, copy or disassemble unlicensed upstream implementation
source or binaries to implement this engine.** No such source was consulted.
Exported model/tokenizer/KB data, their format metadata and previously captured
black-box observations may be examined. Historical opaque builds are not an
implementation reference; no legacy build path remains in this tree.

The phase-1 C interface is `include/chatds_engine.h`. Tests and adapters should
use it rather than importing legacy engine headers.

## Sources actually consulted

- Project-owned format/math and model-training experiment notes. Those historical
  plans/reports remain in private history, not the public code snapshot.
- Project-owned training/export context, including `train/qat4.py` and the
  since-removed export/parity scripts. The old exporter delegated conversion to
  an unlicensed upstream script; that converter was **not** opened. The QAT
  math/doc descriptions informed stored-scale interpretation, not a kernel port.
- Diego-owned host/runtime/harness context: `corpus/chatds_runtime.py`,
  `tools/{env.sh,melon-lib.sh,chatds_prompt.py}` and the old emulator launcher,
  `ds/hwkit/{README.md,make-kit.sh,smoke-test.sh}`, the old calibration Makefile, root
  `README.md`, `.gitignore`, and `LICENSE`.
- The old calibration source's first 170 lines were read during the scout. Its
  comment identified an upstream-derived IRQ timer. That timer was **not**
  copied or used; the calibration lane was removed during public preparation.
  The scout called public BlocksDS timing functions; this library has no timer
  dependency at all.
- Exported P2c data, not source. Model SHA-256:
  `cf2f9e9e1928d61834937959759623a6cf512e2378bceba3660bf763b2d9818b`.
  The little-endian header, row scales, padding and final file length determined
  layout. A wrong early matrix-alignment hypothesis was rejected against raw
  data and black-box logits, not repaired by looking at a legacy parser.
- Black-box `tie-run --help`, generated IDs/logits, and the existing P2c ROM
  (SHA-256 `bfdd06e783d7d7f6b0e6f8e8406295b0ee5d2cca319e163d46b0b3aa9404f9b1`).
  An optional missing tokenizer-oracle build failed during the scout; its build
  log was not read, and that optional check was dropped explicitly.
- Public [GBATEK](https://mgba-emu.github.io/gbatek/) CPU/cache/TCM/VRAM/DMA
  descriptions; public installed BlocksDS timing/heap/TCM declarations and
  `nds/ndstypes.h`. No library implementation was copied. The only libnds header
  the core uses directly is `nds/ndstypes.h` for section attributes on ARM9.
- Official melonDS 1.1 README/configuration observations for the scout harness,
  not emulator implementation code. Only **our own** compiled kernel was
  disassembled to confirm ARM halfword-MAC emission.

## Evidence and limits

The initial fixed-shape scout ran complete inference in headless melonDS and
was timed against the frozen kit. Those timings belong to the scout binary and
are not claimed for this generalized core. Fresh product measurements are in
[ENGINE-BENCHMARK.md](../docs/ENGINE-BENCHMARK.md). The scout's answers matched all 46 IDs plus three EOS decisions;
full numeric logits did not match. See [ARITHMETIC.md](ARITHMETIC.md).

This generalized core independently verifies scalar/fast equality and bounded
arithmetic with synthetic models/oracles requiring no private assets. Local
host checks still match the three frozen-kit sequences with zero saturation.
The product now validates all 33 frozen cases, with 31 retained behaviors and
two explicitly accepted numerical divergences; see the product README.
Synthetic black-box input experiments are documented in
[ENGINE-SWAP-INVESTIGATION.md](../docs/ENGINE-SWAP-INVESTIGATION.md). No experimental
reciprocal-wrap or prompt-specific arithmetic was adopted. No scratch ROMs, private weights, copied legacy source, or emulator
configuration are shipped in this directory.
