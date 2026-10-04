# engine compatibility gate

The fixtures are code-free black-box observations retained from the private
project. The independent product uses only `chatds/` on its card. The historical
runtime, card layout, logit capture CLI and baseline build path are not shipped.

## run the product gate

Supply the model, tokenizer and frozen **KB2** index whose hashes match fixture
provenance. The release/demo **KB3** index is different.

```sh
source tools/env.sh
make
python3 tools/benchmark_product.py --rom clean/product/build/chatds.nds \
  --model path/to/model.bin --tokenizer path/to/tok.bin --kb path/to/kb2.bin \
  --known-divergences ds/engine-golden/clean-known-divergences.json \
  --out build/product-golden
```

Expected: 26 exact outputs, five explicit safety passes and two pinned known
divergences. `tools/engine_golden.py check` is stricter and fails those two
historical sequence differences. Product parity accepts only the recorded clean
outputs, not arbitrary answers to the same questions.

All boots are offscreen, with private ROM copy, image, config, home and cache.
The harness bounds startup/commands/polling and kills only its exact ROM process.
Failures retain artifacts. A timeout never counts as a safety rejection.

## adapters and new fixtures

`check --adapter EXECUTABLE` calls `EXECUTABLE REQUEST.json RESULT.json` once per
case. `--session-adapter` calls it once with all inputs and expects an ordered
result list, allowing repeated-call/KV-reset coverage. Requests contain inputs
only, never `expected` or host-oracle data. Supplied assets must match hashes.

`freeze --cases CASES.json --fixtures NEW.json` records a new clean-device
observation set; it refuses to overwrite an existing file. Successful captures
record IDs, stop, prompt count, context and text. The optional historical
float-logit capture path was removed. Existing frozen host logits are static
observations, explicitly marked `host-float32-not-device`.

Host checks (`make test`) need no private model, KB or emulator. They execute
the fixture consumer with fake executable adapters to test mismatch, crash,
cleanup, input isolation and asset-pinning behavior, plus real engine tests on
synthetic models. [ds/engine-golden/README.md](../ds/engine-golden/README.md)
describes the metadata-only normalization performed for public prep.
