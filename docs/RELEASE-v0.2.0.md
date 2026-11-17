# chatds v0.2.0

the build the owner demoed on a real DS on 2026-10-21. download
`chatds-v0.2.0-sdcard.zip`, extract it on a PC, copy `chatds.nds` and `chatds/`
to the card root.

**hardware-tested: owner ran this exact ROM (sha256
`b2e7a47ba0c4509fdbefa6401261f6717810ec6bfad8f447ea2b92d88e1b100a`) on a real
DS via DSpico, 2026-10-21 (qualitative).** no timings were recorded.

## what changed since c24

- **no startup question.** the editor starts empty with the ready note. the
  card's `run.txt` still has `prompt=`, but only batch runs use it.
- **new model and tokenizer.** same shape (2.75M params, DSQ8, 2048 tokens),
  retrained from scratch on permitted pairs only: no GLM- or DeepSeek-authored
  data. fewer code leaks, but it can be confidently wrong more often on casual
  chat. [weights review](WEIGHTS_RELEASE_REVIEW.md).
- **license.** the owner decided on 2026-10-21 to publish as is: MIT confirmed
  by the owner for `chatds/model.bin` and `chatds/tok.bin`. 4,514 GPT-authored
  and 854 Claude-authored rows are among the 275,272 training pairs. the owner
  waived the round3 heads audit under the GLM/DeepSeek rule. details in the
  [weights review](WEIGHTS_RELEASE_REVIEW.md).
- **owner-style rule fixes.** identity slang ("who r u"), caps aliases
  (`caps "a b!"`), reverse-word spans ("flip the order of the words in ..."),
  task labels ("math: 3+4"), conversational prefixes before a fact question,
  and wrapped requests ("could you 3+4 for me").
- **casual math.** arithmetic inside prose computes ("whats 17 plus 6 tho").
  phone numbers and dates are left alone; a bare number only echoes after a
  calculation.
- **routing.** "what can you help me with" and similar go to capabilities;
  "how long is the word nile" is a text op, "how long is the nile" stays a fact.
- **faster model math.** an ARMv5TE assembly kernel for the main matvec, with a
  boot self-check that falls back to C on any mismatch. emulated decode is
  about 10-14% faster per token.
- **RAM.** packed tokenizer tables free about 18 KB of heap.
- KB4 and the three learned heads are unchanged.

## files

| file | sha256 |
|---|---|
| `chatds.nds` | `b2e7a47ba0c4509fdbefa6401261f6717810ec6bfad8f447ea2b92d88e1b100a` |
| `chatds/model.bin` | `ff0a3e8289845f7f9906103e8405f1b2044ed2711f06825837b40f26c3e95eb9` |
| `chatds/tok.bin` | `582766d0ff7962690aa51e52dfdddaa9efcde220015ff7a84952342bb72d7203` |
| `chatds/kb4.bin` | `08d4ce833e91fbf49c610075d25e55a79c915797f9504e72d6ea77385b209ba7` |
| `chatds/kb-ATTRIBUTION.txt` | `b1576e4941978e51efe29ac92320e5335da0101b89ab15cfd1b083311206087d` |
| `chatds/round3-act.cdsr` | `e46781ffcdbff485f33fcc918dffc5994d3da826241ae54a712befb38a1a7a86` |
| `chatds/round3-op.cdsr` | `aa825e492e45b537df09150e64915636cba777ca7e9e2ddbaadc2c9e30db5dd1` |
| `chatds/round3-tag.cdsr` | `6b7bd4db559fd936f1a260a45003da88135e69dc61814b2ec51451ba5cccacf4` |
| `chatds/run.txt` | `55d836226cf0bc24e99c75f7687d8ddbecc879b4f998270d3cb25f3c80916579` |

the ZIP also has `README.txt`, `LICENSE`, `THIRD_PARTY_NOTICES.md`, `kit.json`
(names, sizes, hashes) and `licenses/`, the same layout as v0.1.0.

## rebuild

Use the `v0.2.0` tag, not a later source tree:

```sh
git checkout v0.2.0
source tools/env.sh
# assets/ holds the release model.bin and tok.bin plus their SHA256SUMS
bash ds/hwkit/make-kit.sh --mode assistant --asset-dir assets \
  --kb4 path/to/kb4.bin --kb-attribution path/to/kb-ATTRIBUTION.txt \
  --prompt 'whats 38 times 47' --out build/v0.2.0-kit
sha256sum build/v0.2.0-kit/chatds.nds
```

with BlocksDS 1.23.0 and Wonderful ARM GCC 16.2.0 the tagged tree rebuilds the ROM
byte-for-byte, and the kit's `chatds/` matches the release files above.

## validation

run on the v0.2.0 tree, ROM `b2e7a47b…100a`. all emulator checks are headless melonDS
(Qt offscreen, display sockets denied, no GPU), not hardware certification.

- **ROM rebuild.** `make-kit.sh --asset-dir` from the tagged tree gives ROM
  `b2e7a47b…100a`, byte-identical to the card build, and the same 8 `chatds/`
  files.
- **host tests.** `python3 tools/run_host_tests.py`: 851 passed, 3 skipped
  (optional training references absent), standalone tokenizer suite, engine
  tests including the asm seam (2,880 guard/fallback rows), UI 724 checks,
  0 failed. `make -C clean sanitize` (ASan/UBSan) passes.
- **golden parity.** the 33 frozen golden inputs with frozen KB2 and the v0.2.0
  model/tokenizer: ROM == host on text, IDs, counts, stop, context and
  calculator for all 33 (28 answers, 5 expected clean failures). the gs32 boot
  self-check passed on every boot. answers differ from the old fixture text
  because the model changed; inputs and fixtures are untouched.
- **headless screen and UI.** cold and warm boots on one FAT32 image: the editor
  starts empty, typed questions answer the same as host (`whats 17 plus 6 tho`
  -> `calc(17+6) = 23`, `what is the capital of france` -> `Paris.`). then a
  16-turn rule smoke, four screen protocols (71 turns) and a full-context
  (256 KV positions) turn: 88 turns, 0 host/ROM mismatches.
- **RAM.** minimum sampled heap reserve 81,920 B (gate 65,536 B), minimum
  painted stack free 5,816 B (gate 1,536 B), across all 41 boots above. these
  are sampled high-water marks, not all-input guarantees.

the screen protocols and the gate driver come from the private validation
harness, as [RELEASE.md](RELEASE.md) says.

## limits

answers are AI-generated and often wrong. the model trades fewer code leaks
for more confident-wrong casual answers than c24's model. emulator timings are
not DS timings. the owner's demo is the only hardware run.

- casual math skips unknown words around the math, so "what is 12 times 12 in
  binary" or "sqrt 16 plus 9" can get a plain calculator answer (144, 25) that
  ignores the conversion or function word.
