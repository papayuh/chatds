# public-release audit

Scope: the current tracked snapshot, not private history or ignored worktree
files. Public publication has not happened. Export one reviewed tree into a new
repository with one initial commit; retain full history privately. Never mirror
refs, change private-repo visibility, or copy the entire working directory.

## removed or rewritten

- Deleted the entire unlicensed vendored runtime tree (101 tracked files),
  including converters, planner, tokenizer helpers, DS port, copied kernels,
  test fixtures and vendored images. No product dependency remains.
- Removed `ds/calib/`: its source explicitly said its IRQ timer was copied from
  unlicensed upstream code. Also removed `tools/run-melonds.sh` and
  `tools/calib-to-profile.py`, which served that obsolete calibration lane.
- Removed the old runtime's NDS launch script, `tools/ds-ui-test.sh` and
  `tools/ds_ui_scenario.py`. Current headless product/UI gates replace them.
- Removed `tools/parity-prefix.py`, `tools/test-parity-prefix.sh` and the
  `ds/golden-{15m,260k,pipe3m,pipe3m-dsq8,pipe3m-paged}.{ids,txt}` files: they
  compared old-runtime token-id output. `ds/engine-golden/` fixtures remain.
- Removed `corpus/test_chatds_parity.py` and `eval/dsq4_fakequant_parity.py`;
  clean plumbing parity and independent engine/kernel tests remain.
- Removed `train/export_run.py` and its tests: they delegated device conversion
  to the unlicensed converter. **No DSQ8 converter is currently shipped.**
- Removed the runtime-specific automated search controller/support/tests and
  campaign wrappers (`run_python_agent.sh`, `run_curriculum.sh`,
  `run_chatds_p2c.sh`, `run_chatds_p4.sh`, `run_chatds_p4c.sh`). These required the
  deleted planner/exporter; they were not standalone training tools.
- Rewrote `eval/run_eval.py` and `run_chatds_suites.sh` to execute the independent
  clean host product. Kept answer-only capture, calculator-call evidence,
  token counts, termination and strict scorer coverage. Subsets are labeled.
- Replaced the QAT test's exporter/C dependency with scalar math, zero-group,
  straight-through-gradient and shape-boundary tests. QAT itself is experimental;
  the device supports DSQ8, not DSQ4.
- Removed obsolete host-test exclusions/deselections, old workflow sandbox
  configuration, upstream fetch targets and old build/docs/card-layout references.
- Removed private planning/review/decision/sprint experiment reports. Rewrote
  public corpus/training, hardware, curriculum and release instructions.

## fixtures and measurement integrity

The retained goldens contain exported inputs/outputs, IDs, context sentences,
float observations and hashes, **not engine implementation code**. They need no
unlicensed runtime to consume. Successful expected outputs and numeric logits
were preserved, not regenerated to hide regressions. Public preparation renamed
only the safety class to `baseline-fails` and anonymized historical diagnostic
card paths. The original fixture hash is recorded in provenance; the divergence
policy pins the normalized file. Historical benchmark hashes still identify
original evidence, not the reserialized metadata.

The product ROM rebuilt with SHA256
`394a455ba5e3605dd1734844f21fe31b9b9e4809d9bc7c363407c5328738bff2`, byte-identical
to the measured clean int16 ROM. README timing numbers still refer to that
binary. No physical DS timing or stack-canary gate has been completed for it.

## secrets, identifiers and private material

Byte-scanned tracked files for the removed
engine name, personal home paths, email addresses, private-key headers and
common GitHub/AWS/provider key shapes. Manually checked credential loading,
`.env.example`, fetch/build scripts, documentation and model metadata strings.
No matching credential/key, personal-home path or email remains in the tracked
tree. This is a targeted audit, not a guarantee against every possible secret.

Easy fixes: removed a personal email from the KB download User-Agent (callers
can set `CHATDS_USER_AGENT`), removed machine-specific training launch paths,
and excluded account/balance/private campaign notes from the public snapshot.
The intentional author copyright and public upstream repository URLs remain.
SentencePiece metadata has only relative training/model-prefix paths, not
personal home paths. The UI device gate now uses a checkout-local lock rather
than a home-specific shared temporary namespace; emulator logs honor `TMPDIR`.
Runtime temp paths and synthetic `/home/user` prompts are
not personal identifiers. Ignored `.env`, build logs, emulator images and
optional reference clones must never be copied to the public snapshot.

## third-party source

| remaining material | provenance / terms | action |
|---|---|---|
| independent engine, tokenizer, plumbing, UI/font | project-authored; MIT | public MIT license approved; preserve notices |
| `train/configurator.py` | nanoGPT, MIT, Copyright 2022 Andrej Karpathy | added full notice in `THIRD_PARTY_NOTICES.md` |
| `train/train_instruct.py` | derived from MIT ds-llm/llama2.c trainer; upstream notice Copyright 2023 Andrej | added full notice; optional model/export modules remain external |
| `reference/` | optional ignored MIT ds-llm dependency, fetched explicitly | not part of source export |
| BlocksDS/libnds, FatFs, picolibc/libm, ARM7 runtime | external SDK; zlib/BSD-style and mixed toolchain terms | linked-runtime notice inventory still required before ROM redistribution |
| melonDS, DSpico tooling | external test tool / user's cart | not bundled in source or kit; retain upstream terms if ever bundled |

Upstream MIT notices were checked via GitHub's license API for
`karpathy/nanoGPT`, `karpathy/llama2.c` and `RileyGreiff/ds-llm`. No other known
unlicensed third-party implementation remains. Existing root `LICENSE` was
already MIT before this task; its obsolete deleted-tree exclusion was removed.
The owner subsequently approved **MIT** for the public code repository `chatds`;
the private repository will be named `chatds-private`. Root `LICENSE` supplies
that code license, without granting rights to separate datasets or binaries.

## binaries and model assets

The remaining tracked tree is roughly 2.2 MB of source/docs/observations. No
tracked ROM, ELF, weight checkpoint, FAT image or full index remains. Largest
tracked file is the code-free observation fixture (~440 KB). `docs/demo.gif`
is 37,241 bytes, generated from the clean ROM; source/hash is `docs/demo.json`.

| artifact | bytes | tracked? | redistribution assessment |
|---|---:|---|---|
| former `models/tok1024-s2.model` | 14,641 | no, removed | omitted by owner's decision; do not copy into public snapshot |
| former `models/tok2048-s2.model` | 30,426 | no, removed | same; tests now generate synthetic tokenizers locally |
| current DSQ8 weights | 2,986,240 | no | mixed project/DeepSeek-generated training data; model license/provenance review pending; do not upload |
| matching runtime tokenizer binary | 25,526 | no | derived from the 2048 model; same provenance review before download release |
| release KB3 index | 42,321,808 | no | Wikipedia CC BY-SA 4.0 permits redistribution with attribution, modification notice and share-alike terms |
| frozen KB2 index | 59,641,232 | no | same Wikipedia terms; compatibility asset, not release KB3 |
| local FAT image | 67,108,864 | no | ignored emulator input; may contain private/uncleared assets, never publish as-is |
| rebuilt clean ROM | 238,592 | no | application sources are clean; linked-runtime notice and physical gate pending |

TinyStories' model card declares CDLA-Sharing-1.0. Sections 1.11 and 3.5 of the
[primary license](https://cdla.dev/sharing-1-0/) leave publication of Results
unrestricted when they contain no more than a de minimis portion of source data.
That can permit trained-artifact redistribution; it does not automatically
license copied datasets or establish full mixed-source provenance. Confirm
artifact status before any future download release. Both pretrained tokenizer
models have been omitted from the source snapshot. `tools/train_tokenizer.py`
lets users train compatible new tokenizers on text they have rights to use;
the README explains that a new tokenizer does not match existing weights.
Do not infer
weights rights from MIT source headers.

## datasets and excerpts

- `eval/*.jsonl`, `eval/auto-dev-seeds.tsv`, project tasks/templates and synthetic
  generators are project-authored evaluation/training specifications under the
  existing code notice. No full generated training dataset is tracked.
- `corpus/kb/fixtures/mini.xml` is a 1,671-byte edited/synthetic test fixture with
  article-like text. Conservatively attributed as Wikipedia-derived data.
- KB-derived contexts in goldens, benchmark JSON and historical result text
  retain CC BY-SA 4.0 obligations. Attribution/source/license/modification notices
  now appear in `THIRD_PARTY_NOTICES.md`, separate from the code license.
- Wikipedia dumps/indexes are ignored. The release index attribution records
  its 2026-09-01 dump identity; future downloads need their own attribution.
- TinyStories datasets are not vendored. DeepSeek teacher outputs and generated
  corpora are not tracked; provider terms and dataset/weight provenance still
  need review before releasing those artifacts.

## validation

Local results after deletion (all passed):

- `make all host`: clean ROM and host runtime build; ROM hash remains identical
  to the recorded measurement identity above. Standalone ARM9 engine library
  with `TARGET=nds DTCM_POOL=1` also builds.
- `make test` / `tools/run_host_tests.py`: **331 pytest tests**, standalone
  tokenizer suite, generalized engine tests, **8,200 independent kernel-oracle
  rows**, and UI's **724 checks**. Optional MIT training dependency was fetched
  into ignored `reference/` for this full run; no training tests were excluded.
  After removing pretrained tokenizer models, the same 331-test host gate passes
  with locally trained synthetic tokenizers. Two additional public training-CLI
  tests pass, including whitespace/ID/export behavior and overwrite refusal.
- Core and UI ASan/UBSan gates passed using Clang; plumbing's sanitized probes
  and ARM9 static-frame checks ran in pytest.
- CPU trainer selftest: three iterations/checkpoint round-trip passed. Offline
  generation, training-shard export, curriculum suite checks, prompt/calculator
  TSV export, postprocess/provider-request selftests all passed without paid
  provider calls. Scorer canonical **300/300** and negative **3/3** selftests pass.
- Fresh clean KB3 kit builds. Three-question kit smoke and one-token publication
  gate match clean host IDs, stop, context, text and calculator result.
- Full **33-case KB2 device gate**: **26 exact, five safety passes, two exact
  pinned known divergences**; no new failure or waiver.
- Clean UI ROM headless screen parity: **36 files identical**, 703 frames,
  12 dumps, `status=OK`. Only this checkout's emulator artifacts/lock are used.
- Rewritten evaluator's seven capture/input-limit/calculator regression tests
  pass. An actual-model calculator capture scores **1/1**, preserving raw
  `calc(38*47)` evidence and separately published `1786` result. This is a smoke
  result, not a broad capability claim.
- Tracked-tree byte scan has zero removed-runtime-name, personal-home, email or
  common key-pattern matches. Python syntax, shell syntax, relative Markdown
  links and `git diff --check` pass.

CI is explicitly skipped for this shipping task; host workflow configuration is
not evidence of an emulator/hardware run. No public repository or release is
created here.

## remaining publication gates

Public source decisions are resolved: repository `chatds`, code license MIT,
and no pretrained tokenizer models, weights, KB indexes, FAT images or ROMs.
Self-contained tests train project-authored synthetic tokenizer fixtures at run
time rather than depending on the removed artifacts.

1. Future weight/data downloads need a declared model license, provider/data
   provenance and dump-specific KB attribution/share-alike terms.
2. Binary kits need exact linked-runtime notices and a physical DS lite + DSpico
   test, including stack headroom. Current emulator timings are not hardware claims.
3. Publish only the reviewed tracked snapshot as one initial commit. Keep all
   private history, ignored binaries, corpora, credentials and logs private.
