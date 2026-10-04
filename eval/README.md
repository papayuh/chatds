# eval/ — frozen evaluation suite

The suite exists so no model result in this project is ever a cherry-picked prompt
(Sol run-1, "What model-quality result could be cherry-picked?"). It was frozen
**before any custom model was trained**.

**Score against v1.1.** v1 is kept for provenance only.

```
suite-v1.1.jsonl  300 items, 6 categories x 50   <- CURRENT
sha256            3ee6483b0724e5ab65120e6db0a8fd146842e36743d3588f588e5a2be7aadcd4
bytes             64605

suite-v1.jsonl    300 items, 6 categories x 50   <- superseded, never edited
sha256            a90305e5c540b3be93e9b24ffd9102d6f204cb6581c35e89cc7e2bb95a8236a9
bytes             63146
```

`sha256sum -c` this before every reported result and before every training-set
contamination check. If the hash of the file you scored is not that, the number is
not comparable to anything else in this repo.

### v1.1 (2026-08-30) — what changed and why

Sol's sprint-1 hardening review, objection 6: "suite regex/contains rules award
demonstrably wrong answers". v1.1 is the new file rule 2 requires. **Same 300 ids,
same 300 prompts, same 300 canonical answers, byte-identical** — only `accept`,
`reject` and `scoring` are tightened. Verify with:

```bash
diff <(jq -r '.id+" | "+.prompt+" | "+.answer' eval/suite-v1.jsonl) \
     <(jq -r '.id+" | "+.prompt+" | "+.answer' eval/suite-v1.1.jsonl)   # empty
```

What was tightened:

- **`contains` now matches at word boundaries** (a `score.py` change, so it applies
  to both files). `"nice"` no longer contains `"ice"` (fact-013); `"because"` no
  longer contains `"au"` (fact-034). This is the single change that closes the
  whole substring-trap class, not just the two Sol named. `fact-020`'s accept list
  gained `"0 degrees"` because the old entry `"0 degree"` was a prefix match.
- **96 of the 100 `python`/`bugfix` items are now `python-ast`**: the accept regex
  must match *and* the scored line must parse as Python. This kills unterminated
  strings and brackets as a class — `f"hello {name` (py-037), `d.get("k"` (py-024),
  `print(len(xs)` (bug-024). The four exceptions are items whose canonical answer
  or accepted alternative is not a parseable line: `py-046`, `bug-017`, `bug-029`,
  `bug-047`.
- **23 items gained `reject` patterns**, mostly bugfix items that accepted the
  unchanged buggy line: `xrange` (bug-015), `raw_input` (bug-016), `lenght`,
  `retrun`, `has_key`, `iteritems`, `.push(`, `xs[3]`, `i+1`, `,,`, `x == 5` where
  `x = 5` was asked for, and the rest. An audit probe that feeds every bugfix item
  the buggy line from its own prompt now finds **zero** acceptances (it found three
  in v1: bug-015, bug-016, bug-031).
- **Comprehension items use backreferences** so the variable has to be consistent:
  `[y ** 2 for x in xs]` no longer scores (py-041, py-042, py-049).
- **`py-046` now requires the `try`**, not just an `except ... as e:` clause.
- **Invalid shell flags rejected where the flag *is* the answer**: `uname -x`
  (shell-038), `tar -cz` without `f` (shell-043), `du -x dir` (shell-022),
  `ls -zla` (shell-050), bare `curl ` with no target (shell-031).
- **Prompt-echo accepts anchored**: `shell-011` (`date` — the prompt says "date and
  time"), `py-048` (`x is None` — the prompt says "whether x is None"),
  `bug-031`. `arith-018` and `trans-027` gained a `reject` for a bare restatement
  of the prompt.
- **`trans-025`** no longer accepts `"big"` as a synonym for *big*.
- **`trans-047`** initials are case-sensitive again.

Deliberately **not** tightened: a stray flag on an item where the *command name* is
the answer (`date -x`, `ls -x`, `whoami -x`). There the flag is noise, not the
answer, and rejecting all flags would also reject `ls -l`. Recorded here so the next
reviewer knows it was seen and decided, not missed.

v1.1 is **strictly harder** than v1. v1 and v1.1 numbers are never averaged or
compared (rule 2). No v1 model result had been reported when v1.1 was cut, so
nothing was invalidated. `score.py --suite eval/suite-v1.jsonl --selftest` now
reports 299/300 (fact-020, the `"0 degree"` prefix) — expected, and the reason v1 is
provenance-only.

## Rules

1. **Frozen.** Never edit `suite-v1.jsonl`. Not to fix a typo, not to loosen an
   `accept` list that a model narrowly missed. Both are how a suite quietly turns
   into a training target.
2. A change means a **new file** — `suite-v1.1.jsonl`, `suite-v2.jsonl`, its own
   sha256, and a section in this README saying what changed and why. v1 results and
   v1.1 results are never averaged together. (v1.1 was cut under this rule; see
   above.)
3. **No item, prompt, or canonical answer from this file may appear in training
   data**, including teacher prompts used to generate synthetic data. See
   Contamination below.
4. Report per-category accuracy, never only the overall number — the categories
   have very different difficulty and a model that only does arithmetic is not an
   assistant.

## Categories

Exactly the six the original handoff spec names:

| category | n | what it tests | scoring |
|---|---|---|---|
| `factual` | 50 | basic factual / definition recall | contains |
| `arithmetic` | 50 | arithmetic and simple logic | numeric |
| `transform` | 50 | text transformations | exact / contains / regex / numeric |
| `shell` | 50 | shell / Linux commands | regex |
| `python` | 50 | tiny Python tasks | python-ast (49) / regex (1) |
| `bugfix` | 50 | bug-fix / syntax tasks | python-ast (47) / regex (3) |

`judge` items: **0**. Every item is deterministically scored, so the suite runs
offline, on the DS, with no teacher in the loop. The `judge` branch exists in
`score.py` only so a future version can add one.

Each item also carries a `tier`, which is the *expected* difficulty, not a measured
one:

| tier | n | meaning |
|---|---|---|
| 1 | 85 | a competent 3M model should get these |
| 2 | 161 | the 8M target band |
| 3 | 54 | reach items; an 8M model getting half of these would be a good result |

A suite where a 260K TinyStories model scores well, or where 8M scores 0, is
useless. Tier-1 exists to keep the floor off zero; tier-3 keeps the ceiling
visible. If measured tier accuracy is not monotone (1 > 2 > 3) the *tiering* is
wrong — that is a finding to record, not a reason to edit the file.

## Item schema

```json
{"id":"fact-001","category":"factual","prompt":"What is the capital of France?",
 "answer":"Paris","accept":["paris"],"scoring":"contains","max_new_tokens":24,"tier":1}
{"id":"bug-015","category":"bugfix","prompt":"Fix this Python line: for i in xrange(5): print(i)",
 "answer":"for i in range(5): print(i)","accept":["range\\(\\s*5\\s*\\)"],
 "reject":["\\bxrange\\b"],"scoring":"python-ast","max_new_tokens":40,"tier":2}
```

`answer` is the canonical answer (also the selftest input and the string that
`tokens_per_answer.py` prices). `accept` is what actually gets matched. `reject` is
optional: if any `reject` pattern matches the scored line the item is wrong no
matter what `accept` says — this is how a bugfix item refuses the unchanged buggy
line. `score.py` schema-checks the suite on load and exits 2 on a malformed file.

## Scoring semantics (`score.py`)

Only the **first non-empty line** of the model output is scored, and a leading echo
of the prompt is stripped. Answers in this suite are all single-line; anything the
model says after the first line is ignored, which is deliberate — a tiny model that
rambles after a correct answer still gets the point, and `max_new_tokens` keeps the
ramble short.

| scoring | rule |
|---|---|
| `exact` | normalized output equals some `accept` entry |
| `contains` | some `accept` entry appears in the normalized output **at word boundaries** — `"nice"` does not contain `"ice"` |
| `regex` | some `accept` pattern matches, compiled `IGNORECASE`; use `(?-i:...)` where case is the thing being tested (uppercase/title-case items do) |
| `python-ast` | as `regex`, **and** the scored line must parse as Python (a header line ending in `:` may be given a synthesized `pass` body; nothing else is synthesized, so `except X as e:` with no `try` still fails) |
| `numeric` | the **last** number on the line equals `answer` (llama2.c-style completions end with the answer: "7 times 8 is 56") |
| `judge` | not scored; counted and reported separately |

Applies to every mode: an optional `reject` list of regexes, any of which matching
the scored line makes the item wrong.

Normalization = collapse whitespace, lowercase, strip surrounding quotes/backticks
and trailing `.!?,;:`.

```bash
python3 eval/score.py --selftest                          # 300/300, must stay 100%
python3 eval/score.py --negative-selftest                 # 3 forged runs, all must FAIL
python3 eval/score.py --outputs runs/8m-d.jsonl           # human table
python3 eval/score.py --outputs runs/8m-d.jsonl --json    # machine
python3 eval/score.py --outputs runs/ds-60.jsonl --subset # 60-item hardware subset
```

The default `--suite` is `eval/suite-v1.1.jsonl`.

Output file, one JSON object per line:

```json
{"id":"fact-001","output":"Paris.","tokens_generated":3,
 "bytes_read":7100000,"startup_bytes":0}
```

### Strict mode is the only mode (Sol sprint-1 hardening, objection 6→5)

Sol produced three forged runs the old scorer accepted with exit 0: one correct
line reporting `accuracy=1.0` with `missing=299`; a duplicated line with
`bytes_read=-100` reporting `-100 bytes/correct`; and a one-line dump of all 300
canonical answers scoring **227/300** while claiming one generated token. All three
now fail. `--negative-selftest` runs exactly those three through the production path
on every invocation, so the hole cannot silently reopen.

**Run-level violations** print `VIOLATION:` to stderr and **exit 1**:

- an outputs line that is not a JSON object, or has a missing / non-string /
  unknown / duplicate `id`, or a non-string `output`
- `tokens_generated`, `bytes_read` or `startup_bytes` that is not a nonnegative
  integer (the bad value is discarded, never averaged into a metric)
- **coverage**: any expected item with no output line. All 300 by default. Pass
  `--subset` to expect the 60-item hardware subset instead — and only that subset;
  outputs from outside it are then a violation too. There is no other partial mode:
  a run that answered 40 items is not a suite result.
- any item over `max_new_tokens`, or over its character budget (below)

**Item-level**, scoring the item wrong rather than aborting:

- `tokens_generated` greater than the item's `max_new_tokens` — the run broke its
  own cap
- a scored line longer than `max(64, 8 x max_new_tokens)` characters. This is the
  cap check for a run that does not report a token count, or lies about it. The
  all-canonical-answers dump is 3,268 characters against a 192-character budget, so
  it now scores 0/300 *and* raises a violation.

Sanity floor: an output file where every line is `"the answer is 1. print(x) ls
file.txt"` scores 0/300. If a change to `score.py` makes that pass, the change is
wrong.

### The bytes metric is amortized (Sol kernel-study, objection 7)

`bytes_read` is the **recurring** per-answer read; `startup_bytes` is the one-time
cold-start read. A RAM-resident model has `bytes_read ~ 0` and
`startup_bytes = the whole model`, which is exactly why a single
bytes-per-correct-answer number collapsed: it scored every resident model at zero
regardless of size or quality. The report now emits

```
(startup_bytes + N x recurring_bytes_per_answer) / (N x accuracy)
```

at **N = 1, 10, 100, 1000** answers. A resident model wins at large N and loses at
N=1; a streamed model is flat. RAM and latency are reported separately and are not
folded into this number — see the Sprint 2 design §1.

## Hardware subset

The full 300 items on real hardware costs roughly `300 x tokens/answer x s/token`
— at the projected 8M rates that is hours (see the Sprint 2 design). For DS runs
use the stratified 60-item subset: **every item whose id number is ≡ 1 (mod 5)**
(`fact-001`, `fact-006`, ...). Score it with `--subset`, which is the *only* way to
get a number out of a partial file — without the flag a missing item is a violation
and the run exits 1. Full 300 runs happen on the host simulator; the DS runs the
subset to confirm the host prediction. A `--subset` number and a full-300 number are
different numbers; label which one you are reporting.

## Clean host capture

`make host` builds the independent product runner. Capture generated answers
without prompt echo, then score the output JSONL:

```sh
python3 eval/run_eval.py --suite eval/chatds-acceptance-30.jsonl \
  --model path/to/model.bin --tokenizer path/to/tok.bin \
  --chatds-kb path/to/kb.bin --out build/eval.jsonl
```

The model must use the supported resident DSQ8 layout; KB is optional.
`--limit` is a labeled development subset, never a full-suite score. Product
input limits apply (ASCII, single-line question <=127 bytes, 1..256 output
steps). Historical runtime CLI/tokenizer capture and DSQ4 parity tools were
removed. Use `score.py --selftest` for the canonical-answer scoring invariant.

## Tokenizer pricing

```bash
python3 eval/tokens_per_answer.py eval/suite-v1.1.jsonl tok512.bin tok1024.model ...
```

Counts tokens for every prompt and every canonical answer, per tokenizer.
`.bin` = llama2.c `tokenizer.bin` (parsed in stdlib, greedy score-ordered BPE, same
algorithm as llama2.c `encode()`); `.model` = sentencepiece, if the module imports.

Measured (2026-08-30), mean tokens per canonical answer:

| tokenizer | vocab | prompt tok (mean) | answer tok (mean) | chars/token |
|---|---|---|---|---|
| `reference/ds-llm/tok512.bin` | 512 | 27.6 | 8.2 | 1.20 |
| `train/data/tok1024.model` | 1024 | 21.4 | 7.3 | 1.36 |
| `reference/ds-llm/tokenizer.bin` | 32000 | 11.0 | 4.3 | 2.29 |

That is the honest price of a small vocab: 512 needs **1.9x** the decode steps of
32k, so "small vocab makes each token cheap" only wins if bytes/token falls faster
than tokens/answer rises. Both of these tokenizers are TinyStories-trained; a
code-aware tokenizer at the same vocab will do better on `shell`/`python`/`bugfix`.
Re-run this table with the real Sprint-2 tokenizers before pricing anything.

## Contamination

Policy for any corpus used to train a model that is scored on this suite:

1. The teacher never sees `suite-v1.jsonl`. Synthetic data is generated from topic
   seeds, not from eval items.
2. Before training, scan the corpus for (a) exact matches of any `prompt`, (b) exact
   matches of any `answer` longer than 12 characters, (c) 8-gram overlap with any
   prompt *or answer*. Drop the hits and record the counts in the training report.
   (`corpus/postprocess.py` enforces (c) against both fields, stricter than
   scanning prompts alone — a small, fixed drop count vs. missing a templated
   eval answer's phrasing leaking into a corpus answer, so the code is left
   stricter rather than loosened to match a prompts-only reading.)
   `corpus/postprocess.py` also runs a Jaccard ≥0.8 word-overlap check on top
   of (a)-(c), to catch near-paraphrases of an eval prompt that dodge exact
   and 8-gram matching. That check compares corpus *prompts* against eval
   *prompts* only — never against answers, per rule 4 below.
3. Record the suite sha256 in the training report next to the corpus hash, so a
   later reader can tell which suite a corpus was screened against.
4. Overlap on short answers ("Paris", `ls`, `len(xs)`) is unavoidable and fine —
   that is domain knowledge, not contamination. Overlap on *prompts* is not.
   (The 8-gram check in rule 2(c) still applies to answers too — this rule is
   about short, incidental answer overlap, not long shared phrasing.)
