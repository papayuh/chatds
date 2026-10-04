# corpus/ — D13 teacher-data generation plan

Builds the "45% teacher-generated instruction pairs" slice of the Sprint 2 data mix
(instruction-pair recipe): ~650k pairs from
`claude-haiku-4-5` via the Batch API, hard-capped at $105. Mirrors the eval suite's
six skill families (`eval/README.md`) plus a general-chat category, without ever
looking at `eval/suite-v1.jsonl` during generation — contamination is checked
after the fact (`postprocess.py`), not avoided by construction.

## Category mix (650,000 pairs total)

| category | pairs | requests (5/req) | mirrors eval category | rationale |
|---|---:|---:|---|---|
| arithmetic | 130,000 | 26,000 | `arithmetic` | huge, free combinatorial space (operand x operation) — cheapest to generate abundantly and the one place a tiny model can be reliably correct |
| transform | 110,000 | 22,000 | `transform` | word/phrase x transform-type crossing scales easily; heavy repetition across surface forms is the explicit design goal |
| python | 110,000 | 22,000 | `python` | code the model must produce, not just recognize; scales via varname/pattern crossing |
| bugfix | 100,000 | 20,000 | `bugfix` | broken-snippet patterns x varname/value crossing |
| shell | 90,000 | 18,000 | `shell` | command-task x file/dir-name crossing |
| factual | 60,000 | 12,000 | `factual` | bounded by how many real facts exist to ask about; compensated with heavy template x phrasing repetition per fact, not padding with more facts |
| chat | 50,000 | 10,000 | *(new — not an eval category)* | general short-instruction/conversational fluency; per the Sprint 2 design §2, an 8M model with no fluency prior "produces degenerate text" — this is a teacher-quality top-up to TinyStories' free narrative share, sized smaller since narrative fluency is TinyStories' job, not this pipeline's |

Sum = 650,000, evenly divisible by `PAIRS_PER_REQUEST=5` in every row (130,000
requests total, 13 shards of ≤10,000).

Every category also carries a `tier` (1/2/3) on each generated prompt, assigned by
the template/range that produced it (e.g. arithmetic operand range, factual
domain), mirroring `eval/README.md`'s tier scheme. Not calibrated against the eval
suite's actual tier distribution — it's a generation-time difficulty label, not a
measured one, same caveat eval/README.md gives for its own tiers.

## Design: we write the prompts, the teacher only answers

`gen_requests.py` deterministically expands templates x topic-lists x
difficulty into the **exact prompt text** for all 650,000 pairs — no LLM call
is involved in deciding what to ask. Each Batch API request bundles
`PAIRS_PER_REQUEST=5` of these prompts and asks Haiku only to *answer* them,
one strict JSON object per line:

```json
{"i": 0, "answer": "..."}
```

Why this split instead of asking Haiku to invent the pairs (as
the Sprint 2 design's "20 pairs per call" sketch does):

1. **Cost is decided at generation time, before a dollar is spent.** The full
   set of prompts (and their sizes) exists before submission, so
   `gen_requests.py`'s printed cost estimate is not a guess about what Haiku
   *might* generate — it is close to the actual bill.
2. **Contamination can be checked before submission.** Since we already know
   every prompt, `postprocess.py`'s filter can (and, per the dry-run evidence
   below, was) run against the real 650k prompts pre-submission as a sanity
   check, not just the post-hoc synthetic fixture.
3. **"Heavy repetition of skills across surface forms" is a property we
   control directly** — the diversity strategy (below) *is* the corpus, not a
   hope about what an LLM will produce when asked for "20 varied pairs."

The tradeoff: Haiku can't invent facts we didn't think to ask about. That's
fine — the Sprint 2 design's own accuracy argument is about the trunk learning a
fixed *skill* (answer format, tiny-code style, arithmetic), not about breadth
of world knowledge a 3-8M model was never going to hold anyway.

## Diversity strategy: template x topic x difficulty crossing, seeded

Every category builds a **base list** of (question, tier) pairs from a
curated topic list (countries, elements, variable-name pools, shell
file/dir pools, word lists, bug patterns, etc.) crossed with several
phrasing templates. Six of the seven categories (all but `arithmetic`, whose
diversity comes from its huge numeric range instead) then cross that base
list against a shared pool of **30 generic phrasing wrappers**
(`Q: {q}`, `Briefly: {q}`, `Pop quiz: {q}`, `For my homework: {q}`, ...) —
the mechanism that turns "500 things worth asking" into "50,000 ways to ask
about them," which is the literal mechanism behind "heavy repetition of
skills across surface forms."

`random.Random(42 + stable_hash(category))` (a `zlib.crc32` hash, not
Python's built-in `hash()` — string hashing is randomized per-process by
`PYTHONHASHSEED` unless disabled, which silently breaks determinism; verified
below) shuffles each category's cross-product space and takes the first N
unique prompts. Re-running `gen_requests.py` is byte-identical
(verified: sha256 of every shard matches across two runs, and a third run
with `PYTHONHASHSEED=random` forced).

Arithmetic gets special handling: its "tier 1" range (operands 1-20) has a
combinatorial ceiling around 4,800 unique prompts, far below what a naive
`n/3`-per-tier split would demand at 130,000 total — the fix is a fixed
*fraction* of `n` per tier (documented in `gen_requests.py`), not a fixed
count, so `--smoke`'s much smaller `n` doesn't try to way overshoot tier 1's
ceiling either.

## Prompt templates for the teacher

System prompt (shared across every request, `common.SYSTEM_PROMPT`):

> You are generating training examples for a tiny (3-8M parameter) assistant
> model with a 4096-word vocabulary and single-line answers. You will be
> given several numbered questions or instructions. Answer each one directly
> and correctly in 30 words or fewer; for code, give the minimal code only,
> no explanation or markdown fences. Output ONLY one JSON object per line, in
> this exact form and nothing else: `{"i": <number>, "answer": "<answer>"}`.
> One line per question, no blank lines, no commentary before or after.

User turn: the 5 numbered prompts, one per line (`common.user_message`), e.g.:

```
0. Pop quiz: Who is credited with inventing the polio vaccine?
1. For my homework: What is the capital of Panama?
2. Here's a question: On which continent is Tonga found?
3. Djibouti's capital city is what?
4. I was wondering, What city is the capital of Ivory Coast?
```

`max_tokens=200` per request (tight, per D13) — sized for 5 short answers
(~40 tokens each including JSON overhead), not the 20-pairs/~1200-token call
the Sprint 2 design sketches; see cost math below for why this still clears the
budget with room to spare.

## Cost math

Pricing assumption (state explicitly, per instructions): `claude-haiku-4-5`
standard rate is $1.00 / $5.00 per MTok in/out (current per the bundled
`claude-api` skill's cached pricing table, checked 2026-08-30); the Batch API
is a flat 50% discount on all token usage, giving **$0.50 / $2.50 per MTok
in/out** for batch requests. This matches the Sprint 2 design's stated
assumption exactly, so no drift between the two documents.

- 650,000 pairs / 5 per request = **130,000 requests**, 13 shards of ≤10,000.
- Measured (not guessed) average input size across the real generated
  650,000-pair request set: system prompt (≈445 chars) + 5-prompt user
  message, ≈220 tokens/request (chars/4 + fixed overhead — `gen_requests.py`
  computes this from the actual shard content, not an estimate written by
  hand).
- Output: `max_tokens=200` is the ceiling per request, not the expected
  average (short answers to short factual/code prompts rarely need it).

| scenario | in tokens | out tokens | cost |
|---|---:|---:|---:|
| expected (avg ~32 tok/answer x 5) | 28.58M | 20.80M | **$66.29** |
| worst case (every request maxes `max_tokens=200`) | 28.58M | 26.00M | **$79.29** |

Worst-case headroom under the $105 cap: **(105 − 79.29) / 105 = 24.5%**,
clearing the 15% requirement even under the pessimistic assumption that every
single one of 130,000 requests uses its full token budget (Haiku won't —
answers this short rarely hit 200 tokens — so $66-70 is the realistic number).
The submitter uses the worst-case formula for its budget guard, so the
"spent estimate" in the local `batches.json` ledger is always conservative relative to
the real bill.

## Contamination filter (design notes; full policy in `eval/README.md`)

Three checks, in order, against `eval/suite-v1.jsonl` (sha256 pinned and
verified before every run — `postprocess.py` refuses to run against a suite
file that doesn't match `a90305e5...236a9`):

1. Exact normalized prompt match.
2. Exact normalized answer match, answers >12 chars only (short-answer
   overlap like "Paris" or "ls" is domain knowledge per eval/README.md, not
   contamination).
3. **Jaccard word-overlap ≥ 0.8** between a candidate's *prompt* and any eval
   *prompt* (prompts only, never answers — matches rule 2 above: short answer
   overlap is domain knowledge, not contamination), normalized (lowercase,
   punctuation stripped, word set). Jaccard, not containment — a pre-submission dry run against the
   real 650k-prompt set (not just the synthetic self-test fixture) showed
   containment produces false positives on short templated text: "capital of
   Wakanda" vs eval's "capital of France" scores 0.83 containment purely from
   shared scaffolding words, and a 1-word answer like "apple" scores 1.0
   containment against eval's 3-word answer "apple mango pear". Jaccard's
   union-sized denominator fixes both (0.71 and 0.33 respectively — neither
   trips 0.8), while still catching genuine near-duplicates (a sampled
   pre-submission scan found ~0.11% of prompts, mostly legitimate near-misses
   like an unclosed-tuple bugfix pattern that happens to closely restate an
   eval bugfix item).

## What's out of scope for this pipeline

The other three legs of the Sprint 2 design's corpus mix — TinyStories narrative
(30%, free/public), programmatic arithmetic (10%, free/unlimited), and any
curated public code snippets folded into the 15% "tiny code" share — are not
built here. This directory only produces the teacher-generated instruction
pairs D13 approved a budget for.
