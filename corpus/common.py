"""Shared constants for the D13 teacher-data pipeline. stdlib only."""

MODEL = "claude-haiku-4-5"
PAIRS_PER_REQUEST = 5          # K: how many templated prompts one batch request bundles
MAX_TOKENS = 200               # tight, per D13 spec — see spec.md cost math
SHARD_SIZE = 10_000            # requests per shard file (Batch API allows up to 100k)
SEED = 42

CATEGORIES = ["factual", "arithmetic", "transform", "shell", "python", "bugfix", "chat"]

# Target pair counts per category. Sum = 650,000. Each is a multiple of
# PAIRS_PER_REQUEST so request counts divide evenly.
TARGET_PAIRS = {
    "factual":    60_000,
    "arithmetic": 130_000,
    "transform":  110_000,
    "shell":       90_000,
    "python":     110_000,
    "bugfix":     100_000,
    "chat":        50_000,
}
assert sum(TARGET_PAIRS.values()) == 650_000
for _cat, _n in TARGET_PAIRS.items():
    assert _n % PAIRS_PER_REQUEST == 0, f"{_cat} count not divisible by PAIRS_PER_REQUEST"

# Batch API pricing for claude-haiku-4-5, batch (50% off standard $1.00/$5.00 per MTok).
# ponytail: hardcoded assumption, re-check shared/live-sources.md pricing before a real submit
# if this pipeline is reused months from now.
PRICE_IN_PER_MTOK = 0.50
PRICE_OUT_PER_MTOK = 2.50

BUDGET_CAP_USD = 105.00

SYSTEM_PROMPT = (
    "You are generating training examples for a tiny (3-8M parameter) assistant "
    "model with a 4096-word vocabulary and single-line answers. You will be given "
    "several numbered questions or instructions. Answer each one directly and "
    "correctly in 30 words or fewer; for code, give the minimal code only, no "
    "explanation or markdown fences. Output ONLY one JSON object per line, in "
    "this exact form and nothing else: {\"i\": <number>, \"answer\": \"<answer>\"}. "
    "One line per question, no blank lines, no commentary before or after."
)


def user_message(items):
    """items: list of {"i": int, "prompt": str}. Returns the batched user turn text."""
    lines = [f'{it["i"]}. {it["prompt"]}' for it in items]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# python_natural addendum (Diego, 2026-09-05): deepseek-only category, not
# part of the 650k main corpus / not in CATEGORIES or TARGET_PAIRS, so it
# never touches the main corpus's budget bookkeeping or shard directory.
# See corpus/python_tasks.py for the 246 canonical tasks it's built from.
# ---------------------------------------------------------------------------
PYTHON_NATURAL_TARGET_PAIRS = 120_000    # ~24,000 requests at PAIRS_PER_REQUEST=5
PYTHON_NATURAL_MAX_TOKENS = 400          # headroom for phrasing + answer, not just an answer

PYTHON_NATURAL_SYSTEM_PROMPT = (
    "You are generating training examples for a tiny (3-8M parameter) assistant "
    "model with a 4096-word vocabulary and single-line answers. You will be given "
    "several numbered Python tasks, each with a task description and a reference "
    "answer. For each one, invent ONE natural, casual way a real person might ask "
    "for it in chat -- not a template, vary the phrasing style every time -- then "
    "answer it the same way as the reference answer (minimal code only, no "
    "explanation, no markdown fences; adapt the reference answer only if the "
    "phrasing truly requires it). Output ONLY one JSON object per line, in this "
    "exact form and nothing else: {\"i\": <number>, \"prompt\": \"<your natural "
    "phrasing>\", \"answer\": \"<answer>\"}. One line per task, no blank lines, "
    "no commentary before or after."
)

MAX_TOKENS_BY_CATEGORY = {"python_natural": PYTHON_NATURAL_MAX_TOKENS}
SYSTEM_PROMPT_BY_CATEGORY = {"python_natural": PYTHON_NATURAL_SYSTEM_PROMPT}


def max_tokens_for(category):
    return MAX_TOKENS_BY_CATEGORY.get(category, MAX_TOKENS)


def system_prompt_for(category):
    return SYSTEM_PROMPT_BY_CATEGORY.get(category, SYSTEM_PROMPT)


def estimate_cost(num_requests, avg_input_tokens, avg_output_tokens):
    in_tok = num_requests * avg_input_tokens
    out_tok = num_requests * avg_output_tokens
    cost = (in_tok / 1_000_000) * PRICE_IN_PER_MTOK + (out_tok / 1_000_000) * PRICE_OUT_PER_MTOK
    return cost, in_tok, out_tok
