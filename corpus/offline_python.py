#!/usr/bin/env python3
"""Offline (API-free) Python training-data generator, stdlib only.

Builds prompt/answer pairs from the trusted (task, reference_answer, tier)
tuples in corpus/python_tasks.py by wrapping each task in a small set of
DETERMINISTIC natural phrasing templates -- no model call of any kind, and
no snippet is ever executed (ast.parse only, which is parse-only by
construction). Syntax validation is best-effort: answers that parse as a
statement/expression/block are tagged as such; reference fragments that are
grammatically incomplete on purpose ("except ValueError:", "else:", loop
headers without a body) are tagged "fragment" and kept, because they are the
intended reference style (corpus/python_tasks.py docstring). Syntax !=
semantic correctness: parsing proves nothing about runtime behavior, and
this generator makes no claim of unseen-phrasing generalization -- it
produces fixed template variants of trusted tasks only.

Contamination-filtered against the frozen eval/suite-v1.1.jsonl (sha256
pinned inside postprocess.load_eval_overlap_sets) using the exact same
checks postprocess.py applies to teacher data: exact-normalized prompt /
long-answer match, 8-gram overlap, prompt-vs-eval-prompt Jaccard >= 0.8,
plus refusal/overlong/empty and exact dedup. Optional --extra-eval filters
against an additional held-out personal set.

Output: JSONL records {prompt, answer, category, tier, source, task_family}
with the same field conventions as corpus/results/full-train.jsonl plus
source/task_family; export_train.py needs only prompt/answer and ignores
the rest. This script only ever writes --out; it refuses to write onto the
existing full corpus. See docs/OFFLINE-DATA.md.

Usage:
    python3 corpus/offline_python.py --out /tmp/offline-python.jsonl \
        [--seed 42] [--extra-eval my-heldout.jsonl]
"""
import argparse
import ast
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import postprocess as pp  # noqa: E402  (sha-pinned suite loaders + filter helpers)
from python_tasks import TASKS  # noqa: E402

CATEGORY = "python_offline"
SOURCE = "offline_python"
DEFAULT_SEED = 42
# Fixed phrasing diversity only. Contamination is checked below; different
# scaffolding alone does not guarantee separation from evaluation prompts.
TEMPLATES = [
    "How do you {task} in Python?",
    "In Python, how can I {task}?",
    "What's the Python code to {task}?",
    "Quick Python question: {task}.",
    "Python help: {task}.",
    "Show me the Python to {task}.",
    "{Task} in Python -- how do I do that?",
    "I keep forgetting how to {task} in Python.",
]
# This generator must never overwrite an existing corpus artifact.
PROTECTED_OUTPUTS = (
    os.path.join(HERE, "results", "full-train.jsonl"),
    os.path.join(HERE, "results", "full.jsonl"),
    os.path.join(HERE, "train-v1.jsonl"),
)

_HEADER_RE = re.compile(r"^\s*#\s+([\w][\w -]*?)\s*\(\d+\)\s*$")
_ENTRY_RE = re.compile(r"^\s*\(")


def task_families():
    """(task, answer, tier, family) tuples. Families come from the
    '# name (N)' section comments in python_tasks.py, in file order; the
    count is asserted to match TASKS so a future edit that breaks the
    mapping fails loudly instead of silently mislabeling."""
    families = []
    current = "python"
    with open(os.path.join(HERE, "python_tasks.py")) as f:
        for line in f:
            header = _HEADER_RE.match(line)
            if header:
                current = header.group(1).strip()
            elif _ENTRY_RE.match(line):
                families.append(current)
    if len(families) != len(TASKS):
        raise SystemExit(
            f"python_tasks.py parse drift: {len(families)} entries found vs "
            f"{len(TASKS)} in TASKS -- fix the section comments or this regex")
    return [t + (fam,) for t, fam in zip(TASKS, families)]


def syntax_label(answer):
    """'exec' / 'eval' / 'block' if the answer parses (as a statement, an
    expression, or a block header once completed with `pass`), else
    'fragment' for intentionally incomplete reference snippets like
    'except ValueError:'. Parse-only: ast.parse never executes or evaluates
    anything. Syntax validity is NOT semantic correctness."""
    for mode, src in (("exec", answer), ("eval", answer),
                      ("block", answer + "\n    pass")):
        try:
            ast.parse(src, mode="exec" if mode == "block" else mode)
            return mode
        except SyntaxError:
            continue
    return "fragment"


def build_candidates():
    """Every (task x template) pair, deterministically, in file order."""
    out = []
    for task, answer, tier, family in task_families():
        for tmpl in TEMPLATES:
            prompt = tmpl.format(task=task, Task=task[0].upper() + task[1:])
            out.append({
                "prompt": prompt,
                "answer": answer,  # verbatim reference answer: whitespace preserved
                "category": CATEGORY,
                "tier": tier,
                "source": SOURCE,
                "task_family": family,
            })
    return out


def filter_pairs(pairs, extra_eval=None):
    """Apply the same gates as postprocess.process() to offline pairs.
    Returns (kept, stats). The frozen suite is loaded through
    postprocess.load_eval_overlap_sets(), which hard-fails on a sha256
    mismatch of eval/suite-v1.1.jsonl. `extra_eval` accepts None, a single
    path string, or a list/tuple of path strings -- overlap sets from all
    given paths are merged in."""
    prompt_sets, exact_prompts, exact_long_answers, eval_ngrams = \
        pp.load_eval_overlap_sets()
    extra_paths = []
    if extra_eval:
        extra_paths = [extra_eval] if isinstance(extra_eval, str) else list(extra_eval)
    for path in extra_paths:
        ep, ex_p, ex_a, en = pp.load_extra_eval_overlap_sets(path)
        prompt_sets = prompt_sets + ep
        exact_prompts = exact_prompts | ex_p
        exact_long_answers = exact_long_answers | ex_a
        eval_ngrams = eval_ngrams | en

    stats = {"candidates": len(pairs), "kept": 0, "drop_empty": 0,
             "drop_refusal": 0, "drop_overlong": 0,
             "drop_contamination_exact_prompt": 0,
             "drop_contamination_exact_answer": 0,
             "drop_contamination_8gram": 0,
             "drop_contamination_overlap": 0, "drop_duplicate": 0,
             "syntax": {}}
    kept, seen = [], set()
    for rec in pairs:
        prompt, answer = rec["prompt"], rec["answer"]
        label = syntax_label(answer)
        stats["syntax"][label] = stats["syntax"].get(label, 0) + 1
        if not prompt.strip() or not answer.strip():
            stats["drop_empty"] += 1
            continue
        if pp.is_refusal(answer):
            stats["drop_refusal"] += 1
            continue
        if len(answer.split()) > pp.MAX_ANSWER_WORDS:
            stats["drop_overlong"] += 1
            continue
        p_norm, a_norm = pp.normalize_key(prompt), pp.normalize_key(answer)
        if p_norm in exact_prompts:
            stats["drop_contamination_exact_prompt"] += 1
            continue
        if len(a_norm) > 12 and a_norm in exact_long_answers:
            stats["drop_contamination_exact_answer"] += 1
            continue
        if (pp.ngrams(pp.normalize_word_list(prompt)) & eval_ngrams
                or pp.ngrams(pp.normalize_word_list(answer)) & eval_ngrams):
            stats["drop_contamination_8gram"] += 1
            continue
        if pp.max_overlap(pp.normalize_words(prompt), prompt_sets) \
                >= pp.OVERLAP_THRESHOLD:
            stats["drop_contamination_overlap"] += 1
            continue
        if (p_norm, a_norm) in seen:
            stats["drop_duplicate"] += 1
            continue
        seen.add((p_norm, a_norm))
        kept.append(rec)
        stats["kept"] += 1
    return kept, stats


def generate(out_path, seed=DEFAULT_SEED, extra_eval=None):
    """Build, filter, shuffle (seeded) and write. Byte-identical for the
    same seed + inputs, so re-running is idempotent; only --out is written."""
    real = os.path.realpath(out_path)
    for protected in PROTECTED_OUTPUTS:
        if real == os.path.realpath(protected):
            raise SystemExit(
                f"refusing to overwrite corpus artifact {protected} -- "
                f"the offline generator must never mutate the full corpus")
    pairs, stats = filter_pairs(build_candidates(), extra_eval)
    random.Random(seed).shuffle(pairs)  # content is seed-independent; order isn't
    dirname = os.path.dirname(os.path.abspath(out_path))
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(out_path, "w") as f:
        for rec in pairs:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    return pairs, stats


def main():
    ap = argparse.ArgumentParser(
        description="Offline (API-free) Python training-pair generator "
                    "from trusted canonical tasks. See docs/OFFLINE-DATA.md.")
    ap.add_argument("--out", required=True,
                    help="output JSONL path (required; must not be an existing "
                         "corpus artifact)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="shuffle seed (default 42); pair content is "
                         "seed-independent")
    ap.add_argument("--extra-eval", default=None,
                    help="additional eval-schema jsonl (id/prompt/answer per "
                         "line) to contamination-filter against on top of the "
                         "frozen sha-pinned suite-v1.1")
    args = ap.parse_args()

    pairs, stats = generate(args.out, args.seed, args.extra_eval)
    print(f"wrote {len(pairs)} pairs -> {args.out} "
          f"(source={SOURCE}, category={CATEGORY})")
    print(f"contamination-checked against frozen suite-v1.1"
          + (" + --extra-eval " + args.extra_eval if args.extra_eval else ""))
    for k in sorted(stats):
        if k != "syntax":
            print(f"  {k}: {stats[k]}")
    print("  syntax (parse-only; syntax != semantic correctness):")
    for lab, n in sorted(stats["syntax"].items()):
        print(f"    {lab}: {n}")


if __name__ == "__main__":
    main()
