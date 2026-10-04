#!/usr/bin/env python3
"""Deterministic, API-free Python curriculum generator (R3 data-coverage
experiment). stdlib only; no network, no .env, no model calls.

Hypothesis this serves (measured, not a quality claim): R1/R2 models are
weak on basic Python recipes (max/min/sum/reverse/abs/literal+variable
binding) and existing corpus counts for those basics are tiny or zero.
This generator builds a small, balanced, standalone curriculum of BASIC
Python operations -- 77 Python recipe families based on the canonical
concepts in corpus/python_tasks.py and standard Python semantics, plus
11 simple bugfix families -- with deliberately varied natural phrasing (largest/maximum/
biggest, reverse/backwards, items/count/length) and varied identifiers,
numbers and quoted literals that must be reflected verbatim in the answer.

Everything is data: RECIPES at the bottom are declarative specs; the
engine below renders prompts from (wrapper x core-synonym) templates and
pools, builds answers AST-safely (repr() for literals, f-strings for
names -- never naive string replace), and validates parse-only
(ast.parse; NOTHING is ever exec'd/eval'd).

DEVELOPMENT split is deterministic and structural, not a random row slice:
dev uses dev-only phrasing wrappers+cores AND dev-only identifier/literal
pool values. Known operation families overlap between train and dev on
purpose: dev measures unseen surface forms + unseen parameter values, NOT
unseen API families, and common short answer overlap is unavoidable and
is not independence proof.

Contamination: train and dev are filtered through the EXACT existing
helpers (offline_python.filter_pairs -> postprocess.load_eval_overlap_sets,
sha256-pinned frozen suite-v1.1) with no weakened thresholds. Train is
additionally filtered against dev prompts (exact-normalized).

Outputs (refuses to overwrite anything existing, and refuses protected
corpus paths incl. corpus/results/):
  <out>/python-curriculum-train.jsonl  {id, prompt, answer, category
                                        (python|bugfix), tier, source,
                                        task_family, template_id}
  <out>/python-curriculum-dev.jsonl    eval-suite schema (id/category/
                                        prompt/answer/accept/scoring=
                                        python-ast/max_new_tokens/tier +
                                        source/task_family/template_id),
                                        works with eval/run_eval.py +
                                        score.py; accept = escaped collapsed
                                        first line of the reference answer
                                        (score.py scores the first line).
  <out>/manifest.json                  EXPERIMENTAL label, counts, family
                                        distribution, filter stats, notes.

Usage: python3 corpus/python_curriculum.py --out-dir /tmp/curriculum-r3
"""
import argparse
import ast
import itertools
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import postprocess as pp  # noqa: E402  (sha-pinned suite + filter helpers)
import offline_python as op  # noqa: E402  (trusted filter_pairs contract)

SOURCE = "python_curriculum"
DEFAULT_SEED = 42
TRAIN_COMBO_CAP = 64   # parameter diversity, not 32 rephrasings of 16 answers
DEV_COMBO_CAP = 4
TRAIN_FAMILY_CAP = 512 # balance family weight independently of combo count
TRAIN_FILE = "python-curriculum-train.jsonl"
DEV_FILE = "python-curriculum-dev.jsonl"
MANIFEST_FILE = "manifest.json"
# This generator must never touch existing corpus artifacts, and never
# writes into the shared corpus/results directory at all.
PROTECTED_OUTPUTS = tuple(
    os.path.join(HERE, *parts) for parts in (
        ("results", "full-train.jsonl"),
        ("results", "full.jsonl"),
        ("train-v1.jsonl"),
        ("offline.jsonl"),
    ))
FORBIDDEN_DIR = os.path.realpath(os.path.join(HERE, "results"))

PIPELINE_STATS = {}  # last run()'s stats, for tests/introspection


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Phrasing wrappers. Train and dev wrappers are disjoint surface forms;
# combined with per-recipe dev-only core synonyms this is the "phrasing
# template" half of the split.
# ---------------------------------------------------------------------------

WRAPPERS = [
    "How do you {core}?",
    "In Python, how can I {core}?",
    "What's the Python code to {core}?",
    "Show me how to {core}.",
    "Python help: {core}.",
    "I need to {core} -- how?",
    "Quick question: {core}?",
    "Write the Python to {core}.",
]

DEV_WRAPPERS = [
    "{core}, in Python?",
    "Any way to {core}?",
    "For Python, {core}?",
    "Remind me how to {core}.",
]


# ---------------------------------------------------------------------------
# Parameter pools. Train/dev slices are disjoint per pool; that is the
# "identifier/literal pools" half of the split. "kind" decides rendering:
#   name  -> bare identifier in prompt, same in code
#   quote -> "value" in prompt, repr(value) in code (pool restricted to
#            plain letters/spaces so repr never emits escapes)
#   bare  -> unquoted message words in prompt, repr(value) in code
#   num   -> decimal in prompt, repr in code (small signed ints/floats)
# ---------------------------------------------------------------------------

POOLS = {
    "list_var":     {"kind": "name", "train": [
        "nums", "values", "data", "items", "ages", "prices", "scores",
        "temps", "heights", "weights", "speeds", "times", "rates", "volts",
        "amps", "watts", "runs", "hits", "rows", "cols", "xs", "lst",
        "items1", "values2", "data_x"],
        "dev": ["marks", "points", "votes", "coins", "pips", "slots"]},
    "list_var_b":   {"kind": "name", "train": [
        "extra", "others", "second", "rest", "spare", "added", "newer",
        "older"], "dev": ["alt", "copy", "twin", "dual"]},
    "str_var":      {"kind": "name", "train": [
        "s", "text", "word", "line", "name", "caption", "note", "msg",
        "body", "head", "tail", "sign", "blob", "slug", "term", "unit"],
        "dev": ["entry", "reply", "greet", "chant", "motto", "label"]},
    "str_var_b":    {"kind": "name", "train": [
        "other", "suffix", "prefix", "ending", "piece", "chunk", "intro",
        "outro", "front", "back", "mid", "core", "rim", "wrap"],
        "dev": ["closer", "opener"]},
    "dict_var":     {"kind": "name", "train": [
        "d", "info", "config", "stock", "user", "props", "cache", "table",
        "fields", "record", "bag", "parts", "stats", "loads", "sheet",
        "roster"], "dev": ["specs", "menu", "codes", "cards"]},
    "set_var":      {"kind": "name", "train": [
        "seen", "tags", "flags", "kinds", "types", "group", "chosen",
        "picked"], "dev": ["roles", "modes"]},
    "set_var_b":    {"kind": "name", "train": ["known", "past"],
                     "dev": ["sides", "zones"]},
    "tuple_var":    {"kind": "name", "train": [
        "t", "pair", "trio", "quad", "slot", "node", "edge", "link",
        "step", "jump", "move", "turn", "fold", "wrap", "bind", "knot"],
                     "dev": ["duo", "box", "pack"]},
    "num_var":      {"kind": "name", "train": [
        "n", "x", "value", "count", "total", "num", "amount", "size",
        "width", "limit", "base", "rate", "depth", "span", "level",
        "grade"], "dev": ["q", "r", "v", "g"]},
    "num_var_b":    {"kind": "name", "train": ["y", "m", "w", "k", "z"],
                     "dev": ["h", "p"]},
    "item_var":     {"kind": "name", "train": ["item", "elem", "val", "unit",
                                               "part", "chip"],
                     "dev": ["bit", "nib"]},
    "idx_var":      {"kind": "name", "train": ["i", "idx", "pos", "at"],
                     "dev": ["j", "spot"]},
    "fn_name":      {"kind": "name", "train": [
        "add", "sum2", "plus", "tally", "sum_all", "add_all", "sum_up",
        "add_up"], "dev": ["combine", "sum_n"]},
    "word_lit":     {"kind": "quote", "train": [
        "apple", "banana", "cherry", "grape", "lemon", "melon", "peach",
        "berry", "mango", "olive", "pear", "lime", "corn", "rice", "oats",
        "salt"], "dev": ["kiwi", "plum", "fig", "date"]},
    "word_lit_b":   {"kind": "quote", "train": [
        "toast", "soup", "cake", "pie", "jam", "tea"],
        "dev": ["milk", "egg"]},
    "key_lit":      {"kind": "quote", "train": [
        "name", "age", "city", "price", "color", "score", "email", "phone"],
        "dev": ["title", "year"]},
    "sep_lit":      {"kind": "quote", "train": [", ", "-", " ", "_", ". "],
                     "dev": ["/", "; "]},
    "small_int":    {"kind": "num", "train": [
        2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50, 60, 100,
        -3, -7], "dev": [11, 14, 35, -9]},
    "small_int_b":  {"kind": "num", "train": [1, 2, 3, 4, 5, 6, 8, 10],
                     "dev": [13, 17]},
    "float_val":    {"kind": "num", "train": [
        2.5, 3.14, 0.5, 7.25, 1.75, 9.5, 4.25, 8.75, 6.5, 2.75, 5.25, 0.25,
        3.5, 6.25, 8.5, 1.5], "dev": [4.5, 6.75, 9.25, 0.75]},
    "step_int":     {"kind": "num", "train": [2, 3, 5], "dev": [11]},
    "range_stop":   {"kind": "num", "train": [10, 12, 15, 20, 25, 30, 40, 50],
                     "dev": [17, 19, 35]},
    "range_start":  {"kind": "num", "train": [1, 2, 3, 4, 5, 6, 7, 8],
                     "dev": [11, 13]},
    "numeric_text": {"kind": "quote", "train": [
        "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "15", "20",
        "25", "30", "40", "50", "-3", "-7"], "dev": ["11", "14", "35", "-9"]},
    "message_lit": {"kind": "bare", "train": [
        "hello there", "good morning", "well done", "ready now", "fresh start",
        "nice work", "welcome back", "keep going", "stay curious", "all ready",
        "blue sky", "green grass", "sunny day", "warm tea", "new idea", "good news"],
        "dev": ["good luck", "all clear", "bright future", "safe travels"]},
}


def _render_prompt(kind, value):
    if kind in ("name", "bare"):
        return value
    if kind == "quote":
        return '"%s"' % value
    return repr(value)


def _render_code(kind, value):
    if kind == "name":
        return value
    return repr(value)


# ---------------------------------------------------------------------------
# AST helpers (parse-only; semantics of generated code is NEVER evaluated).
# ---------------------------------------------------------------------------

def answer_identifiers(code):
    """Every identifier the answer binds/uses: Names, def/class names,
    function parameters, etc. -- so tests can prove the prompt's named
    variables actually reach the answer."""
    names = set()
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.alias):
            names.add(node.name.split(".")[0])
    return names


def answer_constants(code):
    """Every literal constant value (folding unary minus into ints/floats),
    so tests can prove prompt literals reach the answer verbatim."""
    values = []
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Constant) and isinstance(
                node.value, (int, float, str)) and not isinstance(node.value, bool):
            values.append(node.value)
        elif (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
              and isinstance(node.operand, ast.Constant)
              and isinstance(node.operand.value, (int, float))
              and not isinstance(node.operand.value, bool)):
            values.append(-node.operand.value)
    return values


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def _recipe_templates(recipe, split):
    """(template_id, prompt_skeleton) pairs: wrapper x core synonym, in
    fixed order. Dev template ids carry a dev/ prefix so no template id
    can ever be shared across splits."""
    cores = recipe["dev_cores"] if split == "dev" else recipe["cores"]
    wrappers = DEV_WRAPPERS if split == "dev" else WRAPPERS
    tag = "dev-c" if split == "dev" else "c"
    out = []
    for ci, core in enumerate(cores):
        for wi, wrapper in enumerate(wrappers):
            out.append((f"{recipe['family']}/{tag}{ci}w{wi}",
                        wrapper.format(core=core)))
    return out


def _combos(recipe, split):
    """Deterministic diverse parameter tuples. Never stride a Cartesian
    product: strides divisible by a pool size freeze trailing parameters.
    A complementary second tuple varies every feasible axis; the remaining
    sample is shuffled with a family/split-local seed (CLI seed is order only).
    """
    keys = list(recipe["slots"])
    pools = [POOLS[recipe["slots"][k]][split] for k in keys]
    combos = [dict(zip(keys, tup)) for tup in itertools.product(*pools)]
    guard = recipe.get("guard")
    if guard:
        combos = [c for c in combos if guard(c)]
    cap = DEV_COMBO_CAP if split == "dev" else TRAIN_COMBO_CAP
    if len(combos) > cap:
        random.Random(f"{recipe['family']}:{split}:params-v2").shuffle(combos)
        first = combos.pop(0)
        opposite = max(range(len(combos)),
                       key=lambda i: sum(combos[i][k] != first[k] for k in keys))
        second = combos.pop(opposite)
        combos = [first, second] + combos[:cap - 2]
    return combos


def build_candidates(split):
    """Balanced deterministic sample of each family's templates and combos.
    Records carry a _bindings debug dict (stripped before writing)."""
    assert split in ("train", "dev")
    records = []
    for recipe in RECIPES:
        combos = _combos(recipe, split)
        templates = _recipe_templates(recipe, split)
        prepared = []
        for combo in combos:
            code = {k: _render_code(POOLS[recipe["slots"][k]]["kind"], v)
                    for k, v in combo.items()}
            answer = recipe["answer"](code)
            extra = recipe["prompt_extra"](code) if recipe.get("prompt_extra") else {}
            if extra:
                for v in code.values():
                    assert v in extra.get("bug", ""), (recipe["family"], v)
            rendered = {k: _render_prompt(POOLS[recipe["slots"][k]]["kind"], v)
                        for k, v in combo.items()}
            prepared.append((combo, answer, {**rendered, **extra}))
        # Round-robin covers all sampled parameter tuples and all surface
        # templates without letting multi-parameter recipes dominate by size.
        cap = TRAIN_FAMILY_CAP if split == "train" else len(prepared) * len(templates)
        seq = 0
        for round_index in range(len(templates)):
            for ci, (combo, answer, rendered) in enumerate(prepared):
                if seq >= cap:
                    break
                tid, skeleton = templates[(round_index + ci) % len(templates)]
                records.append({
                    "id": f"curr-tr-{recipe['family']}-{seq:04d}" if split == "train"
                    else f"curr-dev-{recipe['family']}-{seq:04d}",
                    "prompt": skeleton.format(**rendered), "answer": answer,
                    "category": recipe["category"], "tier": recipe["tier"],
                    "source": SOURCE, "task_family": recipe["family"],
                    "template_id": tid, "_bindings": combo,
                })
                seq += 1
    return records


def _check_prompt(prompt):
    return (prompt.isascii() and "\n" not in prompt and "\\" not in prompt
            and len(prompt.encode("utf-8")) <= 127
            and all(32 <= ord(c) < 127 for c in prompt))


def _apply_prompt_and_syntax_gates(records, stats):
    kept = []
    for rec in records:
        if not _check_prompt(rec["prompt"]):
            stats["drop_prompt_shape"] += 1
            continue
        try:
            ast.parse(rec["answer"])  # full exec-mode snippet or construction bug
        except SyntaxError:
            stats["syntax_fail"] += 1
            continue
        kept.append(rec)
    return kept


def assert_writable(out_dir):
    real = os.path.realpath(out_dir)
    if os.path.isdir(real):
        existing = [n for n in (TRAIN_FILE, DEV_FILE, MANIFEST_FILE)
                    if os.path.exists(os.path.join(real, n))]
        if existing:
            raise SystemExit(f"refusing to overwrite existing outputs in {out_dir}: "
                             f"{', '.join(existing)}")
    for protected in PROTECTED_OUTPUTS:
        if real == os.path.realpath(protected):
            raise SystemExit(f"refusing to write onto corpus artifact {protected}")
    if real == FORBIDDEN_DIR or real.startswith(FORBIDDEN_DIR + os.sep):
        raise SystemExit(f"refusing to write into shared corpus results dir "
                         f"{FORBIDDEN_DIR}")


def run(out_dir, seed=DEFAULT_SEED):
    """Build both splits, filter, write outputs + manifest. Byte-identical
    for identical inputs. Returns the manifest dict."""
    assert_writable(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    stats = {"train": {"drop_prompt_shape": 0, "syntax_fail": 0,
                       "drop_train_dev_prompt": 0, "drop_train_dev_pair": 0},
             "dev": {"drop_prompt_shape": 0, "syntax_fail": 0}}

    train_raw = build_candidates("train")
    dev_raw = build_candidates("dev")

    # exact-template-id disjointness is a construction guarantee
    tr_ids = {r["template_id"] for r in train_raw}
    dv_ids = {r["template_id"] for r in dev_raw}
    if tr_ids & dv_ids:
        raise SystemExit(f"template id leak across splits: {sorted(tr_ids & dv_ids)[:5]}")

    # shared contamination contract (sha-pinned frozen suite, no weakening)
    dev_kept, dev_stats = op.filter_pairs(dev_raw)
    train_kept, train_stats = op.filter_pairs(train_raw)
    stats["dev"].update(dev_stats)
    stats["train"].update(train_stats)

    dev_kept = _apply_prompt_and_syntax_gates(dev_kept, stats["dev"])
    train_kept = _apply_prompt_and_syntax_gates(train_kept, stats["train"])

    # train must not reuse dev prompts or pairs (belt-and-braces on top of
    # the structural template/pool split)
    dev_prompts = {pp.normalize_key(r["prompt"]) for r in dev_kept}
    dev_pairs = {(pp.normalize_key(r["prompt"]), pp.normalize_key(r["answer"]))
                 for r in dev_kept}
    filtered = []
    for rec in train_kept:
        pn = pp.normalize_key(rec["prompt"])
        if pn in dev_prompts:
            stats["train"]["drop_train_dev_prompt"] += 1
            continue
        if (pn, pp.normalize_key(rec["answer"])) in dev_pairs:
            stats["train"]["drop_train_dev_pair"] += 1
            continue
        filtered.append(rec)
    train_kept = filtered

    rng = random.Random(seed)
    rng.shuffle(train_kept)
    random.Random(seed + 1).shuffle(dev_kept)

    train_path = os.path.join(out_dir, TRAIN_FILE)
    dev_path = os.path.join(out_dir, DEV_FILE)
    with open(train_path, "w") as f:
        for rec in train_kept:
            f.write(json.dumps({k: v for k, v in rec.items()
                                if not k.startswith("_")}, sort_keys=True) + "\n")
    with open(dev_path, "w") as f:
        for rec in dev_kept:
            first = " ".join(rec["answer"].splitlines()[0].split())
            f.write(json.dumps({
                "id": rec["id"],
                "category": rec["category"],
                "prompt": rec["prompt"],
                "answer": rec["answer"],
                "accept": ["^" + re.escape(first) + "$"],
                "scoring": "python-ast",
                "max_new_tokens": 64,
                "tier": rec["tier"],
                "source": rec["source"],
                "task_family": rec["task_family"],
                "template_id": rec["template_id"],
            }, sort_keys=True) + "\n")

    fam_stats = {}
    for rec in train_kept:
        fam_stats.setdefault(rec["task_family"], {"train_count": 0, "dev_count": 0})
        fam_stats[rec["task_family"]]["train_count"] += 1
    for rec in dev_kept:
        fam_stats.setdefault(rec["task_family"], {"train_count": 0, "dev_count": 0})
        fam_stats[rec["task_family"]]["dev_count"] += 1
    for recipe in RECIPES:
        fam = fam_stats.setdefault(recipe["family"],
                                   {"train_count": 0, "dev_count": 0})
        fam.update(category=recipe["category"], tier=recipe["tier"],
                   mutates=recipe.get("mutates", False),
                   derived_from=recipe["derived"],
                   templates_train=len(recipe["cores"]) * len(WRAPPERS),
                   templates_dev=len(recipe["dev_cores"]) * len(DEV_WRAPPERS),
                   parameter_values_train={slot: len({r['_bindings'][slot]
                       for r in train_kept if r['task_family'] == recipe['family']})
                       for slot in recipe['slots']},
                   parameter_values_dev={slot: len({r['_bindings'][slot]
                       for r in dev_kept if r['task_family'] == recipe['family']})
                       for slot in recipe['slots']})

    manifest = {
        "label": ("EXPERIMENTAL -- data-coverage curriculum for the R3 "
                  "hypothesis; NOT a capability certification"),
        "source": SOURCE,
        "seed": seed,
        "train_count": len(train_kept),
        "dev_count": len(dev_kept),
        "train_file": TRAIN_FILE,
        "dev_file": DEV_FILE,
        "train_sha256": sha256_file(train_path),
        "dev_sha256": sha256_file(dev_path),
        "train_combo_cap": TRAIN_COMBO_CAP,
        "train_family_cap": TRAIN_FAMILY_CAP,
        "parameter_sampling": "complementary tuples + stable shuffle; round-robin templates",
        "dev_combo_cap": DEV_COMBO_CAP,
        "families": fam_stats,
        "mutating_families": [r["family"] for r in RECIPES if r.get("mutates")],
        "filter_stats": stats,
        "prompt_bytes_max": max(len(r["prompt"].encode()) for r in train_kept),
        "answer_chars_max": max(len(r["answer"]) for r in train_kept),
        "split_policy": (
            "dev = dev-only wrappers x dev-only core synonyms x dev-only "
            "pool values; train filtered against dev prompts/pairs; "
            "operation families OVERLAP by design -- dev is an unseen "
            "surface/parameter diagnostic, not unseen-API-family "
            "generalization"),
        "notes": [
            "contamination filtered via offline_python.filter_pairs -> "
            "postprocess.load_eval_overlap_sets (sha256-pinned "
            "suite-v1.1); thresholds unchanged",
            "all answers are full exec-parseable snippets (no fragments); "
            "validated parse-only, never executed",
            "common short answer overlap between train and dev (e.g. "
            "max(nums)-shaped code) is unavoidable and is NOT independence "
            "proof",
            "dev scoring=python-ast checks score.py's first scored line "
            "against the escaped reference; use eval/audit_python.py for "
            "full-reference AST diagnostics (conservative: valid "
            "alternatives are false negatives)",
            "no escaped-newline/backslash literal prompts; ASCII one-line "
            "prompts <= 127 bytes; device budget 256 incl. prompt is "
            "re-checked by the parent's exporter with the real tokenizer",
        ],
    }
    with open(os.path.join(out_dir, MANIFEST_FILE), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    PIPELINE_STATS.clear()
    PIPELINE_STATS.update(stats)
    return manifest


# ---------------------------------------------------------------------------
# Declarative recipes. `derived` cites the trusted corpus/python_tasks.py
# task(s) each family is derived from. `answer` builds code from
# code-rendered slots; `guard` skips semantically invalid combos (e.g.
# range(a, b) with a >= b). Bugfix recipes additionally build the buggy
# line via prompt_extra.
# ---------------------------------------------------------------------------

def _R(family, category, tier, derived, cores, dev_core, slots, answer,
       mutates=False, guard=None, prompt_extra=None):
    return {"family": family, "category": category, "tier": tier,
            "derived": derived, "cores": cores, "dev_cores": [dev_core],
            "slots": slots, "answer": answer, "mutates": mutates,
            "guard": guard, "prompt_extra": prompt_extra}


RECIPES = [
    # Foundational recipes/literal binding (trusted print, variables,
    # conversions, len, and slicing tasks; no evaluation-suite templates).
    _R("print_literal", "python", 1, ["print: print a message to the console"],
       ["print the text {lit}", "display the string {lit} with print",
        "write {lit} to the console", "output the literal {lit} using print"],
       "send the string {lit} to standard output",
       {"lit": "word_lit"}, lambda p: f"print({p['lit']})"),
    _R("print_variable", "python", 1, ["print: print a variable value"],
       ["print the value of {n}", "display the variable {n}",
        "output what {n} contains", "show {n} on the console"],
       "send the value of {n} to standard output",
       {"n": "num_var"}, lambda p: f"print({p['n']})"),
    _R("print_message", "python", 1, ["print: print a message to the console"],
       ["print the words {msg}", "display the message {msg}",
        "output the text {msg}", "write the message {msg} to the console"],
       "send the message {msg} to standard output",
       {"msg": "message_lit"}, lambda p: f"print({p['msg']})"),
    _R("assign_literal", "python", 1, ["variables: assign a number to a variable"],
       ["set the variable {n} to {k}", "assign {k} to {n}",
        "store the number {k} in {n}", "give {n} the value {k}"],
       "bind the name {n} to the number {k}",
       {"n": "num_var", "k": "small_int"}, lambda p: f"{p['n']} = {p['k']}"),
    _R("str_int_literal", "python", 1, ["type conversion: string to integer"],
       ["convert the string {lit} to an integer", "parse the text {lit} as an int",
        "turn the quoted number {lit} into an integer", "get an integer from {lit}"],
       "read the numeric string {lit} as a whole number",
       {"lit": "numeric_text"}, lambda p: f"int({p['lit']})"),
    _R("num_str_literal", "python", 1, ["strings: convert a number to a string"],
       ["convert the number {k} to a string", "turn the integer {k} into text",
        "get the string form of {k}", "represent the number {k} as a string"],
       "render the integer {k} as text",
       {"k": "small_int"}, lambda p: f"str({p['k']})"),
    _R("abs_literal", "python", 1, ["math: get the absolute value of a number"],
       ["get the absolute value of {k}", "find how far {k} is from zero",
        "remove the sign from {k}", "compute abs of {k}"],
       "calculate the absolute value of the number {k}",
       {"k": "small_int"}, lambda p: f"abs({p['k']})"),
    _R("str_len_literal", "python", 1, ["strings: get the length of a string"],
       ["count the characters in {lit}", "get the length of the string {lit}",
        "find how many characters {lit} contains", "return the character count of {lit}"],
       "measure the length of the text {lit}",
       {"lit": "word_lit"}, lambda p: f"len({p['lit']})"),
    _R("list_first", "python", 1, ["lists: get the first item of a list"],
       ["get the first item of the list {xs}", "read the first element of {xs}",
        "fetch the value at index zero in {xs}", "return the first value from {xs}"],
       "select the element at the start of {xs}",
       {"xs": "list_var"}, lambda p: f"{p['xs']}[0]"),
    _R("list_last", "python", 1, ["lists: get the last item of a list"],
       ["get the last item of the list {xs}", "read the final element of {xs}",
        "fetch the value at index minus one in {xs}", "return the last value from {xs}"],
       "select the element at the end of {xs}",
       {"xs": "list_var"}, lambda p: f"{p['xs']}[-1]"),
    _R("list_first_n", "python", 1, ["slicing: get the first few items"],
       ["get the first {k} items of {xs} as a new list", "take the first {k} elements of {xs}",
        "slice {xs} to keep its first {k} values", "copy the first {k} entries from {xs}"],
       "select the first {k} elements from the list {xs}",
       {"xs": "list_var", "k": "small_int"}, lambda p: f"{p['xs']}[:{p['k']}]",
       guard=lambda c: c["k"] > 0),
    _R("list_sorted_desc", "python", 1, ["sorting: sort a list in descending order"],
       ["get a descending sorted copy of {xs}", "sort {xs} largest first without changing it",
        "make a new list from {xs} sorted highest to lowest", "return {xs} sorted in reverse order as a copy"],
       "produce a new descending-order list from {xs}",
       {"xs": "list_var"}, lambda p: f"sorted({p['xs']}, reverse=True)"),
    # ---------------------------------------------------------- lists
    _R("list_max", "python", 1, ["sorting: find the largest value in a list"],
       ["find the largest value in the list {xs}",
        "get the maximum value in the list {xs}",
        "find the biggest number in the list {xs}",
        "return the highest value in the list {xs}"],
       "pick out the top value from the list {xs}",
       {"xs": "list_var"}, lambda p: f"max({p['xs']})"),
    _R("list_min", "python", 1, ["sorting: find the smallest value in a list"],
       ["find the smallest value in the list {xs}",
        "get the minimum value in the list {xs}",
        "find the lowest number in the list {xs}",
        "return the tiniest value in the list {xs}"],
       "pick out the bottom value from the list {xs}",
       {"xs": "list_var"}, lambda p: f"min({p['xs']})"),
    _R("list_sum", "python", 1, ["ranges: get the sum of numbers from 1 to 100"],
       ["add up all the numbers in the list {xs}",
        "get the total of the list {xs}",
        "sum every value in the list {xs}",
        "compute the sum of the items in {xs}"],
       "give the combined sum of all values in the list {xs}",
       {"xs": "list_var"}, lambda p: f"sum({p['xs']})"),
    _R("list_len", "python", 1, ["len: get the number of items in a list"],
       ["count the items in the list {xs}",
        "get the length of the list {xs}",
        "find how many elements {xs} holds",
        "tell me the number of values in {xs}"],
       "say the item count of the list {xs}",
       {"xs": "list_var"}, lambda p: f"len({p['xs']})"),
    _R("list_count", "python", 1, ["lists: check if a value is in a list"],
       ["count how many times {k} appears in the list {xs}",
        "get the number of occurrences of {k} in {xs}",
        "tally how often {k} shows up in the list {xs}",
        "find the count of {k} inside {xs}"],
       "number the appearances of {k} in the list {xs}",
       {"xs": "list_var", "k": "small_int"},
       lambda p: f"{p['xs']}.count({p['k']})"),
    _R("list_append", "python", 1, ["lists: add an item to the end of a list"],
       ["add the value {k} to the end of the list {xs}",
        "append {k} to the list {xs}",
        "attach {k} onto the end of {xs}",
        "put {k} at the end of the list {xs}, changing it in place"],
       "stick the value {k} onto the list {xs}",
       {"xs": "list_var", "k": "small_int"},
       lambda p: f"{p['xs']}.append({p['k']})", mutates=True),
    _R("list_concat", "python", 1, ["lists: combine two lists into one"],
       ["combine the lists {xs} and {ys} into a new list",
        "put {xs} and {ys} together without changing either",
        "build one list out of {xs} followed by {ys}",
        "join {xs} and {ys} into a fresh list"],
       "glue the lists {xs} and {ys} into one new list",
       {"xs": "list_var", "ys": "list_var_b"},
       lambda p: f"{p['xs']} + {p['ys']}", mutates=False),
    _R("list_contains", "python", 1, ["lists: check if a value is in a list"],
       ["check whether the value {k} is in the list {xs}",
        "see if {xs} contains {k}",
        "test if {k} appears anywhere in {xs}",
        "check whether {k} is one of the elements of the list {xs}"],
       "determine whether the list {xs} holds the value {k}",
       {"xs": "list_var", "k": "small_int"},
       lambda p: f"{p['k']} in {p['xs']}"),
    _R("list_sorted_copy", "python", 1, ["sorting: sort a list in ascending order"],
       ["get a sorted copy of the list {xs} in ascending order",
        "return the items of {xs} sorted, leaving {xs} untouched",
        "make a new ascending list from {xs} without sorting it in place",
        "sort the values of {xs} into a new list, smallest first"],
       "hand back an ordered copy of the list {xs}",
       {"xs": "list_var"}, lambda p: f"sorted({p['xs']})", mutates=False),
    _R("list_sort_inplace", "python", 1, ["sorting: sort a list in place"],
       ["sort the list {xs} in place",
        "arrange {xs} into ascending order itself",
        "sort {xs} directly without making a copy",
        "reorder the list {xs} in place, smallest first"],
       "tidy the list {xs} into order in place",
       {"xs": "list_var"}, lambda p: f"{p['xs']}.sort()", mutates=True),
    _R("list_reverse_slice", "python", 2, ["slicing: reverse a list using slicing"],
       ["reverse the list {xs} using slicing",
        "get {xs} back as a new list written backwards",
        "produce a reversed copy of {xs} with a slice",
        "read the list {xs} backwards into a new list"],
       "flip the list {xs} around with a slice",
       {"xs": "list_var"}, lambda p: f"{p['xs']}[::-1]", mutates=False),
    _R("list_reverse_inplace", "python", 2, ["loops: loop over a list backwards"],
       ["reverse the list {xs} in place",
        "flip {xs} around itself, no copy",
        "turn the list {xs} backwards directly",
        "reorder {xs} back to front, mutating it"],
       "turn the list {xs} around in place",
       {"xs": "list_var"}, lambda p: f"{p['xs']}.reverse()", mutates=True),
    # -------------------------------------------------------- strings
    _R("str_upper", "python", 1, ["string methods: convert a string to uppercase"],
       ["convert the string {s} to uppercase",
        "make an uppercase version of {s}",
        "get {s} in capital letters",
        "change {s} to all caps"],
       "shout the string {s} in capitals",
       {"s": "str_var"}, lambda p: f"{p['s']}.upper()"),
    _R("str_lower", "python", 1, ["string methods: convert a string to lowercase"],
       ["convert the string {s} to lowercase",
        "make a lowercase version of {s}",
        "get {s} in small letters",
        "change {s} to all lowercase"],
       "quiet the string {s} into lowercase",
       {"s": "str_var"}, lambda p: f"{p['s']}.lower()"),
    _R("str_strip", "python", 1, ["string methods: strip whitespace"],
       ["remove the leading and trailing whitespace from {s}",
        "trim the spaces off both ends of {s}",
        "strip {s} of surrounding whitespace",
        "clean the outer whitespace from the string {s}"],
       "neaten the string {s} by trimming its ends",
       {"s": "str_var"}, lambda p: f"{p['s']}.strip()"),
    _R("str_split", "python", 1, ["string methods: split into words"],
       ["split the string {s} into a list of words",
        "break {s} apart on whitespace",
        "chop {s} into words",
        "turn {s} into a word list"],
       "shatter the string {s} into words",
       {"s": "str_var"}, lambda p: f"{p['s']}.split()"),
    _R("str_split_sep", "python", 2, ["string methods: split"],
       ["split the string {s} on every {sep}",
        "break {s} apart wherever {sep} occurs",
        "cut {s} into pieces divided by {sep}",
        "divide the string {s} using {sep} as the separator"],
       "carve the string {s} up on {sep}",
       {"s": "str_var", "sep": "sep_lit"},
       lambda p: f"{p['s']}.split({p['sep']})"),
    _R("str_join", "python", 2, ["string methods: join a list of strings"],
       ["join the list {xs} into one string with {sep} between items",
        "glue the words in {xs} together using {sep}",
        "combine {xs} into a single string separated by {sep}",
        "build one string out of {xs} with {sep} between the parts"],
       "string the list {xs} along with {sep}",
       {"xs": "list_var", "sep": "sep_lit"},
       lambda p: f"{p['sep']}.join({p['xs']})"),
    _R("str_replace", "python", 1, ["string methods: replace a substring"],
       ['replace every {lit} in {s} with {lit2}',
        'swap all occurrences of {lit} for {lit2} in the string {s}',
        'change {lit} to {lit2} throughout {s}',
        'substitute {lit2} for {lit} in the string {s}'],
       'trade every {lit} in {s} for {lit2}',
       {"s": "str_var", "lit": "word_lit", "lit2": "word_lit_b"},
       lambda p: f"{p['s']}.replace({p['lit']}, {p['lit2']})"),
    _R("str_startswith", "python", 1, ["string methods: startswith"],
       ["check whether {s} starts with {lit}",
        "see if the string {s} begins with {lit}",
        "test if {lit} is the opening of {s}",
        "check if {s} opens with {lit}"],
       "confirm the string {s} kicks off with {lit}",
       {"s": "str_var", "lit": "word_lit"},
       lambda p: f"{p['s']}.startswith({p['lit']})"),
    _R("str_contains", "python", 1, ["strings: check if a string contains a substring"],
       ["check whether {s} contains {lit}",
        "see if the substring {lit} appears in {s}",
        "test if {lit} occurs inside the string {s}",
        "check if {lit} is part of {s}"],
       "determine whether the string {s} includes {lit}",
       {"s": "str_var", "lit": "word_lit"},
       lambda p: f"{p['lit']} in {p['s']}"),
    _R("str_len", "python", 1, ["len: number of characters in a string"],
       ["count the characters in the string {s}",
        "get the length of {s}",
        "find how many characters {s} has",
        "tell me the character count of {s}"],
       "say the length of the string {s}",
       {"s": "str_var"}, lambda p: f"len({p['s']})"),
    _R("str_concat", "python", 1, ["strings: concatenate two strings"],
       ["concatenate the strings {s} and {t}",
        "join {s} and {t} into one string",
        "stick {t} onto the end of {s}",
        "build one string from {s} plus {t}"],
       "meld the strings {s} and {t} together",
       {"s": "str_var", "t": "str_var_b"},
       lambda p: f"{p['s']} + {p['t']}"),
    # ----------------------------------------------------------- dicts
    _R("dict_get", "python", 1, ["dicts: get a value by key"],
       ["get the value stored under the key {key} in {d}",
        "read {key} from the dictionary {d}",
        "look up the {key} entry of {d}",
        "fetch what {d} has at {key}"],
       "pull the {key} value out of the dict {d}",
       {"d": "dict_var", "key": "key_lit"},
       lambda p: f"{p['d']}[{p['key']}]"),
    _R("dict_get_default", "python", 2, ["dicts: get with a default if missing"],
       ["get {key} from {d}, falling back to {k} if it is missing",
        "read the {key} entry of {d} with default {k}",
        "look up {key} in {d} and use {k} when it is absent",
        "fetch {key} out of {d}, or {k} if there is no such key"],
       "grab {key} from the dict {d} with fallback {k}",
       {"d": "dict_var", "key": "key_lit", "k": "small_int_b"},
       lambda p: f"{p['d']}.get({p['key']}, {p['k']})"),
    _R("dict_set", "python", 1, ["dicts: add a key-value pair"],
       ["set the key {key} in {d} to {k}",
        "store {k} under {key} in the dictionary {d}",
        "add or update the {key} entry of {d} with {k}",
        "put {k} into {d} at the key {key}"],
       "write {k} under {key} in the dict {d}",
       {"d": "dict_var", "key": "key_lit", "k": "small_int"},
       lambda p: f"{p['d']}[{p['key']}] = {p['k']}", mutates=True),
    _R("dict_has_key", "python", 1, ["dicts: check if a key exists"],
       ["check whether the key {key} exists in {d}",
        "see if {d} already has {key}",
        "test if {key} is one of the keys of {d}",
        "check whether the dictionary {d} contains {key}"],
       "confirm the dict {d} carries the key {key}",
       {"d": "dict_var", "key": "key_lit"},
       lambda p: f"{p['key']} in {p['d']}"),
    _R("dict_keys", "python", 1, ["dicts: get all keys"],
       ["get all the keys in the dictionary {d}",
        "list the keys of {d}",
        "collect every key {d} has",
        "view just the key side of {d}"],
       "hand me the keys of the dict {d}",
       {"d": "dict_var"}, lambda p: f"{p['d']}.keys()"),
    _R("dict_values", "python", 1, ["dicts: values"],
       ["get all the values in the dictionary {d}",
        "list the values of {d}",
        "collect every value {d} holds",
        "view just the value side of {d}"],
       "hand me the values of the dict {d}",
       {"d": "dict_var"}, lambda p: f"{p['d']}.values()"),
    _R("dict_pop", "python", 2, ["lists: remove the last item and get it"],
       ["remove the key {key} from {d} and get its value",
        "pop the {key} entry out of the dictionary {d}",
        "delete {key} from {d}, returning what it held",
        "take away {key} in {d} and keep its value"],
       "yank the {key} entry out of the dict {d}",
       {"d": "dict_var", "key": "key_lit"},
       lambda p: f"{p['d']}.pop({p['key']})", mutates=True),
    _R("dict_from_zip", "python", 2, ["zip: make a dictionary from two lists"],
       ["build a dictionary from the keys list {xs} and the values list {ys}",
        "map the items of {xs} to the items of {ys} in a dict",
        "construct a dict pairing {xs} with {ys}",
        "turn the parallel lists {xs} and {ys} into one dictionary"],
       "shape a dict out of the lists {xs} and {ys}",
       {"xs": "list_var", "ys": "list_var_b"},
       lambda p: f"dict(zip({p['xs']}, {p['ys']}))"),
    # ------------------------------------------------------------ sets
    _R("set_union", "python", 2, ["sets: find the union of two sets"],
       ["find the union of the sets {a} and {b}",
        "merge {a} and {b} keeping every member",
        "combine sets {a} and {b} into one",
        "get everything in {a} or {b}"],
       "unite the sets {a} and {b}",
       {"a": "set_var", "b": "set_var_b"},
       lambda p: f"{p['a']} | {p['b']}"),
    _R("set_intersection", "python", 2, ["sets: find the intersection"],
       ["find the intersection of the sets {a} and {b}",
        "get what {a} and {b} share",
        "collect the members common to {a} and {b}",
        "keep only the items in both {a} and {b}"],
       "find the shared members of the sets {a} and {b}",
       {"a": "set_var", "b": "set_var_b"},
       lambda p: f"{p['a']} & {p['b']}"),
    _R("set_difference", "python", 2, ["sets: find the difference"],
       ["find the difference between the sets {a} and {b}",
        "get the members of {a} that are not in {b}",
        "subtract set {b} from set {a}",
        "keep what {a} has after removing {b}'s items"],
       "trim the set {b} away from the set {a}",
       {"a": "set_var", "b": "set_var_b"},
       lambda p: f"{p['a']} - {p['b']}"),
    _R("set_add", "python", 1, ["sets: add an item to a set"],
       ["add the value {k} to the set {a}",
        "put {k} into {a}",
        "insert {k} as a member of the set {a}",
        "grow the set {a} with {k}"],
       "drop the value {k} into the set {a}",
       {"a": "set_var", "k": "small_int"},
       lambda p: f"{p['a']}.add({p['k']})", mutates=True),
    _R("set_remove", "python", 1, ["sets: remove an item"],
       ["remove the value {k} from the set {a}",
        "take {k} out of {a}",
        "delete the member {k} from the set {a}",
        "shrink the set {a} by dropping {k}"],
       "erase the value {k} from the set {a}",
       {"a": "set_var", "k": "small_int"},
       lambda p: f"{p['a']}.remove({p['k']})", mutates=True),
    _R("set_contains", "python", 1, ["sets: check membership"],
       ["check whether the value {k} is in the set {a}",
        "see if {a} contains {k}",
        "test if {k} is a member of the set {a}",
        "check whether {k} is one of the elements of {a}"],
       "determine whether the set {a} holds the value {k}",
       {"a": "set_var", "k": "small_int"},
       lambda p: f"{p['k']} in {p['a']}"),
    _R("set_from_list", "python", 1, ["sets: create a set from a list"],
       ["make a set from the list {xs}, removing duplicates",
        "deduplicate {xs} by turning it into a set",
        "get the unique values of {xs} as a set",
        "convert {xs} into a set so repeats vanish"],
       "squeeze the list {xs} into a set",
       {"xs": "list_var"}, lambda p: f"set({p['xs']})"),
    # ---------------------------------------------------------- tuples
    _R("tuple_index", "python", 1, ["tuples: get the first item of a tuple"],
       ["get the item at index {k} of the tuple {t}",
        "read position {k} out of {t}",
        "fetch the element at zero-based index {k} of the tuple {t}",
        "look up spot {k} in {t}"],
       "grab index {k} of the tuple {t}",
       {"t": "tuple_var", "k": "small_int_b"},
       lambda p: f"{p['t']}[{p['k']}]"),
    _R("tuple_len", "python", 1, ["len: number of items in a tuple"],
       ["count the items in the tuple {t}",
        "get the length of {t}",
        "find how many elements {t} holds",
        "tell me the size of the tuple {t}"],
       "say the item count of the tuple {t}",
       {"t": "tuple_var"}, lambda p: f"len({p['t']})"),
    _R("tuple_unpack", "python", 1, ["tuples: unpack into two variables"],
       ["unpack the tuple {t} into the variables {a} and {b}",
        "split {t} apart into {a} and {b}",
        "assign the two parts of {t} to {a} and {b}",
        "take {t} apart in one line as {a}, {b}"],
       "open the tuple {t} up into {a} and {b}",
       {"t": "tuple_var", "a": "num_var", "b": "num_var_b"},
       lambda p: f"{p['a']}, {p['b']} = {p['t']}"),
    _R("tuple_from_list", "python", 1, ["tuples: convert a list to a tuple"],
       ["convert the list {xs} to a tuple",
        "turn {xs} into a tuple",
        "make a tuple out of the list {xs}",
        "repackage {xs} as a tuple"],
       "recast the list {xs} as a tuple",
       {"xs": "list_var"}, lambda p: f"tuple({p['xs']})"),
    # ------------------------------------------- conversions / numerics
    _R("str_to_int", "python", 1, ["type conversion: string to integer"],
       ["convert the string {s} to an integer",
        "parse {s} as an int",
        "turn the string {s} into a whole number",
        "get the integer value of {s}"],
       "read the string {s} as an integer",
       {"s": "str_var"}, lambda p: f"int({p['s']})"),
    _R("str_to_float", "python", 1, ["type conversion: string to float"],
       ["convert the string {s} to a float",
        "parse {s} as a floating point number",
        "turn the string {s} into a decimal number",
        "get the float value of {s}"],
       "read the string {s} as a float",
       {"s": "str_var"}, lambda p: f"float({p['s']})"),
    _R("num_to_str", "python", 1, ["strings: convert a number to a string"],
       ["convert the number {n} to a string",
        "turn {n} into text",
        "get the string form of {n}",
        "represent {n} as a string"],
       "render the number {n} as a string",
       {"n": "num_var"}, lambda p: f"str({p['n']})"),
    _R("float_trunc", "python", 1, ["type conversion: float to int, truncating"],
       ["convert the float {f} to an integer, truncating it",
        "chop {f} down to a whole number",
        "drop the decimals off {f} with int",
        "truncate {f} toward zero"],
       "cut the float {f} to an integer",
       {"f": "float_val"}, lambda p: f"int({p['f']})"),
    _R("abs_value", "python", 1, ["math: get the absolute value of a number"],
       ["get the absolute value of {n}",
        "find how far {n} is from zero",
        "strip the sign off {n}",
        "compute abs of {n}"],
       "take the absolute value of the number {n}",
       {"n": "num_var"}, lambda p: f"abs({p['n']})"),
    _R("round_num", "python", 1, ["math: round to nearest integer"],
       ["round the number {f} to the nearest integer",
        "round {f} off",
        "find the nearest whole number to {f}",
        "round {f} with no decimal places"],
       "round the value {f} off",
       {"f": "float_val"}, lambda p: f"round({p['f']})"),
    _R("round_places", "python", 1, ["math: round to 2 decimal places"],
       ["round the number {f} to 2 decimal places",
        "round {f} to two decimals",
        "round the numeric value {f} to two decimal places",
        "round {f} off after the second decimal"],
       "round the value {f} to a couple of decimals",
       {"f": "float_val"}, lambda p: f"round({p['f']}, 2)"),
    _R("floor_div", "python", 1, ["math: get the floor of a division"],
       ["divide {n} by {k} using floor division",
        "divide {n} by {k} and round the quotient down",
        "integer-divide {n} by {k}, rounding down",
        "compute {n} floor-divided by {k}"],
       "floor the division of {n} by {k}",
       {"n": "num_var", "k": "num_var_b"},
       lambda p: f"{p['n']} // {p['k']}"),
    _R("mod", "python", 1, ["math: get the remainder of a division"],
       ["get the remainder of {n} divided by {k}",
        "find {n} modulo {k}",
        "compute what is left over from {n} over {k}",
        "take the remainder of {n} by {k}"],
       "find the leftover of {n} divided by {k}",
       {"n": "num_var", "k": "num_var_b"},
       lambda p: f"{p['n']} % {p['k']}"),
    _R("power", "python", 1, ["math: raise a number to a power"],
       ["raise {n} to the power {k}",
        "compute {n} to the {k}th power",
        "exponentiate {n} by {k}",
        "get {n} raised to {k}"],
       "take {n} up to the power {k}",
       {"n": "num_var", "k": "num_var_b"},
       lambda p: f"{p['n']} ** {p['k']}"),
    # ------------------------------------------------ ranges/enumerate/zip
    _R("range_stop", "python", 1, ["ranges: make a range from 0 to n"],
       ["make a range from 0 up to but not including {k}",
        "get the numbers 0 through {k} minus one as a range",
        "build range({k})",
        "create a range that stops before {k}"],
       "form a range running from zero to just under {k}",
       {"k": "small_int"}, lambda p: f"range({p['k']})",
       guard=lambda c: c["k"] > 0),
    _R("range_start_stop", "python", 1, ["ranges: make a range from a to b"],
       ["make a range from {a} to {b}, excluding {b}",
        "build range({a}, {b})",
        "get the numbers {a} up to just below {b}",
        "create a range starting at {a} and stopping before {b}"],
       "form a range from {a} running up to but excluding {b}",
       {"a": "range_start", "b": "range_stop"},
       lambda p: f"range({p['a']}, {p['b']})",
       guard=lambda c: c["a"] < c["b"]),
    _R("range_step", "python", 2, ["ranges: make a range counting by steps"],
       ["make a range from {a} to before {b} counting by {step}",
        "build range({a}, {b}, {step})",
        "count from {a} toward {b} in steps of {step}",
        "create a stepped range: start {a}, stop {b}, step {step}"],
       "form a range from {a} to before {b} hopping by {step}",
       {"a": "range_start", "b": "range_stop", "step": "step_int"},
       lambda p: f"range({p['a']}, {p['b']}, {p['step']})",
       guard=lambda c: c["a"] < c["b"]),
    _R("range_to_list", "python", 1, ["ranges: convert a range to a list"],
       ["convert the range from {a} to {b} into a list",
        "expand the numbers from {a} up to but excluding {b} into a list",
        "materialize range({a}, {b}) as a list",
        "get every number from {a} to just under {b} in a list"],
       "turn the range from {a} to before {b} into a list",
       {"a": "range_start", "b": "range_stop"},
       lambda p: f"list(range({p['a']}, {p['b']}))",
       guard=lambda c: c["a"] < c["b"]),
    _R("enumerate_loop", "python", 2, ["enumerate: loop with index and value"],
       ["loop over the list {xs} printing each index {i} and value {x}",
        "walk {xs} with both position and item as {i} and {x}",
        "iterate {xs} so {i} counts and {x} is the element, printing both",
        "go through {xs} printing the place {i} beside the value {x}"],
       "run through the list {xs} printing {i} with {x}",
       {"xs": "list_var", "i": "idx_var", "x": "item_var"},
       lambda p: f"for {p['i']}, {p['x']} in enumerate({p['xs']}):\n"
                 f"    print({p['i']}, {p['x']})"),
    _R("zip_loop", "python", 2, ["zip: loop over two lists in parallel"],
       ["loop over {xs} and {ys} at the same time, printing each pair as {a} and {b}",
        "walk the lists {xs} and {ys} together with {a} and {b}",
        "iterate {xs} beside {ys} so {a} meets {b}, printing both",
        "go through {xs} and {ys} in step, printing {a} with {b}"],
       "run through the lists {xs} and {ys} side by side printing {a} and {b}",
       {"xs": "list_var", "ys": "list_var_b", "a": "num_var", "b": "num_var_b"},
       lambda p: f"for {p['a']}, {p['b']} in zip({p['xs']}, {p['ys']}):\n"
                 f"    print({p['a']}, {p['b']})"),
    # ----------------------------------- functions/conditionals/comps
    _R("def_add", "python", 2, ["functions: define a function that adds two numbers"],
       ["define a function {fn} that takes {a} and {b} and returns their sum",
        "write def {fn}({a}, {b}) returning {a} plus {b}",
        "create {fn} to add its two parameters {a} and {b}",
        "make a function {fn} that returns {a} + {b}"],
       "put together a function {fn} adding {a} and {b}",
       {"fn": "fn_name", "a": "num_var", "b": "num_var_b"},
       lambda p: f"def {p['fn']}({p['a']}, {p['b']}): return {p['a']} + {p['b']}"),
    _R("def_default", "python", 2, ["default args: optional argument"],
       ["define {fn}({a}, {b}) where {b} defaults to {k}, returning their sum",
        "write def {fn}({a}, {b}={k}) that adds {a} and {b}",
        "create {fn}({a}, {b}) so {b} is optional with default {k}",
        "make a function {fn}({a}, {b}) whose second argument {b} falls back to {k}"],
       "shape a function {fn}({a}, {b}) with {b} defaulting to {k}",
       {"fn": "fn_name", "a": "num_var", "b": "num_var_b", "k": "small_int_b"},
       lambda p: f"def {p['fn']}({p['a']}, {p['b']}={p['k']}): "
                 f"return {p['a']} + {p['b']}"),
    _R("lambda_double", "python", 2, ["lambda: assign a lambda to a variable"],
       ["assign a lambda that multiplies {x} by {k} to {fn}",
        "write {fn} = lambda {x}: {x} * {k}",
        "bind a function {fn} that scales {x} by {k}",
        "set {fn} to an inline function multiplying {x} by {k}"],
       "hook a lambda scaling {x} by {k} onto {fn}",
       {"fn": "fn_name", "x": "num_var", "k": "small_int_b"},
       lambda p: f"{p['fn']} = lambda {p['x']}: {p['x']} * {p['k']}"),
    _R("even_check", "python", 1, ["conditionals: check if a number is even"],
       ["check whether the number {n} is even",
        "test if {n} divides evenly by 2",
        "write the even test for {n}",
        "check whether {n} is an even number"],
       "confirm whether the number {n} comes out even",
       {"n": "num_var"}, lambda p: f"{p['n']} % 2 == 0"),
    _R("squares_comp", "python", 2, ["comprehensions: list of squares"],
       ["build a list of the squares of each {x} from 0 up to {k} minus one",
        "make a list of {x} squared for each {x} below {k}",
        "list the squares of every {x} from 0 to before {k} with a comprehension",
        "collect the squares of each {x} under {k} into a list"],
       "gather the squares of each {x} below {k}",
       {"x": "item_var", "k": "small_int"},
       lambda p: f"[{p['x']} ** 2 for {p['x']} in range({p['k']})]",
       guard=lambda c: c["k"] > 0),
    _R("filter_comp", "python", 2, ["comprehensions: keep only some numbers"],
       ["collect only the {x} values greater than {k} from the list {xs}",
        "filter {xs} down to each {x} over {k}",
        "keep each {x} of {xs} that beats {k}",
        "make a list of every {x} in {xs} with {x} above {k}"],
       "sift the list {xs} for each {x} above {k}",
       {"xs": "list_var", "x": "item_var", "k": "small_int"},
       lambda p: f"[{p['x']} for {p['x']} in {p['xs']} if {p['x']} > {p['k']}]"),
    _R("upper_comp", "python", 2, ["comprehensions: uppercase every string"],
       ["make a new list with every string {x} in {xs} uppercased",
        "uppercase each {x} in the list {xs} into a fresh list",
        "build the all-caps version of each {x} in the word list {xs}",
        "convert every {x} of {xs} to capitals in a new list"],
       "lift every string {x} in {xs} into capitals in a new list",
       {"xs": "list_var", "x": "item_var"},
       lambda p: f"[{p['x']}.upper() for {p['x']} in {p['xs']}]"),
]

# ------------------------------------------------------------ bugfix
# Simple, unambiguous single-error one-line fixes. The buggy line is built
# AST-safely from the same code-rendered slots; the prompt stays one line.


def _buggy(line):
    return lambda p: {"bug": line(p)}


RECIPES += [
    _R("bf_if_colon", "bugfix", 1, ["conditionals: if statements"],
       ["fix the syntax error in this line: {bug}",
        "repair this broken Python line: {bug}",
        "this line has one error -- correct it: {bug}",
        "spot and fix the mistake: {bug}"],
       "patch this line: {bug}",
       {"n": "num_var", "k": "small_int"},
       lambda p: f"if {p['n']} > {p['k']}: print({p['n']})",
       prompt_extra=_buggy(lambda p: f"if {p['n']} > {p['k']} print({p['n']})")),
    _R("bf_for_colon", "bugfix", 1, ["loops: for statements"],
       ["fix the syntax error in this line: {bug}",
        "repair this broken Python line: {bug}",
        "this line has one error -- correct it: {bug}",
        "spot and fix the mistake: {bug}"],
       "patch this line: {bug}",
       {"i": "idx_var", "k": "small_int"},
       lambda p: f"for {p['i']} in range({p['k']}): print({p['i']})",
       prompt_extra=_buggy(lambda p: f"for {p['i']} in range({p['k']}) print({p['i']})"),
       guard=lambda c: c["k"] > 0),
    _R("bf_def_colon", "bugfix", 1, ["functions: def statements"],
       ["fix the syntax error in this line: {bug}",
        "repair this broken Python line: {bug}",
        "this line has one error -- correct it: {bug}",
        "spot and fix the mistake: {bug}"],
       "patch this line: {bug}",
       {"fn": "fn_name", "a": "num_var", "k": "small_int_b"},
       lambda p: f"def {p['fn']}({p['a']}): return {p['a']} * {p['k']}",
       prompt_extra=_buggy(lambda p: f"def {p['fn']}({p['a']}) return {p['a']} * {p['k']}")),
    _R("bf_false_typo", "bugfix", 1, ["conditionals: booleans"],
       ["fix the typo in this line: {bug}",
        "repair this broken Python line: {bug}",
        "this line has one error -- correct it: {bug}",
        "spot and fix the mistake: {bug}"],
       "patch this line: {bug}",
       {"n": "num_var"},
       lambda p: f"{p['n']} = False",
       prompt_extra=_buggy(lambda p: f"{p['n']} = flase")),
    _R("bf_eq_assign", "bugfix", 1, ["conditionals: comparison operators"],
       ["fix the broken comparison in this line: {bug}",
        "repair this bad Python comparison: {bug}",
        "this comparison line has one error -- correct it: {bug}",
        "spot and fix the comparison mistake: {bug}"],
       "patch this comparison line: {bug}",
       {"n": "num_var", "k": "small_int"},
       lambda p: f"if {p['n']} == {p['k']}: print({p['n']})",
       prompt_extra=_buggy(lambda p: f"if {p['n']} = {p['k']}: print({p['n']})")),
    _R("bf_unclosed_paren", "bugfix", 1, ["print: function calls"],
       ["fix the syntax error in this line: {bug}",
        "repair this broken Python line: {bug}",
        "this line has one error -- correct it: {bug}",
        "spot and fix the mistake: {bug}"],
       "patch this line: {bug}",
       {"lit": "word_lit"},
       lambda p: f"print({p['lit']})",
       prompt_extra=_buggy(lambda p: f"print({p['lit']}")),
    _R("bf_concat_int", "bugfix", 2, ["strings: concatenation needs str"],
       ["fix the bug in this line: {bug}",
        "repair this broken Python line: {bug}",
        "this line has one error -- correct it: {bug}",
        "spot and fix the mistake: {bug}"],
       "patch this line: {bug}",
       {"lit": "word_lit", "k": "small_int_b"},
       lambda p: f"print({p['lit']} + str({p['k']}))",
       prompt_extra=_buggy(lambda p: f"print({p['lit']} + {p['k']})")),
    _R("bf_list_add", "bugfix", 1, ["lists: append method"],
       ["fix this method call on the list {xs}: {bug}",
        "repair this line for a list named {xs}: {bug}",
        "correct this attempt to append to list {xs}: {bug}",
        "fix the wrong list method for {xs}: {bug}"],
       "patch this call on the list {xs}: {bug}",
       {"xs": "list_var", "k": "small_int"},
       lambda p: f"{p['xs']}.append({p['k']})",
       prompt_extra=_buggy(lambda p: f"{p['xs']}.add({p['k']})"), mutates=True),
    _R("bf_append_reassign", "bugfix", 2, ["lists: append returns None"],
       ["fix this append so {xs} stays a list: {bug}",
        "repair this line without replacing list {xs}: {bug}",
        "keep {xs} as a list when fixing this append: {bug}",
        "fix the unwanted reassignment of list {xs}: {bug}"],
       "patch this append without losing the list {xs}: {bug}",
       {"xs": "list_var", "k": "small_int"},
       lambda p: f"{p['xs']}.append({p['k']})",
       prompt_extra=_buggy(lambda p: f"{p['xs']} = {p['xs']}.append({p['k']})"), mutates=True),
    _R("bf_len_typo", "bugfix", 1, ["len: builtin name"],
       ["fix the typo in this line: {bug}",
        "repair this broken Python line: {bug}",
        "this line has one error -- correct it: {bug}",
        "spot and fix the mistake: {bug}"],
       "patch this line: {bug}",
       {"xs": "list_var"},
       lambda p: f"print(len({p['xs']}))",
       prompt_extra=_buggy(lambda p: f"print(lenght({p['xs']}))")),
    _R("bf_print_typo", "bugfix", 1, ["print: builtin name"],
       ["fix the typo in this line: {bug}",
        "repair this broken Python line: {bug}",
        "this line has one error -- correct it: {bug}",
        "spot and fix the mistake: {bug}"],
       "patch this line: {bug}",
       {"lit": "word_lit"},
       lambda p: f"print({p['lit']})",
       prompt_extra=_buggy(lambda p: f"prnt({p['lit']})")),
]

FAMILY_NAMES = [r["family"] for r in RECIPES]
RECIPE_BY_FAMILY = {r["family"]: r for r in RECIPES}
assert len(FAMILY_NAMES) == len(set(FAMILY_NAMES)), "duplicate family name"

# Construction invariant: every parameter slot must be named in every
# prompt template (bugfix recipes route slots through {bug}, checked per
# combo at build time) -- otherwise identical prompts would map to
# different answers, which is exactly the literal/variable-binding noise
# this curriculum exists to fix.
for _r in RECIPES:
    for _cores in (_r["cores"], _r["dev_cores"]):
        for _core in _cores:
            if _r.get("prompt_extra"):
                assert "{bug}" in _core, (_r["family"], _core)
            else:
                for _slot in _r["slots"]:
                    assert ("{%s}" % _slot) in _core, (_r["family"], _slot, _core)


def main():
    ap = argparse.ArgumentParser(
        description="Deterministic API-free basic-Python curriculum "
                    "generator (EXPERIMENTAL R3 data experiment). "
                    "See docs/PYTHON-CURRICULUM.md.")
    ap.add_argument("--out-dir", required=True,
                    help="output directory (must not already contain "
                         "curriculum outputs; corpus paths are refused)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="shuffle seed (default 42); content is "
                         "seed-independent, order is not")
    args = ap.parse_args()
    manifest = run(args.out_dir, args.seed)
    print(f"wrote {manifest['train_count']} train + {manifest['dev_count']} dev "
          f"pairs -> {args.out_dir} (label: {manifest['label']})")
    cats = {}
    for fam in manifest["families"].values():
        cats[fam["category"]] = cats.get(fam["category"], 0) + 1
    print(f"families: {len(manifest['families'])} "
          f"({', '.join(f'{c}={n}' for c, n in sorted(cats.items()))})")
    fs = manifest["filter_stats"]
    for split in ("train", "dev"):
        drops = {k: v for k, v in fs[split].items()
                 if k.startswith("drop_") and v}
        print(f"  {split} filters: candidates={fs[split]['candidates']} "
              f"kept={fs[split]['kept']} drops={drops or 'none'}")


if __name__ == "__main__":
    main()
