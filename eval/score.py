#!/usr/bin/env python3
"""Score a model-output jsonl against a frozen eval suite. Stdlib only.

Usage:
  score.py --suite eval/suite-v1.1.jsonl --outputs run.jsonl [--json]
  score.py --suite eval/suite-v1.1.jsonl --outputs run.jsonl --subset   # 60-item DS subset
  score.py --suite eval/suite-v1.1.jsonl --selftest        # canonical answers, must be 100%
  score.py --suite eval/suite-v1.1.jsonl --negative-selftest  # forged runs must FAIL

STRICT BY DEFAULT (Sol sprint-1 hardening, objection 5). A run is only a number
if it is a complete, well-formed run. Any violation prints to stderr and exits 1:

  * outputs line is not a JSON object, or `id` is missing / unknown / duplicated
  * `output` is not a string; `tokens_generated` / `bytes_read` / `startup_bytes`
    not a nonnegative integer
  * coverage: an expected id has no output line (300 by default, 60 with --subset)

Per-item scoring violations do not abort the run, they score the item WRONG:

  * `tokens_generated` > the item's `max_new_tokens` (the run broke its own cap)
  * the scored line is longer than the item's character budget
    (8 chars per allowed token, floor 64) -- this is the cap check for runs that
    do not report, or lie about, their token count. A dump of every canonical
    answer on one line is thousands of characters and now scores 0.

outputs jsonl, one object per line:
  {"id": "fact-001", "output": "...", "tokens_generated": 12,
   "bytes_read": 9437184, "startup_bytes": 0}
`bytes_read` is the RECURRING per-answer read (weights streamed to answer that
item). `startup_bytes` is the one-time cold-start read (a RAM-resident model has
bytes_read ~ 0 and startup_bytes = the whole model). Both optional. When present
the report emits amortized bytes/correct-answer at N = 1, 10, 100, 1000 answers
(Sol kernel-study objection 7) -- a single number collapses for resident models.

Scoring rules (see README.md):
  exact       normalized output == some accept entry
  contains    any accept entry appears in the normalized output, at word
              boundaries ("nice" does not contain "ice")
  regex       any accept entry, compiled IGNORECASE, matches the scored line
              (use an inline (?-i:...) group where case matters)
  python-ast  as regex, AND the scored line must parse as Python
  numeric     the LAST number on the scored line equals `answer`
  judge       not scored here; reported as unscored (suite v1/v1.1 have none)
Optional `reject`: a list of regexes; if any matches the scored line the item is
wrong regardless of `accept`. This is what stops a bugfix item from accepting the
unchanged buggy line.
Only the first non-empty line of `output` is scored, and a leading echo of the
prompt is dropped.
"""
import argparse, ast, contextlib, io, json, re, sys
from collections import defaultdict

NUM = re.compile(r"-?\d+(?:\.\d+)?")
SCORINGS = {"exact", "contains", "regex", "python-ast", "numeric", "judge"}
CHARS_PER_TOKEN = 8       # generous upper bound for any vocab we ship
MIN_CHAR_BUDGET = 64
AMORTIZE_N = (1, 10, 100, 1000)


def first_line(output, prompt):
    s = output or ""
    if prompt and s.lstrip().lower().startswith(prompt.lower()):
        s = s.lstrip()[len(prompt):]
    for line in s.splitlines():
        if line.strip():
            return " ".join(line.split())
    return ""


def norm(s):
    s = " ".join(s.split()).strip().lower()
    return s.strip("`\"' \t").rstrip(".!?,;:")


def py_parses(line):
    """True if `line` is plausible Python. A header line ending in ':' is
    allowed a synthesized `pass` body; nothing else is synthesized, so
    `except X as e:` without a `try` still fails."""
    for cand in (line, line + " pass", "if 1:\n    " + line + " pass"):
        try:
            ast.parse(cand)
            return True
        except (SyntaxError, ValueError):
            pass
    return False


def correct(item, raw_line):
    """None = judge (unscored). True/False otherwise."""
    sc, accept = item["scoring"], item["accept"]
    if sc == "judge":
        return None
    for p in item.get("reject", ()):
        if re.search(p, raw_line, re.IGNORECASE):
            return False
    if sc in ("regex", "python-ast"):
        if not any(re.search(p, raw_line, re.IGNORECASE) for p in accept):
            return False
        return py_parses(raw_line) if sc == "python-ast" else True
    if sc == "numeric":
        found = NUM.findall(raw_line)
        if not found:
            return False
        try:
            return abs(float(found[-1]) - float(item["answer"])) < 1e-6
        except ValueError:
            return False
    n = norm(raw_line)
    if sc == "exact":
        return any(n == norm(a) for a in accept)
    # contains, at WORD BOUNDARIES: "nice" is not "ice", "because" is not "Au".
    return any(re.search(r"(?<!\w)" + re.escape(norm(a)) + r"(?!\w)", n) for a in accept)


def char_budget(item):
    return max(MIN_CHAR_BUDGET, CHARS_PER_TOKEN * int(item["max_new_tokens"]))


# ---------------------------------------------------------------- loading

def load_suite(path):
    """Load and schema-check the suite itself. A malformed suite is fatal."""
    items, ids, bad = [], set(), []
    for n, line in enumerate(open(path), 1):
        if not line.strip():
            continue
        try:
            it = json.loads(line)
        except ValueError as e:
            bad.append("suite line %d: not JSON (%s)" % (n, e)); continue
        for k in ("id", "category", "prompt", "answer", "accept", "scoring",
                  "max_new_tokens"):
            if k not in it:
                bad.append("suite line %d: missing %r" % (n, k))
        it.setdefault("tier", 1)  # chatds-acceptance-30 has no tiers
        if not isinstance(it.get("accept"), list) or not it["accept"]:
            bad.append("suite line %d: accept must be a nonempty list" % n)
        if it.get("scoring") not in SCORINGS:
            bad.append("suite line %d: bad scoring %r" % (n, it.get("scoring")))
        if not isinstance(it.get("max_new_tokens"), int) or it["max_new_tokens"] <= 0:
            bad.append("suite line %d: max_new_tokens must be a positive int" % n)
        for p in list(it.get("accept", [])) + list(it.get("reject", [])):
            if it.get("scoring") in ("regex", "python-ast"):
                try:
                    re.compile(p)
                except re.error as e:
                    bad.append("suite line %d: bad regex %r (%s)" % (n, p, e))
        if it.get("id") in ids:
            bad.append("suite line %d: duplicate id %r" % (n, it.get("id")))
        ids.add(it.get("id"))
        items.append(it)
    if bad:
        for b in bad:
            print("SUITE SCHEMA ERROR: " + b, file=sys.stderr)
        sys.exit(2)
    return items


def load_outputs(lines, known_ids):
    """Return (by_id, violations). Strict: every line is validated.
    `lines` is any iterable of jsonl text lines (a file object, or a list)."""
    by_id, v = {}, []
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            o = json.loads(line)
        except ValueError as e:
            v.append("outputs line %d: not JSON (%s)" % (n, e)); continue
        if not isinstance(o, dict):
            v.append("outputs line %d: not a JSON object" % n); continue
        oid = o.get("id")
        if not isinstance(oid, str):
            v.append("outputs line %d: missing/non-string id" % n); continue
        if oid not in known_ids:
            v.append("outputs line %d: unknown id %r" % (n, oid)); continue
        if oid in by_id:
            v.append("outputs line %d: duplicate id %r" % (n, oid)); continue
        if not isinstance(o.get("output", ""), str):
            v.append("outputs line %d (%s): output is not a string" % (n, oid))
        for k in ("tokens_generated", "bytes_read", "startup_bytes"):
            if o.get(k) is None:
                continue
            val = o[k]
            if isinstance(val, bool) or not isinstance(val, int) or val < 0:
                v.append("outputs line %d (%s): %s must be a nonnegative integer, got %r"
                         % (n, oid, k, val))
                del o[k]   # never let a bad counter reach the metrics
        by_id[oid] = o
    return by_id, v


def subset_ids(suite):
    """The 60-item hardware subset: every id whose number is == 1 (mod 5)."""
    return {i["id"] for i in suite if int(i["id"].rsplit("-", 1)[1]) % 5 == 1}


# ---------------------------------------------------------------- reporting

def report(suite, by_id, expected, as_json=False, violations=()):
    violations = list(violations)
    cats, tiers = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    n_ok = n_scored = n_missing = n_judge = n_overlong = n_overcap = 0
    toks = 0
    byts = byts_seen = 0
    startups = set()
    for item in suite:
        if item["id"] not in expected:
            continue
        o = by_id.get(item["id"])
        if o is None:
            n_missing += 1
            continue
        line = first_line(o.get("output", "") if isinstance(o.get("output", ""), str) else "",
                          item["prompt"])
        ok = correct(item, line)
        if ok is None:
            n_judge += 1
            continue
        if ok and (item.get("require_tool") == "calc" or item["category"] == "math"):
            # chatds: the model must emit calc(...) itself; the runtime-evaluated
            # "output" alone could be a lucky guess. run_eval keeps the raw text.
            raw = (o.get("raw_output", o.get("output")) or "").strip()
            ok = bool(re.fullmatch(r"calc\([0-9 .()]*[-+*/%][0-9 .()+\-*/%]*\)", raw))
        tg = o.get("tokens_generated")
        if isinstance(tg, int) and tg > item["max_new_tokens"]:
            n_overcap += 1
            ok = False
        if len(line) > char_budget(item):
            n_overlong += 1
            ok = False
        n_scored += 1
        n_ok += bool(ok)
        cats[item["category"]][0] += bool(ok); cats[item["category"]][1] += 1
        tiers[item["tier"]][0] += bool(ok); tiers[item["tier"]][1] += 1
        toks += int(tg or 0)
        if o.get("bytes_read") is not None:
            byts += int(o["bytes_read"]); byts_seen += 1
        if o.get("startup_bytes") is not None:
            startups.add(int(o["startup_bytes"]))

    if n_missing:
        violations.append("coverage: %d of %d expected items have no output line"
                          % (n_missing, len(expected)))
    if len(startups) > 1:
        violations.append("startup_bytes disagrees across items: %s" % sorted(startups))
    if n_overcap:
        violations.append("%d item(s) exceeded max_new_tokens (scored wrong)" % n_overcap)
    if n_overlong:
        violations.append("%d item(s) emitted a line over the character budget "
                          "(scored wrong)" % n_overlong)

    acc = n_ok / n_scored if n_scored else 0.0
    res = {
        "expected": len(expected), "scored": n_scored, "correct": n_ok,
        "missing": n_missing, "judge_unscored": n_judge,
        "over_token_cap": n_overcap, "overlong_line": n_overlong,
        "accuracy": acc,
        "mean_tokens_per_answer": toks / n_scored if n_scored else 0.0,
        "per_category": {c: {"correct": v[0], "n": v[1], "accuracy": v[0] / v[1]} for c, v in sorted(cats.items())},
        "per_tier": {str(t): {"correct": v[0], "n": v[1], "accuracy": v[0] / v[1]} for t, v in sorted(tiers.items())},
        "violations": violations,
    }
    if byts_seen or startups:
        startup = max(startups) if startups else 0
        recur = byts / byts_seen if byts_seen else 0.0
        res["startup_bytes"] = startup
        res["recurring_bytes_per_answer"] = recur
        res["bytes_items_with_bytes"] = byts_seen
        # Sol kernel-study #7: one number collapses for a RAM-resident model
        # (recurring ~ 0 regardless of size or quality). Amortize the cold start.
        res["bytes_per_correct_answer_at_N"] = {
            str(N): ((startup + N * recur) / (N * acc)) if acc else float("inf")
            for N in AMORTIZE_N
        }

    if as_json:
        print(json.dumps(res, indent=2))
    else:
        print("category      correct/n   acc")
        for c, v in res["per_category"].items():
            print("  %-12s %3d/%-3d  %5.1f%%" % (c, v["correct"], v["n"], 100 * v["accuracy"]))
        print("tier (1=3M-should, 2=8M-target, 3=reach)")
        for t, v in res["per_tier"].items():
            print("  %-12s %3d/%-3d  %5.1f%%" % (t, v["correct"], v["n"], 100 * v["accuracy"]))
        print("OVERALL       %3d/%-3d  %5.1f%%" % (n_ok, n_scored, 100 * acc))
        print("coverage      %3d/%-3d" % (n_scored + n_judge, len(expected)))
        print("mean tokens/answer      %.2f" % res["mean_tokens_per_answer"])
        if n_overcap:
            print("OVER max_new_tokens     %d (scored wrong)" % n_overcap)
        if n_overlong:
            print("OVERLONG lines          %d (scored wrong)" % n_overlong)
        if n_judge:
            print("judge items (unscored)  %d" % n_judge)
        if "bytes_per_correct_answer_at_N" in res:
            print("startup bytes           %.3f MB" % (res["startup_bytes"] / 1e6))
            print("recurring bytes/answer  %.3f MB (on %d items)"
                  % (res["recurring_bytes_per_answer"] / 1e6, res["bytes_items_with_bytes"]))
            print("BYTES/CORRECT ANSWER amortized over N answers")
            for N in AMORTIZE_N:
                print("  N=%-5d %10.3f MB" % (N, res["bytes_per_correct_answer_at_N"][str(N)] / 1e6))

    for v in violations:
        print("VIOLATION: " + v, file=sys.stderr)
    return res


# ---------------------------------------------------------------- selftests

def selftest(suite, as_json):
    outs = [{"id": i["id"], "output": i["answer"], "tokens_generated": 0} for i in suite]
    by_id = {o["id"]: o for o in outs}
    res = report(suite, by_id, {i["id"] for i in suite}, as_json)
    bad = [i["id"] for i in suite
           if correct(i, first_line(i["answer"], i["prompt"])) is False]
    if bad or res["violations"] or res["correct"] != res["scored"]:
        print("SELFTEST FAIL, canonical answers not accepted: %s"
              % (", ".join(bad) or "(see violations)"), file=sys.stderr)
        return 1
    print("SELFTEST PASS (%d/%d)" % (res["correct"], res["scored"]))
    return 0


def negative_selftest(suite):
    """The three forged runs from Sol's objection 5 must all FAIL loudly.
    Runs them through the exact production path (load_outputs + report)."""
    all_ids = {i["id"] for i in suite}
    dump = " ".join(i["answer"] for i in suite)
    a0 = suite[0]
    cases = [
        ("single-answer run claiming accuracy=1.0",
         [{"id": a0["id"], "output": a0["answer"], "tokens_generated": 1}]),
        ("duplicate id with bytes_read=-100",
         [{"id": a0["id"], "output": a0["answer"], "tokens_generated": 1, "bytes_read": -100},
          {"id": a0["id"], "output": a0["answer"], "tokens_generated": 1, "bytes_read": -100}]),
        ("dump of every canonical answer on one line",
         [{"id": i["id"], "output": dump, "tokens_generated": 1} for i in suite]),
    ]
    fails = 0
    for name, outs in cases:
        by_id, v = load_outputs([json.dumps(o) for o in outs], all_ids)
        with contextlib.redirect_stdout(io.StringIO()):   # the table is noise here
            res = report(suite, by_id, all_ids, as_json=False, violations=v)
        bad = not res["violations"] and res["accuracy"] > 0.05
        print("  %-46s %3d/%-3d correct, %d violation(s) -> %s"
              % (name, res["correct"], res["scored"], len(res["violations"]),
                 "STILL PASSES (BAD)" if bad else "REJECTED (good)"))
        fails += bad
    if fails:
        print("NEGATIVE SELFTEST FAIL: %d forged run(s) still produce a usable number"
              % fails, file=sys.stderr)
        return 1
    print("NEGATIVE SELFTEST PASS (3/3 forged runs rejected)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="eval/suite-v1.1.jsonl")
    ap.add_argument("--outputs")
    ap.add_argument("--subset", action="store_true",
                    help="score only the 60-item hardware subset (ids == 1 mod 5); "
                         "without this, a run missing any of the 300 items FAILS")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--negative-selftest", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    suite = load_suite(a.suite)

    if a.negative_selftest:
        return negative_selftest(suite)
    if a.selftest:
        return selftest(suite, a.json)
    if not a.outputs:
        ap.error("--outputs required unless --selftest/--negative-selftest")

    expected = subset_ids(suite) if a.subset else {i["id"] for i in suite}
    with open(a.outputs) as f:
        by_id, viol = load_outputs(f, {i["id"] for i in suite})
    if a.subset:
        extra = set(by_id) - expected
        if extra:
            viol.append("--subset: %d output(s) outside the hardware subset: %s"
                        % (len(extra), ", ".join(sorted(extra)[:5])))
    res = report(suite, by_id, expected, a.json, viol)
    return 1 if res["violations"] else 0


if __name__ == "__main__":
    sys.exit(main())
