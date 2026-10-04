#!/usr/bin/env python3
"""Copy rate: fraction of fact answers whose longest common word run with the retrieved context is > 6 words.
    python3 eval/copy_rate.py --suite eval/chatds-acceptance-30.jsonl --outputs run/eval/x.jsonl [--kb KB]
Counts only fact-category items where the runtime retrieves a context and the answer is not the idk line."""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "corpus"))
import chatds_runtime as rt  # noqa: E402
from fact_answers import longest_run, grounded, IDK  # noqa: E402


def copy_rate(suite, outputs, kb):
    out = {r["id"]: r["output"].strip() for r in map(json.loads, open(outputs)) if r.get("id")}
    n = c = g = 0
    for item in map(json.loads, open(suite)):
        if item["category"] not in ("facts", "factual") or item["id"] not in out:
            continue
        ctx = rt.retrieve(item["prompt"], kb)
        if ctx and out[item["id"]] != IDK:
            n += 1
            c += longest_run(out[item["id"]], ctx) > 6
            g += grounded(out[item["id"]], item["prompt"], ctx)
    return c, n, g


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True); ap.add_argument("--outputs", required=True)
    ap.add_argument("--kb", default=os.path.join(HERE, "..", "corpus/kb/out/simplewiki.kb"))
    a = ap.parse_args()
    c, n, g = copy_rate(a.suite, a.outputs, a.kb)
    print("copy rate %d/%d = %.1f%%; grounded (no content word outside context/question) %d/%d" % (c, n, 100 * c / max(n, 1), g, n))
