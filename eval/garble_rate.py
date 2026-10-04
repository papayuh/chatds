#!/usr/bin/env python3
"""Garble rate: fraction of answers containing an alphabetic word that is in none of: the retrieved context, the question,
the common-English wordlist (lowercase-only entries of /usr/share/dict/words, so proper nouns must come from the context).
    python3 eval/garble_rate.py --suite eval/kb-probe-150.jsonl --outputs run/eval/x.jsonl [--show]"""
import argparse, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "corpus"))
import chatds_runtime as rt  # noqa: E402
from fact_answers import stem  # noqa: E402

def wl(path="/usr/share/dict/words"):
    return {w.strip() for w in open(path) if w.strip().isalpha() and w.strip().islower()}

def toks(s):
    return re.findall(r"[a-z]+", s.lower())

def garbled(answer, question, ctx, common):
    known = {stem(w) for w in toks((ctx or "") + " " + question)} | {w for w in toks((ctx or "") + " " + question)}
    return [w for w in toks(answer) if w not in known and stem(w) not in known and w not in common and stem(w) not in common
            and w.rstrip("s") not in common and w.rstrip("d") not in common and w[:-2] not in common]

def garble_rate(suite, outputs, kb, common=None, show=False):
    common = common or wl()
    out = {r["id"]: r["output"].strip() for r in map(json.loads, open(outputs)) if r.get("id")}
    n = g = 0
    for item in map(json.loads, open(suite)):
        if item["id"] not in out:
            continue
        ctx = rt.retrieve(item["prompt"], kb)
        bad = garbled(out[item["id"]], item["prompt"], ctx, common)
        n += 1; g += bool(bad)
        if show and bad:
            print(item["id"], item["prompt"], "->", out[item["id"]], bad)
    return g, n

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True); ap.add_argument("--outputs", required=True)
    ap.add_argument("--kb", default=os.path.join(HERE, "..", "corpus/kb/out/simplewiki.kb")); ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    g, n = garble_rate(a.suite, a.outputs, a.kb, show=a.show)
    print("garble rate %d/%d = %.1f%% (answers with a word outside context, question and English wordlist)" % (g, n, 100 * g / max(n, 1)))
