#!/usr/bin/env python3
"""Garble rate split by how the OLD runtime (80-char cut) would have ended the context: sentence (fit whole), clause (cut
at a comma/semicolon) or midphrase (cut at a word boundary). The kind always comes from the KB2 record in --old-kb.
    python3 eval/garble_split.py --suite eval/kb-probe-150.jsonl --outputs RUN.jsonl --kb corpus/kb/out/simplewiki-v2.kb --legacy
    python3 eval/garble_split.py ... --kb corpus/kb/out/simplewiki.kb          (KB3 run: same split, contexts are clean)
--legacy applies the old 80-char cut to --kb records (needed to score a run made with a KB2 KB)."""
import argparse, collections, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "corpus"))
import chatds_runtime as rt  # noqa: E402
import garble_rate as gr  # noqa: E402


def old_cut(text):
    """The deleted runtime best_sentence (CTX_MAX 80). Returns (context, kind)."""
    s = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text)[0]
    if len(s) <= 80:
        return s, "sentence"
    full = s[:80].rsplit(" ", 1)[0]
    c = max(full.rfind(","), full.rfind(";"))
    return (full[:c], "clause") if c >= 40 else (full, "midphrase")


def split(suite, outputs, kb, old_kb, legacy, facts_only=False):
    common, rows, bad = gr.wl(), collections.defaultdict(lambda: [0, 0]), []
    out = {r["id"]: r["output"].strip() for r in map(json.loads, open(outputs)) if r.get("id")}
    for it in map(json.loads, open(suite)):
        if it["id"] not in out or (facts_only and it.get("category") != "facts"):
            continue
        raw = rt.retrieve(it["prompt"], kb)
        ctx = old_cut(raw)[0] if raw and legacy else raw
        old = rt.retrieve(it["prompt"], old_kb)
        kind = old_cut(old)[1] if old else "noctx"
        w = gr.garbled(out[it["id"]], it["prompt"], ctx, common)
        rows[kind][0] += bool(w); rows[kind][1] += 1
        if w:
            bad.append((it["id"], kind, out[it["id"]], w))
    return rows, bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True); ap.add_argument("--outputs", required=True)
    ap.add_argument("--kb", default=os.path.join(HERE, "..", "corpus/kb/out/simplewiki.kb"))
    ap.add_argument("--old-kb", default=os.path.join(HERE, "..", "corpus/kb/out/simplewiki-v2.kb"))
    ap.add_argument("--legacy", action="store_true"); ap.add_argument("--facts-only", action="store_true")
    a = ap.parse_args()
    rows, bad = split(a.suite, a.outputs, a.kb, a.old_kb, a.legacy, a.facts_only)
    for k, (g, n) in sorted(rows.items()):
        print("%-10s %2d/%-3d %.1f%%" % (k, g, n, 100 * g / n))
    cut = rows["midphrase"]; clean = [sum(rows[k][i] for k in ("sentence", "clause")) for i in (0, 1)]
    print("clean (sentence+clause) %d/%d, cut mid-phrase %d/%d" % (*clean, *cut))
    for b in bad:
        print(*b)
