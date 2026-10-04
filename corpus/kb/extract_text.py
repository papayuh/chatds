#!/usr/bin/env python3
"""Full plain text of every Simple English Wikipedia article (build_kb's cleaner), for LM pretraining.
Paragraphs and articles are separated by blank lines. kb-probe-150 articles are left out so that eval stays held out."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_kb
from kb import normalize

ROOT = os.path.join(build_kb.HERE, "..", "..")
MIN_ARTICLE = 400  # chars

def probe_titles():
    out = set()
    for l in open(os.path.join(ROOT, "eval", "kb-probe-150.jsonl")):
        r = json.loads(l)
        p, a = normalize(r["prompt"]).split(), normalize(r["answer"]).split()
        k = max((k for k in range(1, min(len(p), len(a)) + 1) if p[-k:] == a[:k]), default=0)
        if k:
            out.add(" ".join(a[:k]))
    return out

def main(out_path):
    arts, _ = build_kb.parse(os.path.join(build_kb.HERE, "cache", "simplewiki-latest-pages-articles.xml.bz2"),
                             extract=build_kb.extract_text, min_text=MIN_ARTICLE)
    held = probe_titles()
    kept = {t: x for t, x in arts.items() if normalize(t) not in held}
    with open(out_path, "w") as f:
        f.write("\n\n".join(kept.values()) + "\n")
    print(json.dumps(dict(articles=len(kept), held_out=len(arts) - len(kept), probe_titles=len(held),
                          chars=sum(map(len, kept.values())))))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "corpus", "results", "simplewiki-text.txt"))
