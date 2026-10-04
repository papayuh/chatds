#!/usr/bin/env python3
"""Short grounded fact answers (P2c). DeepSeek rewrites "C: context / Q: question" into a <=12 word answer in its own
words; every answer is validated against the context (no new content words, no long copied run).

    python3 corpus/fact_answers.py --gen [--cap 5.0] [--limit N]   # chatds-train.jsonl factual ctx rows -> results/fact-answers.jsonl
Helpers used elsewhere: valid(), longest_run() (build_chatds.py, eval/copy_rate.py)."""
import argparse, concurrent.futures as cf, json, os, re, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
IDK = "i don't know that one."
RAW = os.path.join(HERE, "results", "fact-answers-raw.jsonl")
OUT = os.path.join(HERE, "results", "fact-answers.jsonl")
SRC = os.path.join(HERE, "results", "chatds-train.jsonl")
BATCH = 20
FSTOP = set("""a an the of in on at to for from by with and or but as is are was were be been it its it's this that these those
there their they he she we you i me my your our his her him them us do does did done has have had not no so if than then
also can could will would may might s t one""".split())
SYSTEM = ("You answer short questions using ONLY the context line given with each item. Rules: reply in lowercase, casual, one short "
          "sentence of at most 12 words, ending with a period. Answer the question; do not restate the whole context sentence and do "
          "not copy long phrases from it. Never add a fact that is not in the context. If the context does not answer the question, "
          "reply exactly: i don't know that one.\n"
          "Examples:\nC: Kenya is a country in East Africa, about halfway down the continent\nQ: where is kenya\nA: it's in east africa.\n"
          "C: George Washington was the first president of the United States from 1789 to\nQ: who was george washington\n"
          "A: the first president of the united states.\n"
          "C: Penguins are seabirds in the family Spheniscidae.\nQ: what is a penguin\nA: a kind of seabird.\n"
          "The user message has numbered items, each with C: and Q: lines. Reply with one line per item, exactly `<number>: <answer>`, "
          "nothing else.")


def words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def stem(w):
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def longest_run(a, b):
    """Longest common run of consecutive words between two texts."""
    a, b, best = words(a), words(b), 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else 0)
            best = max(best, cur[-1])
        prev = cur
    return best


def grounded(answer, question, ctx):
    known = {stem(x) for x in words(ctx) + words(question)}
    return all(x in FSTOP or stem(x) in known for x in words(answer))


def valid(answer, question, ctx):
    """<=12 words; every content word (non-stopword, lowercased, crude plural strip) is in the context or question; no run
    of more than 6 words shared with the context. The idk answer always passes."""
    if answer == IDK:
        return True
    w = words(answer)
    if not w or len(w) > 12 or "\n" in answer:
        return False
    if not grounded(answer, question, ctx):
        return False
    return longest_run(answer, ctx) <= 6


CANDS = None  # --cands: jsonl of {prompt, context} still missing an answer (build_chatds.py --dump-missing)
BASE = None   # --base: earlier fact-answers.jsonl whose answers are kept for contexts that did not change


def candidates():
    seen, out = set(), []
    for l in open(CANDS or SRC):
        r = json.loads(l)
        k = (r["prompt"], r.get("context"))
        if r.get("context") and r.get("answer") != IDK and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def gen(cap, limit):
    import submit_deepseek as sd
    key = sd.require_key()
    cands = candidates()[:limit or None]
    done = {}
    if os.path.exists(RAW):
        for l in open(RAW):
            r = json.loads(l)
            done[r["id"]] = r
    usage = sd.Usage()
    for r in done.values():
        usage.add(r["usage"])
    seed_cost = usage.cost
    batches = [(i, cands[i:i + BATCH]) for i in range(0, len(cands), BATCH)]
    todo = [b for b in batches if "b%d" % b[0] not in done]
    print("%d candidates, %d requests, %d to do, spent so far $%.4f (cap $%.2f)" % (len(cands), len(batches), len(todo), seed_cost, cap))
    lock, stop, out_f = threading.Lock(), threading.Event(), open(RAW, "a")

    def one(b):
        i, items = b
        if stop.is_set():
            return
        user = "\n".join("%d.\nC: %s\nQ: %s" % (n + 1, c, q) for n, (q, c) in enumerate(items))
        payload = {"model": sd.DEEPSEEK_MODEL, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                   "max_tokens": 60 * BATCH, "thinking": {"type": "disabled"}}
        for attempt in range(5):
            try:
                resp = sd._post_once(key, payload)
            except Exception:
                resp = None
            if resp is not None and resp.status_code == 200:
                body = resp.json()
                msg, u = body["choices"][0].get("message", {}), body.get("usage", {})
                with lock:
                    usage.add(u)
                    if sd.detects_reasoning(msg, u):
                        print("reasoning detected, aborting", file=sys.stderr); stop.set(); return
                    out_f.write(json.dumps({"id": "b%d" % i, "items": items, "text": msg.get("content") or "", "usage": u}) + "\n")
                    out_f.flush()
                    if usage.n % 100 == 0:
                        print(usage.line(), flush=True)
                    if usage.cost >= cap:
                        print("cap reached", file=sys.stderr); stop.set()
                return
            if resp is not None and resp.status_code not in (429,) and resp.status_code < 500:
                print("HTTP", resp.status_code, file=sys.stderr); return
            time.sleep(2 ** attempt)

    t0 = time.time()
    try:
        with cf.ThreadPoolExecutor(6) as ex:
            list(ex.map(one, todo))
    finally:
        out_f.close()
        spent = usage.cost - seed_cost
        print(usage.line(), "this run $%.4f, %.0fs" % (spent, time.time() - t0))
        st = sd.load_batches()
        st["spent_estimate_usd"] = round(st["spent_estimate_usd"] + spent, 4)
        st["batches"].append({"id": "deepseek-facts-%s" % time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), "provider": "deepseek",
                              "shard": "fact-answers-raw.jsonl", "n_requests": usage.n - len(done), "estimated_cost_usd": round(spent, 4),
                              "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "ended", "fetched": True})
        sd.save_batches(st)
    parse()


def parse():
    n = ok = idk = 0
    reasons = {"len": 0, "unknown_word": 0, "copy": 0, "format": 0}
    res = {}
    if BASE:
        for l in open(BASE):
            r = json.loads(l)
            res[(r["prompt"], r["context"])] = r["answer"]
    for l in open(RAW):
        r = json.loads(l)
        got = {}
        for line in r["text"].splitlines():
            m = re.match(r"^\s*(\d+)\s*[:.)]\s*(.+?)\s*$", line)
            if m:
                got[int(m[1])] = m[2].strip().strip("`").lower()
        for k, (q, c) in enumerate(r["items"]):
            n += 1
            a = got.get(k + 1)
            if a is None:
                reasons["format"] += 1
            elif valid(a, q, c):
                ok += 1; idk += a == IDK
                res[(q, c)] = a
            elif len(words(a)) > 12:
                reasons["len"] += 1
            elif longest_run(a, c) > 6:
                reasons["copy"] += 1
            else:
                reasons["unknown_word"] += 1
    with open(OUT, "w") as f:
        for (q, c), a in res.items():
            f.write(json.dumps({"prompt": q, "context": c, "answer": a}) + "\n")
    print("answers %d, valid %d (%d idk), dropped %s -> %s" % (n, ok, idk, reasons, OUT))


def selftest():
    assert valid("it's in east africa.", "where is kenya", "Kenya is a country in East Africa, about halfway down the continent")
    assert not valid("it's in south america.", "where is kenya", "Kenya is a country in East Africa")
    assert not valid("kenya is a country in east africa about halfway", "where is kenya", "Kenya is a country in East Africa about halfway down")
    assert not valid("one two three four five six seven eight nine ten eleven twelve thirteen", "q", "one two three four five six seven eight nine ten eleven twelve thirteen")
    assert valid(IDK, "q", "c") and longest_run("a b c d", "x b c y") == 2
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="store_true"); ap.add_argument("--parse", action="store_true"); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cap", type=float, default=5.0); ap.add_argument("--limit", type=int)
    ap.add_argument("--cands"); ap.add_argument("--base"); ap.add_argument("--raw"); ap.add_argument("--out")
    a = ap.parse_args()
    CANDS, BASE, RAW, OUT = a.cands, a.base, a.raw or RAW, a.out or OUT
    if a.selftest: selftest()
    elif a.parse: parse()
    elif a.gen: gen(a.cap, a.limit)
