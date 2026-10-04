"""Export prompt/calculator observations from the Python reference.
P<TAB>query<TAB>expected formatted prompt   (retrieve + format_prompt)
C<TAB>model output<TAB>expected display     (apply_calc; unchanged text when it is not a calc)
Tabs/newlines/backslashes are escaped as \\t \\n \\r \\\\.  Usage: gen_chatds_golden.py [--kb KB] [--out FILE] [-n N]"""
import argparse, os, random, struct, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chatds_runtime as rt  # noqa: E402

CUES = ["who was {t}", "who is {t}", "what is {t}", "what is a {t}", "what is the {t}", "whats {t}", "what's {t}",
        "where is {t}", "where is the {t}", "when was {t}", "tell me about {t}", "define {t}", "explain {t}",
        "describe {t}", "which {t}", "who was {t} and what did they do", "who was {t} one of the two", "what is the capital of {t}",
        "tell me about {t} please", "Who was {t}?", "WHAT IS {t}??", "  what   is  {t} ", "what is {t}, exactly?"]
NOCUE = ["how do i list all files", "how do i print hello in python", "make a folder called pics", "count lines in notes.txt",
         "sort scores biggest first", "reverse a string called word", "what folder am i in", "find the word error in log.txt",
         "show last 20 lines of server.log", "what does x = 5 mean", "what is 'abc'", "what is a dict", "what's the code for hello",
         "who wrote my_function", "what is the list of files", "define a function", "explain this loop", "what is `ls`",
         "whats 38 times 47", "what is 12 + 7", "144 divided by 12", "2.5 plus 3.75", "whats 15 percent of 80", "what is 905 minus 377",
         "1234 + 5678", "what is 3-1", "who is 5 * 2", "what is 10%3", "what is 2^8", "where is 7 / 2", "when is 1999 - 1066",
         "what is five times six", "what is half of eighty", "whats the sum of twelve and forty-two", "what is forty two over seven",
         "who is twice three", "what is eight and seventy four", "whats a quarter of one hundred", "what is five more than nine", "whats twenty take away nine", "what is double twenty one", "what is the difference of ten and three", "what is a-b", "what is the year 1999", "who won in 2020", "what's 5 +3", "what is the 5th planet"]
CHAT = ["hiii", "yo how r u", "wat are you", "what can u do", "ok thx", "tell me a joke", "good morning", "lol", "hello there",
        "tell me a story", "how are you", "you are funny", "bye", "can you help me", "what's up", "who are you", "what are you doing",
        "tell me something", "thanks!", "i'm bored", "what's your name", "do you like games"]
SYL = "zor va len mar kel dro qui xa fen bru tal ny sko pel dor wim gha".split()
STOPS = sorted(rt.STOP) + ["python", "list", "file", "who", "what", "where", "when", "which", "define"]


def keys(kbp, rng, n):
    out = []
    with open(kbp, "rb") as f:
        _, _, nb, nk, nrec, bt, ko, to, size = struct.unpack("<4sIIIIIIII", f.read(36))
        while len(out) < n:
            b = rng.randrange(nb)
            f.seek(bt + 4 * b); a, e = struct.unpack("<2I", f.read(8))
            if e == a:
                continue
            f.seek(ko + a); blk = f.read(e - a)
            ks, i = [], 0
            while i < len(blk):
                kl = blk[i + 4]; ks.append(blk[i + 5:i + 5 + kl].decode()); i += 9 + kl
            out.append(rng.choice(ks))
    return out


def typo(w, rng, edits=1):
    for _ in range(edits):
        i = rng.randrange(len(w))
        k = rng.randrange(4)
        c = rng.choice("abcdefghijklmnopqrstuvwxyz")
        if k == 0 and len(w) > 1: w = w[:i] + w[i + 1:]
        elif k == 1: w = w[:i] + c + w[i:]
        elif k == 2: w = w[:i] + c + w[i + 1:]
        elif i + 1 < len(w): w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
    return w


def typo_title(t, rng, edits=1):
    ws = t.split()
    cand = [i for i, w in enumerate(ws) if len(w) >= 5] or list(range(len(ws)))
    i = rng.choice(cand)
    ws[i] = typo(ws[i], rng, edits)
    return " ".join(ws)


def fake(rng):
    return "".join(rng.choice(SYL) for _ in range(rng.randrange(2, 4)))


def queries(kbp, n, rng):
    ks = keys(kbp, rng, n)
    out = []  # (kind, query)
    for i in range(n):
        r, t = rng.random(), ks[i]
        tpl = rng.choice(CUES)
        if r < .28: out.append(("fact", tpl.format(t=t)))
        elif r < .48: out.append(("typo", tpl.format(t=typo_title(t, rng, 1 if rng.random() < .8 else 2))))
        elif r < .58: out.append(("fake", tpl.format(t=fake(rng) + (" " + fake(rng) if rng.random() < .5 else ""))))
        elif r < .64: out.append(("mixed", rng.choice(["who is {f} the {t} painter", "what is {f} {t}", "who is the {t} {f}", "tell me about {f} of {t}"]).format(f=fake(rng), t=t)))
        elif r < .72: out.append(("nocue", rng.choice(NOCUE)))
        elif r < .77: out.append(("nocue", "%s %d %s %d" % (rng.choice(["what is", "whats", "who is", ""]), rng.randrange(100), rng.choice("+-*/%"), rng.randrange(100))))
        elif r < .84: out.append(("chat", rng.choice(CHAT)))
        elif r < .92:
            ws = [rng.choice(STOPS + t.split() + [fake(rng)]) for _ in range(rng.randrange(1, 30))]
            out.append(("soup", " ".join(ws)))
        else: out.append(("fact", tpl.format(t=" ".join(ks[i:i + 1] + [rng.choice(ks)]))))  # two titles glued: long grams / try cap
    return [(k, q) for k, q in out if q.strip() and len(q.encode()) <= 127]


def gen_calcs(rng):
    def num():
        r = rng.random()
        if r < .6: return str(rng.randrange(1000))
        if r < .8: return "%d.%d" % (rng.randrange(100), rng.randrange(10 ** rng.randrange(1, 7)))
        if r < .9: return rng.choice([".5", "5.", "007", "0.00005", "0.000049", "99999999", "0.99995", "1.00005"])
        return str(rng.randrange(10 ** rng.randrange(8, 20)))

    def ex(d):
        r = rng.random()
        if d > 3 or r < .3: return num()
        if r < .4: return "%s%s" % (rng.choice("+-"), ex(d + 1))
        if r < .55: return "(%s)" % ex(d + 1)
        sp = rng.choice(["", " ", "  "])
        return ex(d + 1) + sp + rng.choice(["+", "-", "*", "/", "%", "+", "-", "*", "/", "**", "//"]) + sp + ex(d + 1)
    fixed = ["calc(1/0)", "calc(5%0)", "calc(1+)", "calc()", "calc( )", "calc(2**3)", "calc(2//3)", "calc(1.2.3)", "calc(5 5)", "calc(.)",
             "calc(9)) or 1", "calc(2)(3)", "calc(__import__('os'))", "calc(1+2) extra", "the answer is calc(1+1)", "hello", "", "calc", "CALC(1+1)",
             "calc(" + "(" * 40 + "1" + ")" * 40 + ")", "calc(" + "(" * 32 + "1" + ")" * 32 + ")", "calc(" + "(" * 8 + "1" + ")" * 8 + ")", "calc(" + "(" * 9 + "1" + ")" * 9 + ")", "calc(" + "-" * 33 + "1)", "calc(" + "-" * 31 + "1)",
             "calc(1/0+)", "calc(1/0+1)", "calc(99999999999999999999)", "calc(922337203685477.5807)", "calc(922337203685477.5808)",
             "calc(-7%3)", "calc(7%-3)", "calc(0.5/0.0001)", "calc(-0.0001*0.5)", "calc(0.0001*0.5)", "calc(1/3*3)", "calc(10/4)", "calc(-10/4)"]
    out = list(fixed)
    for _ in range(700):
        e = ex(0)
        pad = rng.choice(["", " ", "\n", "  \t", " \n"])
        out.append(pad + "calc(" + rng.choice(["", " "]) + e + rng.choice(["", " "]) + ")" + pad)
    return [o for o in out if len(o) < 200]


def esc(s):
    return s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default=os.path.join(HERE, "kb", "out", "simplewiki.kb"))
    ap.add_argument("--out", required=True, help="output TSV path; never overwrite a frozen device fixture")
    ap.add_argument("-n", type=int, default=2600)
    a = ap.parse_args()
    rng = random.Random(20260929)
    qs = queries(a.kb, a.n, rng)
    calcs = gen_calcs(rng)
    lines = []
    by = {}
    for kind, q in qs:
        ctx = rt.retrieve(q, a.kb)
        by.setdefault(kind, [0, 0]); by[kind][0] += 1; by[kind][1] += bool(ctx)
        lines.append("P\t%s\t%s" % (esc(q), esc(rt.format_prompt(q, ctx))))
    for c in calcs:
        lines.append("C\t%s\t%s" % (esc(c), esc(rt.apply_calc(c))))
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    ncalc = sum(1 for c in calcs if rt.apply_calc(c) != c)
    print("golden: %d prompts (%d with context) %s; %d calc outputs (%d evaluate, %d error) -> %s" % (
        len(qs), sum(v[1] for v in by.values()), {k: tuple(v) for k, v in sorted(by.items())}, len(calcs), ncalc,
        sum(1 for c in calcs if rt.apply_calc(c).endswith("= error")), a.out))


if __name__ == "__main__":
    main()
