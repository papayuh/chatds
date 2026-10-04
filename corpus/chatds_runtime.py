"""Host mirror of the ChatDS runtime (port to C for the DS): KB retrieval before
generation, the C:/Q:/A: prompt, and calc() evaluation after.
The independent C plumbing is checked by clean/plumbing/test_parity.py. No floats anywhere."""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "kb"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import kb  # noqa: E402
from prompt_format import format_prompt  # noqa: E402,F401  ("C: ctx\nQ: q\nA:" or "Q: q\nA:")

STOP = set("""what who where when why how is are was were the a an of in on to do does can i you me my your
tell about whats which whom this that it its and or for with at by from be as if so please explain define
describe give say know mean means there their they he she we us our
continent currency language capital country city money name main official primary spoken speak located called
use used people plus minus times divided multiply add subtract percent calculate much""".split())  # + words that hit junk articles ("Continent", "Times")
MAX_TRIES = 40
# Gate: prefill costs ~0.35 s per context token on the DS, so only fact-shaped questions get a lookup.
CUES = ("who", "what", "whats", "where", "when", "which", "define", "explain", "describe")  # + "tell me about"
CODE_WORDS = set("list dict string function variable loop file command python code print return sort reverse count plus minus times divided multiplied percent".split())
CODE_CHARS = set("()[]{}=\"`_")
# Spelled-out arithmetic ("what is five times six", "what is half of eighty") is the model's job (words -> calc(digits)),
# so a number word next to a math word never gets a lookup.
NUM_WORDS = set("""zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen
seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million""".split())
MATH_WORDS = set("""half double twice triple quarter over sum product added add subtract subtracted multiply divide difference
squared cubed and away take more less than""".split())


def wants_context(question):
    q = kb.normalize(question)
    if not (q.split()[:1] and q.split()[0] in CUES or q.startswith("tell me about ")):
        return False
    if any(c in CODE_CHARS for c in question) or CODE_WORDS & set(q.split()):
        return False
    if NUM_WORDS & set(q.split()) and MATH_WORDS & set(q.split()):
        return False
    if re.search(r"(?<![A-Za-z])'|'(?![A-Za-z])", question):  # quote, but not a contraction (what's)
        return False
    return re.search(r"\d\s*[-+*/%^]|[-+*/%^]\s*\d", question) is None


FUZZY_MIN = 5       # every word of a fuzzy n-gram needs >= 5 chars; at most one word is corrected
MAX_FUZZY_WORDS = 8  # distinct query words sent to the delete index per question
stats = {"queries": 0, "lookups": 0, "fuzzy_probes": 0}  # cost counters for reporting: exact/candidate lookups, delete-variant probes


def retrieve(question, kb_path):
    """Word n-grams, longest first (4..1), left to right. Per n: every exact lookup, then fuzzy ones
    (one word swapped for a delete-index candidate). A hit is accepted only if its n-gram holds at
    least half of the question's content (non-stop) words. Returns one sentence, or None."""
    if not wants_context(question):
        return None
    stats["queries"] += 1
    words = kb.normalize(question).split()
    content = sum(w not in STOP for w in words)
    tries = ftries = 0
    cands = {}

    def fuzzy(w):
        if w not in cands:
            cands[w] = kb.fuzzy_words(kb_path, w) if len(cands) < MAX_FUZZY_WORDS else []
            stats["fuzzy_probes"] += len(w) + 1
        return cands[w]

    for n in range(4, 0, -1):
        grams = [words[i:i + n] for i in range(len(words) - n + 1)]
        # covers a strict majority of the content words (also skips all-stopword grams and 1-grams under 3 chars)
        grams = [g for g in grams if 2 * sum(w not in STOP for w in g) > content
                 and any(w not in STOP for w in g) and not (n == 1 and len(g[0]) < 3)]
        for g in grams:
            if tries >= MAX_TRIES:
                return None
            tries += 1
            stats["lookups"] += 1
            hit = kb.lookup(kb_path, " ".join(g))
            if hit:
                return hit
        for g in grams:
            if not all(len(w) >= FUZZY_MIN for w in g):
                continue
            for j in range(n):
                for c in fuzzy(g[j]):
                    if ftries >= MAX_TRIES:
                        return None
                    ftries += 1
                    stats["lookups"] += 1
                    hit = kb.lookup(kb_path, " ".join(g[:j] + [c] + g[j + 1:]))
                    if hit:
                        return hit
    return None


# ---- calc: integer fixed point, scale 10^4, int64. C counterpart: clean/plumbing/calc.c.
# Grammar: expr = term {(+|-) term}; term = unary {(*|/|%) unary}; unary = (+|-) unary | number | ( expr ).
# number = digits [. digits] | . digits (leading zeros ok); digits past the 4th decimal round half up on the 5th.
# Rounding: * and / round half away from zero at 4 decimals. % keeps the dividend's sign (C semantics).
# Every value and the raw product / scaled dividend must satisfy |x| <= 2^63-1; overflow and /0 -> "error".
# A syntax error anywhere (or nesting deeper than CALC_DEPTH) leaves the output unchanged.
S, MAXV, CALC_DEPTH = 10000, 2**63 - 1, 8


def _rdiv(n, d):
    q, r = divmod(abs(n), abs(d))
    q += 2 * r >= abs(d)
    return -q if (n < 0) != (d < 0) else q


class _Calc:
    def __init__(self, e):
        self.e, self.i, self.bad, self.ovf = e, 0, False, False

    def peek(self):
        while self.i < len(self.e) and self.e[self.i] == " ":
            self.i += 1
        return self.e[self.i] if self.i < len(self.e) else ""

    def ok(self, v):
        if abs(v) > MAXV:
            self.ovf = True
            return 0
        return v

    def expr(self, depth):
        if depth > CALC_DEPTH:
            self.bad = True
            return 0
        a = self.term(depth)
        while not self.bad and self.peek() in ("+", "-"):
            op = self.e[self.i]; self.i += 1
            b = self.term(depth)
            a = self.ok(a + b if op == "+" else a - b)
        return a

    def term(self, depth):
        a = self.unary(depth)
        while not self.bad and self.peek() in ("*", "/", "%"):
            op = self.e[self.i]; self.i += 1
            b = self.unary(depth)
            if self.ovf:
                continue
            if op == "*":
                a = _rdiv(self.ok(a * b), S)
            elif b == 0:
                self.ovf = True
            elif op == "/":
                a = _rdiv(self.ok(a * S), b)
            else:
                a = (1 if a >= 0 else -1) * (abs(a) % abs(b))
        return a

    def unary(self, depth):
        c = self.peek()
        if depth > CALC_DEPTH:
            self.bad = True
            return 0
        if c in ("+", "-"):
            self.i += 1
            v = self.unary(depth + 1)
            return -v if c == "-" else v
        if c == "(":
            self.i += 1
            v = self.expr(depth + 1)
            if self.peek() != ")":
                self.bad = True
                return 0
            self.i += 1
            return v
        m = re.compile(r"([0-9]*)(?:\.([0-9]*))?").match(self.e, self.i)
        ip, fp = m.group(1), m.group(2) or ""
        if not ip and not fp:
            self.bad = True
            return 0
        self.i = m.end()
        f = int((fp + "0000")[:4]) + (len(fp) > 4 and fp[4] >= "5")
        return self.ok(int(ip or "0") * S + f)


def _fmt(v):
    if v % S == 0:
        return str(v // S)
    s = "%s%d.%04d" % ("-" if v < 0 else "", abs(v) // S, abs(v) % S)
    return s.rstrip("0")


def apply_calc(output):
    """`calc(38*47)` -> `calc(38*47) = 1786`; anything else is returned unchanged."""
    m = re.match(r"^calc\(([0-9 +\-*/().%]+)\)$", output.strip(" \t\n\r\v\f"))
    if not m:
        return output
    expr = m.group(1).strip(" ")
    c = _Calc(expr)
    v = c.expr(0)
    if c.bad or c.peek() != "":
        return output
    return "calc(%s) = %s" % (expr, "error" if c.ovf else _fmt(v))
