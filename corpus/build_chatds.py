#!/usr/bin/env python3
"""Build corpus/results/chatds-train.jsonl: the P1 mixed training pairs.

    python3 corpus/curriculum_v2.py --out-dir train/data/chatds-curriculum \\
        --extra-eval eval/chatds-acceptance-30.jsonl --extra-eval eval/auto-dev-40.jsonl \\
        --extra-eval eval/holdout-blind-40.jsonl
    python3 corpus/build_chatds.py --tokenizer path/to/your-tokenizer.model

Rows: {prompt, answer, category, source[, context]}. `context` is the KB text the
runtime would retrieve for the prompt; export_train puts it on a "C: " line.
"""
import argparse, collections, json, os, random, re, sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "reference", "ds-llm"))
import chatds_runtime as rt  # noqa: E402
import export_train, postprocess as pp  # noqa: E402
kb = rt.kb

ROOT = os.path.join(HERE, "..")
EVALS = ["chatds-acceptance-30", "auto-dev-40", "holdout-blind-40", "acceptance-draft-20", "kb-probe-150", "numwords-40"]
IDK = "i don't know that one."
KEEP = ("synth-identity", "synth-joke", "idk-nohit", "synth-idk")  # rows that are never sampled down
ME = "i'm chatds, a tiny model running on a nintendo ds lite."
MIX = {"python+bugfix": .40, "chat": .12, "factual": .15, "calc": .12, "shell": .11, "transform": .10}
TOTAL = 400_000

WRAP = re.compile(r"^(?:Q: )?(?:(?:I was wondering|I'm curious),? )?(?:(?:Answer briefly|One more question|Real quick|"
                  r"Short answer please|Simple question|Just curious|Mind answering this|Tell me|Quick question|"
                  r"Here's a question|Please answer|Fact check|Answer this|For my homework|Here's a quick one|"
                  r"Trivia question|For the record|Give me a one-word answer|Just the answer|Help me out here|"
                  r"Briefly|No explanation needed|Can you tell me|I need to know|In one word|Pop quiz): )?")
NUM = r"(-?\d+(?:\.\d+)?)"
OPS = {"plus": "+", "+": "+", "minus": "-", "-": "-", "times": "*", "*": "*", "x": "*", "divided by": "/", "/": "/"}
OPRE = "|".join(re.escape(o) for o in sorted(OPS, key=len, reverse=True))


def paren(n):
    return "(%s)" % n if n.startswith("-") else n


def extract_expr(q):
    """Arithmetic question -> expression string, or None when unsure."""
    q = q.strip().rstrip("?.! ").lower()
    m = re.fullmatch(r"(?:what is |what's |whats |how much is )?%s (%s) %s" % (NUM, OPRE, NUM), q)
    if m:
        return paren(m[1]) + OPS[m[2]] + paren(m[3])
    m = re.fullmatch(r"(add|multiply|divide|subtract) %s (and|by|from) %s" % (NUM, NUM), q)
    if m:
        a, b, v = m[2], m[4], m[1]
        if (v, m[3]) == ("add", "and"): return paren(a) + "+" + paren(b)
        if (v, m[3]) == ("multiply", "by"): return paren(a) + "*" + paren(b)
        if (v, m[3]) == ("divide", "by"): return paren(a) + "/" + paren(b)
        if (v, m[3]) == ("subtract", "from"): return paren(b) + "-" + paren(a)
    return None


def last_num(s):
    f = re.findall(r"-?\d+(?:\.\d+)?", re.sub(r"(?<=\d),(?=\d{3})", "", s))  # "45,408" -> 45408
    return Fraction(f[-1]) if f else None


def calc_answer(expr):
    out = rt.apply_calc("calc(%s)" % expr)
    return out.split(" = ")[-1] if " = " in out else None


def rand_num(rng):
    r = rng.random()
    if r < .35: return str(rng.randint(2, 99))
    if r < .65: return str(rng.randint(100, 999))
    if r < .85: return str(rng.randint(1000, 9999))
    if r < .95: return "%.*f" % (rng.choice((1, 2)), rng.uniform(1, 200))
    return str(rng.randint(10000, 99999))


WORD_T = ["whats {a} {w} {b}", "what is {a} {w} {b}", "how much is {a} {w} {b}", "{a} {w} {b}", "calculate {a} {w} {b}",
          "hey what's {a} {w} {b}", "can you do {a} {w} {b}", "what's {a} {w} {b}?"]
SYM_T = ["{a} {s} {b}", "{a}{s}{b}", "{a} {s} {b}?", "whats {a} {s} {b}", "what is {a} {s} {b}", "{a} {s} {b} ="]
WORDS = {"+": ("plus",), "-": ("minus",), "*": ("times", "multiplied by"), "/": ("divided by", "over")}
SYMS = {"+": ("+",), "-": ("-",), "*": ("*", "x"), "/": ("/",)}


def synth_calc(rng, n):
    rows = []
    while len(rows) < n:
        op, a, b, r = rng.choice("+-*/"), rand_num(rng), rand_num(rng), rng.random()
        if r < .04:  # percent
            a = str(rng.choice((5, 10, 12, 15, 20, 25, 30, 35, 40, 50, 75)))
            q = rng.choice(("whats {a} percent of {b}", "{a}% of {b}", "what is {a} percent of {b}")).format(a=a, b=b)
            expr = "%s*%g" % (b, int(a) / 100)
        elif r < .09:  # three terms, precedence
            c, op2 = rand_num(rng), rng.choice("+-*")
            q = "%s %s %s %s %s" % (a, op, b, op2, c)
            expr = q.replace(" ", "")
        else:
            if op == "/" and rng.random() < .7:  # mostly clean quotients
                b = str(rng.randint(2, 99)); a = str(int(b) * rng.randint(2, 99))
            if op == "-" and rng.random() < .8 and float(a) < float(b):
                a, b = b, a
            if rng.random() < .5:
                q = rng.choice(WORD_T).format(a=a, b=b, w=rng.choice(WORDS[op]))
            else:
                q = rng.choice(SYM_T).format(a=a, b=b, s=rng.choice(SYMS[op]))
            expr = a + op + b
        ans = calc_answer(expr)
        if ans and ans != "error":
            rows.append({"prompt": q, "answer": "calc(%s)" % expr, "category": "calc", "source": "synth-calc"})
    return rows


ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def spell(n, rng):
    """0..999999 -> English words: 'forty two' (or 'forty-two'), 'three hundred and five', 'one thousand two hundred'."""
    if n < 20: return ONES[n]
    if n < 100:
        return TENS[n // 10] + ((rng.choice((" ", " ", "-")) + ONES[n % 10]) if n % 10 else "")
    if n < 1000:
        w = ("a" if n < 200 and rng.random() < .15 else ONES[n // 100]) + " hundred"
        return w + ((rng.choice((" and ", " ", " and ")) + spell(n % 100, rng)) if n % 100 else "")
    w = ("a" if n < 2000 and rng.random() < .1 else spell(n // 1000, rng)) + " thousand"
    return w + ((rng.choice((" ", " and ")) + spell(n % 1000, rng)) if n % 1000 else "")


def word_num(rng):
    r = rng.random()
    if r < .30: return rng.randint(0, 19)
    if r < .50: return rng.choice((20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000))
    if r < .75: return rng.randint(20, 99)
    if r < .93: return rng.randint(100, 999)
    return rng.randint(1000, 12000)


NW_OPS = {"+": ("plus", "plus", "and", "added to"), "-": ("minus", "minus", "take away"), "*": ("times", "times", "multiplied by"),
          "/": ("divided by", "divided by", "over")}
NW_T = ["what is {a} {w} {b}", "whats {a} {w} {b}", "{a} {w} {b}", "how much is {a} {w} {b}", "calculate {a} {w} {b}",
        "what's {a} {w} {b}?", "hey whats {a} {w} {b}", "can you do {a} {w} {b}", "{a} {w} {b} =", "tell me {a} {w} {b}", "quick, {a} {w} {b}"]
NW_V = [("add {a} and {b}", "+"), ("multiply {a} by {b}", "*"), ("divide {a} by {b}", "/"), ("subtract {a} from {b}", "-b"),
        ("what is the sum of {a} and {b}", "+"), ("what is the product of {a} and {b}", "*"), ("whats {a} more than {b}", "+b"),
        ("what do you get if you add {a} and {b}", "+"), ("take {a} from {b}", "-b"), ("what is {a} take away {b}", "-")]


def synth_numwords(rng, n):
    """Spelled-out (and mixed digit/word) arithmetic -> calc(digits). The model does the words -> digits conversion."""
    rows = []
    while len(rows) < n:
        def side(v):  # mostly words, sometimes digits ("7 times twelve")
            return str(v) if rng.random() < .2 else spell(v, rng)
        r = rng.random()
        if r < .07:  # half of / double / twice / a quarter of
            v = word_num(rng) * 2 if rng.random() < .5 else word_num(rng)
            t, e = rng.choice((("half of {a}", "%d/2"), ("half of {a}", "%d/2"), ("a half of {a}", "%d/2"), ("double {a}", "%d*2"),
                               ("twice {a}", "%d*2"), ("triple {a}", "%d*3"), ("a quarter of {a}", "%d/4")))
            t = rng.choice(("what is {t}", "whats {t}", "{t}", "what's {t}?", "how much is {t}")).format(t=t.format(a=side(v)))
            expr = e % v
        elif r < .12:  # x percent of y
            p, v = rng.choice((5, 10, 15, 20, 25, 30, 40, 50, 75)), word_num(rng)
            t = rng.choice(("whats {p} percent of {v}", "what is {p} percent of {v}", "{p} percent of {v}")).format(
                p=side(p), v=side(v))
            expr = "%d*%g" % (v, p / 100)
        elif r < .22:  # decimals: two point five plus three
            a, d = word_num(rng) % 100, rng.randint(1, 9)
            t = "%s point %s %s %s" % (spell(a, rng), ONES[d], "plus", side(word_num(rng) % 100)) if rng.random() < .5 else \
                "%s %s %s point %s" % (side(word_num(rng) % 100), "times", spell(a, rng), ONES[d])
            if " plus " in t:
                expr = "%d.%d+%s" % (a, d, t.split(" plus ")[1] if t.split(" plus ")[1].isdigit() else None)
                if expr.endswith("None"): continue
            else:
                expr = "%s*%d.%d" % (t.split(" times ")[0], a, d)
                if not t.split(" times ")[0].isdigit(): continue
            t = rng.choice(("what is {t}", "whats {t}", "{t}")).format(t=t)
        elif r < .35:  # verb forms
            a, b = word_num(rng), word_num(rng)
            t, o = rng.choice(NW_V)
            q = t.format(a=side(a), b=side(b)) if o[0] != "-" or len(o) == 1 else t.format(a=side(a), b=side(b))
            t = q
            expr = {"+": "%d+%d" % (a, b), "*": "%d*%d" % (a, b), "/": "%d/%d" % (a, b), "-b": "%d-%d" % (b, a),
                    "+b": "%d+%d" % (a, b), "-": "%d-%d" % (a, b)}[o]
            if rng.random() < .5:
                t = rng.choice(("", "", "hey ", "pls ", "ok ")) + t
        else:
            op = rng.choice("+-*/")
            a, b = word_num(rng), word_num(rng)
            if op == "*" and rng.random() < .7: a, b = rng.randint(2, 20), rng.randint(2, 99)
            if op == "/" and rng.random() < .7:
                b = rng.randint(2, 20); a = b * rng.randint(2, 60)
            if op == "-" and rng.random() < .8 and a < b: a, b = b, a
            w = rng.choice(NW_OPS[op])
            sa, sb = side(a), side(b)
            if sa.isdigit() and sb.isdigit():
                sa = spell(a, rng)
            t = rng.choice(NW_T).format(a=sa, b=sb, w=w)
            expr = "%d%s%d" % (a, op, b)
        if rng.random() < .1: t = t.capitalize()
        if not any(x in t.lower().replace("-", " ").split() for x in ONES + [w for w in TENS if w != "_"] + ["hundred", "thousand"]):
            continue
        ans = calc_answer(expr)
        if ans and ans != "error":
            rows.append({"prompt": t, "answer": "calc(%s)" % expr, "category": "calc", "source": "synth-numwords"})
    return rows


def identity_rows(rng):
    qa = {
        ("who are you", "wat are you", "what are you", "who r u", "who is this", "what's your name", "whats ur name",
         "what should i call you", "are you a person", "are you real", "introduce yourself"): [
            ME, "i'm chatds. i'm tiny and i live on a nintendo ds lite.", "chatds. a small model on a nintendo ds lite."],
        ("who made you", "who built you", "who created you", "who is your creator", "who programmed you"): [
            "diego made me. i run on a nintendo ds lite.", "diego made me. i'm a tiny model on a nintendo ds lite."],
        ("what can you do", "what can u do", "what do you do", "what are you good at", "how can you help", "what can you help with"): [
            "i can chat, do math, look up simple facts, and write short shell and python one-liners.",
            "math, simple facts, shell and python one-liners, and small talk. that's about it."],
        ("are you chatgpt", "are you claude", "are you gpt", "are you an ai", "are you a bot", "are you siri"): [
            "i'm chatds, a tiny model running on a nintendo ds lite. not chatgpt.", "i'm an ai, a tiny one. chatds, on a nintendo ds lite."],
        ("where do you live", "what are you running on", "what device are you on", "what hardware are you", "are you on a phone"): [
            "i run on a nintendo ds lite.", "a nintendo ds lite. 4 mb of ram, no internet."],
        ("are you smart", "do you know everything", "are you always right", "can you be wrong", "how big is your brain"): [
            "not really. i'm tiny and i can be wrong.", "no. i'm a tiny model, so check anything important."],
        ("hi", "hello", "hey", "yo", "sup", "hiii", "hey there", "hello there", "good morning"): [
            "hi, i'm chatds. what's up?", "hey. what do you need?", "hello."],
        ("how are you", "how r u", "yo how r u", "how's it going", "whats up"): [
            "doing good. what do you need?", "good. what's up?"],
        ("thanks", "thx", "ok thx", "thank you", "ty", "ok thanks", "cool thanks"): ["no problem.", "anytime.", "you're welcome."],
        ("bye", "goodbye", "see ya", "gtg", "cya"): ["bye.", "see you."],
    }
    rows = []
    for qs, ans in qa.items():
        for q in qs:
            for pre in ("", "", "hey ", "yo ", "ok "):
                for post in ("", "?" if "?" not in q and len(q.split()) > 1 else ""):
                    for a in ans:
                        if rng.random() < .5:
                            rows.append({"prompt": pre + q + post, "answer": a, "category": "chat", "source": "synth-identity"})
    return rows


JOKES = [
    ("why did the computer get cold?", "it left its windows open."),
    ("what do you call a sleeping bull?", "a bulldozer."),
    ("why don't eggs tell jokes?", "they'd crack each other up."),
    ("what do you call cheese that isn't yours?", "nacho cheese."),
    ("why did the scarecrow win an award?", "he was outstanding in his field."),
    ("what did the ocean say to the beach?", "nothing, it just waved."),
    ("why can't a bike stand up by itself?", "it's two tired."),
    ("what do you call a fake noodle?", "an impasta."),
    ("why did the cookie go to the doctor?", "it felt crummy."),
    ("what has hands but can't clap?", "a clock."),
    ("why was the math book sad?", "it had too many problems."),
    ("what do you call a bear with no teeth?", "a gummy bear."),
    ("why did the banana go to the doctor?", "it wasn't peeling well."),
    ("what did one wall say to the other?", "i'll meet you at the corner."),
    ("why do bees have sticky hair?", "they use honeycombs."),
    ("what's a cat's favorite color?", "purr-ple."),
    ("why did the tomato turn red?", "it saw the salad dressing."),
    ("what do you call a dinosaur that crashes his car?", "tyrannosaurus wrecks."),
    ("why did the golfer bring two pairs of pants?", "in case he got a hole in one."),
    ("what do you call a pig that does karate?", "a pork chop."),
    ("why are ghosts bad liars?", "you can see right through them."),
    ("what did the left eye say to the right eye?", "between us, something smells."),
    ("why did the student eat his homework?", "the teacher said it was a piece of cake."),
    ("what do you call a boomerang that won't come back?", "a stick."),
    ("why did the picture go to jail?", "it was framed."),
    ("what kind of tree fits in your hand?", "a palm tree."),
    ("why do cows wear bells?", "their horns don't work."),
    ("what do you call a snowman in summer?", "a puddle."),
    ("why did the robot go on vacation?", "it needed to recharge."),
    ("what's orange and sounds like a parrot?", "a carrot."),
    ("why did the chicken join a band?", "it had the drumsticks."),
    ("what do you call a lazy kangaroo?", "a pouch potato."),
    ("why was the calendar so popular?", "it had lots of dates."),
    ("what did the zero say to the eight?", "nice belt."),
    ("why did the coffee file a police report?", "it got mugged."),
    ("what do you get when you cross a snowman and a vampire?", "frostbite."),
    ("why don't skeletons fight each other?", "they don't have the guts."),
    ("what do you call an elephant that doesn't matter?", "an irrelephant."),
    ("why did the mouse stay home?", "it had a cold and needed to sneeze-mouse."),
    ("what's a computer's favorite snack?", "microchips."),
]
JOKE_Q = ["tell me a joke", "got any jokes", "say something funny", "joke please", "make me laugh", "know any jokes",
          "tell me another joke", "another joke", "tell a joke", "do you know a joke", "give me a joke",
          "i want to hear a joke", "say a joke", "can you tell me a joke", "im bored tell me a joke", "one more joke"]


def joke_rows(rng):
    rows = []
    for q in JOKE_Q:
        for pre in ("", "", "hey ", "ok ", "yo ", "lol "):
            for _ in range(2):
                a, b = rng.choice(JOKES)
                rows.append({"prompt": pre + q, "answer": a + " " + b, "category": "chat", "source": "synth-joke"})
    return rows


CUE_T = ["who is {x}", "who was {x}", "what is {x}", "what are {x}", "whats {x}", "where is {x}", "tell me about {x}",
         "what is a {x}", "who is the {x}", "when was {x} founded", "what is {x} known for", "describe {x}"]
SYL = "ka ke ki ko ku la le li lo lu ma me mi mo mu na ne ni no nu ra re ri ro ru sa se si so su ta te ti to tu va ve vi vo vu za ze zi zo zu th br kr gl".split()
KIND = ["bay", "island", "city", "river", "mountain", "lake", "valley", "kingdom", "clan", "tower"]


def nonsense(rng):
    w = lambda: "".join(rng.choice(SYL) for _ in range(rng.randint(2, 4)))
    r = rng.random()
    if r < .4: return w()
    if r < .7: return w() + " " + rng.choice(KIND)
    if r < .85: return w() + " " + w()
    return w() + " the " + w() + " " + rng.choice(("painter", "chef", "king", "singer", "poet", "inventor"))


def synth_idk_rows(rng, kb_path, n):
    """Made-up names in the fact-cue templates. Only kept if the real runtime finds no context for them."""
    rows, seen = [], set()
    while len(rows) < n:
        q = rng.choice(CUE_T).format(x=nonsense(rng))
        if q in seen or rt.retrieve(q, kb_path):
            continue
        seen.add(q)
        rows.append({"prompt": q, "answer": IDK, "category": "factual", "source": "synth-idk"})
    return rows


_EXT = ("txt", "csv", "log", "md", "py", "sh", "json", "dat", "c", "html", "conf", "png")
SHELL = [  # (question template, command template); {f} {g} files, {d} {e} dirs, {w} word, {n} count, {x} extension, {p} pid
    ("list files", "ls"), ("what files are here", "ls"), ("show files in {d}", "ls {d}"), ("list the contents of {d}", "ls {d}"),
    ("show hidden files", "ls -a"), ("list dotfiles too", "ls -a"), ("show everything in {d} including hidden ones", "ls -a {d}"),
    ("long listing of files", "ls -l"), ("show file sizes and dates", "ls -l"), ("list {d} with details", "ls -l {d}"),
    ("create a directory named {d}", "mkdir {d}"), ("new folder {d}", "mkdir {d}"), ("make dir {d}", "mkdir {d}"),
    ("make nested folders {d}/{e}", "mkdir -p {d}/{e}"), ("create the folder {d} and its parents", "mkdir -p {d}"),
    ("remove empty folder {d}", "rmdir {d}"), ("delete the empty directory {d}", "rmdir {d}"),
    ("delete {f}", "rm {f}"), ("remove the file {f}", "rm {f}"), ("get rid of {f}", "rm {f}"), ("force delete {f}", "rm -f {f}"),
    ("delete folder {d} and everything in it", "rm -r {d}"), ("remove directory {d} recursively", "rm -r {d}"),
    ("copy {f} to {g}", "cp {f} {g}"), ("make a copy of {f} called {g}", "cp {f} {g}"), ("copy folder {d} to {e}", "cp -r {d} {e}"),
    ("rename {f} to {g}", "mv {f} {g}"), ("move {f} into {d}", "mv {f} {d}"), ("move {f} to {d}/", "mv {f} {d}/"),
    ("create an empty file {f}", "touch {f}"), ("make a new file called {f}", "touch {f}"),
    ("show what's in {f}", "cat {f}"), ("print the contents of {f}", "cat {f}"), ("display {f}", "cat {f}"),
    ("first {n} lines of {f}", "head -n {n} {f}"), ("show the top {n} lines of {f}", "head -n {n} {f}"), ("first line of {f}", "head -n 1 {f}"),
    ("last {n} lines of {f}", "tail -n {n} {f}"), ("show the end of {f}", "tail {f}"), ("follow {f} as it grows", "tail -f {f}"),
    ("how many lines in {f}", "wc -l {f}"), ("number of lines in {f}", "wc -l {f}"), ("count words in {f}", "wc -w {f}"),
    ("search {f} for {w}", "grep {w} {f}"), ("lines in {f} containing {w}", "grep {w} {f}"),
    ("find {w} in {f} ignoring case", "grep -i {w} {f}"), ("count matches of {w} in {f}", "grep -c {w} {f}"),
    ("search every file under {d} for {w}", "grep -r {w} {d}"), ("recursively look for {w} in {d}", "grep -r {w} {d}"),
    ("find files named {f}", "find . -name {f}"), ("find all {x} files", "find . -name '*.{x}'"), ("locate {f} under {d}", "find {d} -name {f}"),
    ("print working directory", "pwd"), ("show the current path", "pwd"), ("where am i in the filesystem", "pwd"),
    ("go into {d}", "cd {d}"), ("change directory to {d}", "cd {d}"), ("go up one directory", "cd .."), ("go to my home folder", "cd ~"),
    ("make {f} executable", "chmod +x {f}"), ("give {f} run permission", "chmod +x {f}"),
    ("how big is folder {d}", "du -sh {d}"), ("disk usage of {d}", "du -sh {d}"),
    ("how much disk space is free", "df -h"), ("show disk space", "df -h"),
    ("list running processes", "ps aux"), ("show all processes", "ps aux"),
    ("stop process {p}", "kill {p}"), ("kill pid {p}", "kill {p}"), ("force kill process {p}", "kill -9 {p}"),
    ("print {w}", "echo {w}"), ("say {w} in the terminal", "echo {w}"), ("write {w} into {f}", "echo {w} > {f}"), ("append {w} to {f}", "echo {w} >> {f}"),
    ("sort the lines of {f}", "sort {f}"), ("sort {f} numerically", "sort -n {f}"), ("reverse sort {f}", "sort -r {f}"),
    ("remove duplicate lines from {f}", "sort {f} | uniq"), ("count how many times each line appears in {f}", "sort {f} | uniq -c"),
]
VERBS = set("list show create make remove delete copy rename move go change stop kill print find search sort append write give get display force count".split())
PRE = ["", "", "", "", "how do i ", "command to ", "bash: ", "in linux, ", "shell command to ", "i need to ", "how to "]
POST = ["", "", "", "", " please", " pls", "?"]


def shell_rows(rng, n):
    import curriculum_v2 as cv  # dictionary words, the same source the python curriculum draws identifiers from
    W = cv._DICT_WORDS_TRAIN
    def name():
        r = rng.random()
        w = rng.choice(W) if r < .6 else rng.choice(W) + rng.choice("_-") + rng.choice(W)
        return w
    rows, seen = [], set()
    while len(rows) < n:
        qt, ct = rng.choice(SHELL)
        x = rng.choice(_EXT)
        v = dict(f=name() + "." + rng.choice(_EXT), g=name() + "." + rng.choice(_EXT), d=name(), e=name(), w=rng.choice(W),
                 n=rng.choice((2, 3, 5, 10, 15, 20, 25, 30, 50, 100)), x=x, p=rng.randint(100, 99999))
        q, cmd = qt.format(**v), ct.format(**v)
        pre, post = rng.choice(PRE), rng.choice(POST)
        if pre.startswith(("how", "command", "shell", "i need", "in linux")) and qt.split()[0] not in VERBS:
            pre = ""  # "how do i first 5 lines of x" reads wrong
        q = pre + q + post
        if rng.random() < .15:
            q = q.capitalize()
        if q in seen:
            continue
        seen.add(q)
        rows.append({"prompt": q, "answer": cmd, "category": "shell", "source": "shell-curriculum"})
    return rows


def supports(ctx, teacher):
    """True if the (one-sentence) context contains the teacher's answer; otherwise the model would learn to
    parrot the context whatever the question was."""
    core = kb.normalize(re.sub(r"\s*\(.*?\)", "", teacher))
    return bool(core) and " %s " % core in " %s " % kb.normalize(ctx)


def kb_title_rows(kb_path, rng, n):
    """"what is X"/"who was X" questions from KB keys; the runtime retrieves the context for real and the
    answer is that one sentence. Needs the article text to start with the key, so redirects skip."""
    import struct
    with open(kb_path, "rb") as f:
        hd = f.read(512)
        _, _, _, nk, _, _, ko, to, _ = struct.unpack("<4sIIIIIIII", hd[:36])
        f.seek(ko)
        blob = f.read(to - ko)
    keys, i = [], 0
    for _ in range(nk):
        kl = blob[i + 4]
        keys.append(blob[i + 5:i + 5 + kl].decode()); i += 9 + kl
    keys = [k for k in keys if k.replace(" ", "").isalpha() and len(k) >= 3 and len(k.split()) <= 3]
    rng.shuffle(keys)
    rows = []
    for k in keys:
        raw = kb.lookup(kb_path, k) or ""
        low = raw.lower()
        if " was " in low[:len(k) + 12]: t = ("who was {k}", "tell me about {k}", "whats {k}")
        elif re.search(r"\b(country|city|town|state|river|island|mountain|region|province|lake)\b", low[:len(k) + 60]):
            t = ("where is {k}", "what is {k}", "tell me about {k}", "whats {k}")
        else: t = ("what is {k}", "whats {k}", "tell me about {k}", "what are {k}")
        q = rng.choice(t).format(k=k)
        ctx = rt.retrieve(q, kb_path)  # what the runtime would actually feed the model
        if ctx and kb.normalize(ctx).startswith(k):
            rows.append({"prompt": q, "answer": ctx, "category": "factual", "source": "kb-title", "context": ctx})
            if len(rows) >= n:
                break
    return rows


def load(path):
    return (json.loads(l) for l in open(path) if l.strip())


def contamination_filter():
    """Frozen suite + the three eval files, same checks as postprocess.process()."""
    ps, ep, ea, ng = pp.load_eval_overlap_sets()
    for n in EVALS:
        p2, e2, a2, n2 = pp.load_extra_eval_overlap_sets(os.path.join(ROOT, "eval", n + ".jsonl"))
        ps, ep, ea, ng = ps + p2, ep | e2, ea | a2, ng | n2

    def why(prompt, answer):
        pn, an = pp.normalize_key(prompt), pp.normalize_key(answer)
        if pn in ep: return "exact_prompt"
        if len(an) > 12 and an in ea: return "exact_answer"
        if pp.ngrams(pp.normalize_word_list(prompt)) & ng or pp.ngrams(pp.normalize_word_list(answer)) & ng: return "8gram"
        if pp.max_overlap(pp.normalize_words(prompt), ps) >= pp.OVERLAP_THRESHOLD: return "overlap"
    return why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default=os.path.join(ROOT, "corpus/kb/out/simplewiki.kb"))
    ap.add_argument("--full", default=os.path.join(ROOT, "corpus/results/full-train.jsonl"))
    ap.add_argument("--curriculum", default=os.path.join(ROOT, "train/data/chatds-curriculum/python-curriculum-v2-train.jsonl"))
    ap.add_argument("--tokenizer", required=True, help="SentencePiece .model (see README: tokenizer assets)")
    ap.add_argument("--out", default=os.path.join(ROOT, "corpus/results/chatds-train.jsonl"))
    ap.add_argument("--total", type=int, default=TOTAL)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--fact-answers", default=os.path.join(ROOT, "corpus/results/fact-answers.jsonl"),
                    help="validated short grounded answers (corpus/fact_answers.py); copied fact answers are replaced by these")
    ap.add_argument("--dump-missing", help="write the fact (prompt, context) pairs that have no answer yet to this jsonl (input for fact_answers.py --cands)")
    ap.add_argument("--mix", default="", help="share overrides, e.g. factual=.18,calc=.15,transform=.04")
    ap.add_argument("--identity-x", type=int, default=1, help="repeat the synth-identity rows this many times (dupes kept)")
    ap.add_argument("--typo-facts", type=int, default=0, help="extra fact rows with a typo in the title, kept only if the runtime still retrieves the same context")
    ap.add_argument("--shell-extra", type=int, default=0, help="extra generated shell pairs (own rng, other pools unchanged)")
    ap.add_argument("--numwords", type=int, default=20_000, help="spelled-number calc pairs, inside the calc share")
    a = ap.parse_args()
    MIX.update({k: float(v) for k, v in (x.split("=") for x in a.mix.split(",") if x)})
    assert abs(sum(MIX.values()) - 1) < 1e-9, MIX
    rng = random.Random(a.seed)
    stats = collections.defaultdict(collections.Counter)  # stat name -> per-category counts
    ctx_cache = {}

    def retrieve(q):
        if q not in ctx_cache:
            ctx_cache[q] = rt.retrieve(q, a.kb)
        return ctx_cache[q]

    def casual(p):  # 40%: drop the teacher's "Just curious:" wrapper so prompts look like real typing
        return WRAP.sub("", p) if rng.random() < .4 else p

    pools = collections.defaultdict(list)
    idk_seen = set()
    for r in load(a.full):
        c, q = r["category"], r["prompt"]
        bare = WRAP.sub("", q)
        if c == "arithmetic":
            stats["seen"]["calc-convert"] += 1
            e = extract_expr(bare)
            got = calc_answer(e) if e else None
            want = last_num(r["answer"].rstrip("."))
            if got and want is not None and Fraction(got) == want:
                stats["kept"]["calc-convert"] += 1
                pools["calc"].append({"prompt": casual(q), "answer": "calc(%s)" % e, "category": "calc", "source": "full-train"})
        elif c == "factual":
            stats["seen"]["factual"] += 1
            if not rt.wants_context(bare):  # not a fact-cue question: no context, nothing honest to teach
                stats["dropped_no_cue"]["factual"] += 1
                continue
            ctx = retrieve(bare)
            stats["hit"]["factual"] += bool(ctx)
            if ctx and supports(ctx, r["answer"]):
                stats["supported"]["factual"] += 1
                pools["factual"].append({"prompt": casual(q), "answer": ctx, "category": "factual",
                                         "source": "full-train+kb", "context": ctx})
            elif ctx:  # a context that doesn't hold the asked fact: honest answer is idk
                pools["idk-ctx"].append({"prompt": casual(q), "answer": IDK, "category": "factual",
                                         "source": "full-train-idk", "context": ctx})
            elif bare not in idk_seen:  # every distinct no-hit cue question, once
                idk_seen.add(bare)
                pools["factual"].append({"prompt": casual(q), "answer": IDK, "category": "factual", "source": "idk-nohit"})
        elif c in ("python", "bugfix"):
            pools["fullpy"].append({"prompt": q, "answer": r["answer"], "category": c, "source": "full-train"})
        else:
            if c == "chat" and re.search(r"\b(chatgpt|openai|claude|anthropic|deepseek|language model|ai assistant|helpful ai|assistant)\b",
                                         r["answer"], re.I):
                r["answer"] = ME
                stats["rewritten"]["chat"] += 1
            pools[c].append({"prompt": casual(q), "answer": r["answer"], "category": c, "source": "full-train"})
    pools["calc"] += synth_calc(rng, 30_000)
    pools["calc"] += synth_numwords(random.Random(a.seed + 7), int(a.numwords * 1.1))  # own rng: leaves every other pool unchanged
    pools["chat"] += identity_rows(rng) + joke_rows(rng)
    pools["factual"] += synth_idk_rows(rng, a.kb, 5_000)
    pools["factual"] += pools["idk-ctx"][:5_000]  # capped: they teach "context that doesn't answer -> idk"
    shell_cur = shell_rows(rng, 40_000)
    if a.shell_extra:
        shell_cur += shell_rows(random.Random(a.seed + 13), a.shell_extra)
    pools["factual"] += kb_title_rows(a.kb, rng, 45_000)
    short = {(r["prompt"], r["context"]): r["answer"] for r in load(a.fact_answers)}  # copied sentence -> short grounded answer
    kept, missing = [], {}
    for r in pools["factual"]:
        if r["source"] in ("kb-title", "full-train+kb"):
            stats["seen"]["short"] += 1
            if (r["prompt"], r["context"]) not in short:
                missing[(r["prompt"], r["context"])] = 1
                continue
            stats["kept"]["short"] += 1
            r["answer"] = short[(r["prompt"], r["context"])]
            r["source"] += "+short"
        kept.append(r)
    pools["factual"] = kept
    if a.dump_missing:
        with open(a.dump_missing, "w") as f:
            for q, c in missing:
                f.write(json.dumps({"prompt": q, "context": c}) + "\n")
        print("fact rows without an answer:", len(missing), "->", a.dump_missing)
    if a.typo_facts:  # typo'd titles: fuzzy retrieval still finds the context, the model must answer despite the garbled question
        import gen_chatds_golden as gg
        base = [r for r in kept if r["source"] == "kb-title+short" and r["prompt"] == r["prompt"].lower()]
        rng.shuffle(base)
        tr, added = random.Random(a.seed + 11), 0
        for r in base:
            if added >= a.typo_facts:
                break
            title = " ".join(r["context"].split()[:2]).lower()
            if not title.replace(" ", "").isalpha() or title not in r["prompt"]:
                continue
            q = r["prompt"].replace(title, gg.typo_title(title, tr), 1)
            if q != r["prompt"] and retrieve(q) == r["context"]:
                pools["factual"].append({**r, "prompt": q, "source": "typo+short"}); added += 1
        print("typo fact rows added:", added)
    stats["seen"]["synth-identity"] = sum(r["source"] == "synth-identity" for r in pools["chat"])
    cur = [{"prompt": r["prompt"], "answer": r["answer"], "category": r["category"], "source": "curriculum-v2"}
           for r in load(a.curriculum)]
    rng.shuffle(cur)
    rng.shuffle(pools["fullpy"])
    n_py = int(a.total * MIX["python+bugfix"])
    pools["shell"] += shell_cur
    pools["python+bugfix"] = cur[:int(n_py * .7)] + pools["fullpy"][:n_py - int(n_py * .7)]

    why = contamination_filter()
    tok, _ = export_train.load_tokenizer(a.tokenizer)
    drops = collections.defaultdict(collections.Counter)
    final = []
    for key, share in MIX.items():
        pool, seen = [], set()
        rng.shuffle(pools[key])
        for r in pools[key]:
            k = (r["prompt"], r["answer"], r.get("context"))
            if k in seen:
                continue
            seen.add(k)
            reason = why(r["prompt"], r["answer"])
            if reason:
                drops[r["category"]]["contam_" + reason] += 1
                continue
            pool.append(r)
        # sample down (or up, with replacement) to the share; synthetic idk/identity rows always stay
        want = int(a.total * share)
        keep = [r for r in pool if r["source"] in KEEP]
        rest = [r for r in pool if r["source"] not in KEEP]
        if key == "calc":  # 30% converted teacher rows, numword rows, the rest casual synthetic
            n_conv = int(want * .3)
            nw = [r for r in rest if r["source"] == "synth-numwords"][:a.numwords]
            rest = [r for r in rest if r["source"] == "full-train"][:n_conv] + nw + \
                   [r for r in rest if r["source"] == "synth-calc"][:want - n_conv - len(nw)]
        if key == "shell":  # ~90% curriculum, the rest teacher cp/mv
            n_cur = int(want * .9)
            rest = [r for r in rest if r["source"] == "shell-curriculum"][:n_cur] + \
                   [r for r in rest if r["source"] == "full-train"][:want - n_cur]
        if key == "chat":
            keep += [r for r in keep if r["source"] == "synth-identity"] * (a.identity_x - 1)
        need = want - len(keep)
        rest = rest[:need] if len(rest) >= need else rest + [rng.choice(rest) for _ in range(need - len(rest))]
        stats["pool"][key] = len(pool)
        final += keep + rest
    # irrelevant-context retrieval for everything that isn't factual (factual already has its own)
    for r in final:
        if r["category"] != "factual" and "context" not in r:
            stats["seen"]["ctx:" + r["category"]] += 1
            ctx = retrieve(WRAP.sub("", r["prompt"]))
            if ctx:
                r["context"] = ctx
                stats["hit"]["ctx:" + r["category"]] += 1
    # fit check
    out = []
    for r in final:
        if export_train.build_example(tok, r["prompt"], r["answer"], 256, r.get("context")) is None:
            drops[r["category"]]["too_long"] += 1
        else:
            out.append(r)
    rng.shuffle(out)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    import statistics
    extra = collections.defaultdict(list)  # extra prompt tokens the C: line costs, per category (0 when no context)
    for r in out:
        e = 0
        if "context" in r:
            e = len(tok.encode(rt.format_prompt(r["prompt"], r["context"]), bos=True, eos=False)) - \
                len(tok.encode(rt.format_prompt(r["prompt"]), bos=True, eos=False))
        extra[r["category"]].append(e)
    print("extra prompt tokens from context (all rows / rows with ctx): mean, p95")
    for c, v in sorted(extra.items()):
        w = sorted(x for x in v if x)
        print("  %-10s all %.1f/%d   with-ctx %.1f/%d" % (c, statistics.mean(v), sorted(v)[int(.95 * len(v))],
              statistics.mean(w) if w else 0, w[int(.95 * len(w))] if w else 0))
    n = collections.Counter(r["category"] for r in out)
    print("category      pairs   with-ctx  contam-drops  too-long")
    for c in sorted(n):
        print("%-12s %7d %9d %13d %9d" % (c, n[c], sum(1 for r in out if r["category"] == c and "context" in r),
                                          sum(v for k, v in drops[c].items() if k.startswith("contam")), drops[c]["too_long"]))
    print("total", len(out), "->", a.out)
    print("calc: convertible %d/%d (%.1f%%), synth %d, pool %d" % (
        stats["kept"]["calc-convert"], stats["seen"]["calc-convert"],
        100 * stats["kept"]["calc-convert"] / max(1, stats["seen"]["calc-convert"]), 30_000, stats["pool"]["calc"]))
    print("short grounded answers: %d/%d context rows had a validated answer" % (stats["kept"]["short"], stats["seen"]["short"]))
    print("factual KB hit rate: %d/%d (%.1f%%); hit whose text contains the teacher answer: %d (%.1f%%)" % (
        stats["hit"]["factual"], stats["seen"]["factual"], 100 * stats["hit"]["factual"] / stats["seen"]["factual"],
        stats["supported"]["factual"], 100 * stats["supported"]["factual"] / stats["seen"]["factual"]))
    for c in sorted(k[4:] for k in stats["seen"] if k.startswith("ctx:")):
        print("irrelevant-ctx hit rate %-10s %.1f%%" % (c, 100 * stats["hit"]["ctx:" + c] / stats["seen"]["ctx:" + c]))
    print("retrieval cost per gated question (data build): %.1f lookups, %.1f delete-index probes" % (
        rt.stats["lookups"] / max(1, rt.stats["queries"]), rt.stats["fuzzy_probes"] / max(1, rt.stats["queries"])))
    print("factual: cue-less questions dropped %d; idk rows (no-hit, incl. synth) %d" % (
        stats["dropped_no_cue"]["factual"], sum(r["answer"] == IDK and "context" not in r for r in out)))
    print("chat answers rewritten to chatds identity:", stats["rewritten"]["chat"])
    print("contamination drops (pool level):", {k: dict(v) for k, v in drops.items() if v})


if __name__ == "__main__":
    main()
