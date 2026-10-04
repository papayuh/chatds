"""C plumbing vs the Python reference (corpus/chatds_runtime.py, corpus/kb/kb.py).

KB context + prompt strings must match byte for byte over every eval question, the hwkit smoke
questions and the runtime test cases; calc over the reference cases plus a seeded fuzz. Also runs
the C unit tests (stub engine, ASan/UBSan) and, when the ARM toolchain is installed, checks ARM9
stack frames. The fixture KB is built from corpus/kb/fixtures/mini.xml; set CHATDS_KB to a real
kb.bin (or build corpus/kb/out/simplewiki.kb) to also check retrieval against the full KB."""
import glob, json, os, random, re, subprocess, sys
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "corpus"))
sys.path.insert(0, os.path.join(ROOT, "corpus", "kb"))
import build_kb, chatds_runtime as rt, kb as kbref  # noqa: E402
import test_chatds_runtime  # noqa: E402  (its calc/gate cases are the reference's own)

PROBE = os.path.join(HERE, "build", "cds-prompt")
SMOKE = ["who was george washington", "whats 38 times 47", "sort scores biggest first"]  # ds/hwkit/smoke-test.sh
EXTRA = [  # retrieval paths on the fixture KB (corpus/test_chatds_runtime.py) and gate edges
    "what is the capital of france", "Who is Lutetia?", "hiii how are you", "", "what is the of an it", "ab",
    "tell me about cafe society please", "what is france", "how do i sort france", "what is franse",
    "who is lutetiaa", "what is fraance", "what is frnace", "what is xyzqwv", "what is cafe soceity",
    "who is zorvath the france painter", "who is zorvath france", "who is the cafe society zorvath",
    "what " + " ".join("zzz%d" % i for i in range(20)), "What's  the CAPITAL of France?!", "tell me about",
    "tell me about paris", "TELL ME ABOUT  Paris", "what is 3 *4", "what is 3*", "what is - 4", "what is x'",
    "'what is france", "what is france'", "what is ma'am", "who is lutetia\t", "where is france\x1f+ 1",
    "define cafe society", "whats franse and frnace", "what is lutetia paris france cafe", "which france",
    "what is fr4nce", "what is the france of paris", "who is the zorvath", "describe   ", "explain: paris!!",
]


def esc(s):
    return "".join(c if 0x20 <= ord(c) < 0x7F and c != "\\" else "\\x%02x" % ord(c) for c in s)


def unesc(s):
    return re.sub(r"\\x([0-9a-f]{2})", lambda m: chr(int(m.group(1), 16)), s)


def probe(mode, lines, kb=None):
    p = subprocess.run([PROBE, mode] + ([kb] if kb else []), input="".join(esc(l) + "\n" for l in lines),
                       capture_output=True, text=True, check=True)
    out = p.stdout.split("\n")[:-1]
    assert len(out) == len(lines)
    return [unesc(o) for o in out]


def questions():
    qs = []
    for f in sorted(glob.glob(os.path.join(ROOT, "eval", "*.jsonl"))):
        qs += [json.loads(l)["prompt"] for l in open(f) if l.strip()]
    gate = [c[0] for c in test_chatds_runtime.test_gate.pytestmark[0].args[1]]
    qs = list(dict.fromkeys(qs + SMOKE + EXTRA + gate))
    assert all(len(q.encode()) <= 127 for q in qs)  # the DS editor cap; C refuses longer questions
    return qs


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    subprocess.run(["make", "-s", "-C", HERE, "all"], check=True)
    out = str(tmp_path_factory.mktemp("kb") / "mini.kb")
    build_kb.build(os.path.join(ROOT, "corpus", "kb", "fixtures", "mini.xml"), out, "test")
    return out


def check_prompts(kb):
    qs = questions()
    got = probe("prompt", qs, kb)
    want = []
    for q in qs:
        ctx = rt.retrieve(q, kb) if kb else None
        want.append((ctx or "") + "\t" + rt.format_prompt(q, ctx))
    bad = [(q, g, w) for q, g, w in zip(qs, got, want) if g != w]
    assert not bad, bad[:5]
    return sum(1 for w in want if not w.startswith("\t"))


def test_prompts_match_python_fixture_kb(built):
    assert check_prompts(built) >= 10  # the fixture-KB cases do retrieve


def test_prompts_match_python_no_kb(built):
    check_prompts(None)


REAL_KB = os.environ.get("CHATDS_KB") or os.path.join(ROOT, "corpus", "kb", "out", "simplewiki.kb")


@pytest.mark.skipif(not os.path.exists(REAL_KB), reason="no real KB (set CHATDS_KB)")
def test_prompts_match_python_real_kb(built):
    assert check_prompts(REAL_KB) > 100


def test_gate_matches_python(built):
    qs = questions()
    assert probe("gate", qs) == [str(int(rt.wants_context(q))) for q in qs]


def test_fuzzy_matches_python(built):
    words = ["franse", "frnace", "lutetiaa", "cafe", "zzzzzz", "france", "paaris", "societyy", "lutetia"]
    assert probe("fuzzy", words, built) == [" ".join(kbref.fuzzy_words(built, w)) for w in words]
    if os.path.exists(REAL_KB):
        ws = sorted({w for q in questions() for w in kbref.normalize(q).split() if len(w) >= 5})[:300]
        assert probe("fuzzy", ws, REAL_KB) == [" ".join(kbref.fuzzy_words(REAL_KB, w)) for w in ws]


def test_word_lists_match_python(built):
    got = dict(l.split(": ", 1) for l in subprocess.run([PROBE, "lists"], capture_output=True, text=True,
                                                         check=True).stdout.splitlines())
    for name in ("STOP", "CODE_WORDS", "NUM_WORDS", "MATH_WORDS"):
        assert set(got[name].split()) == getattr(rt, name), name
    assert tuple(got["CUES"].split()) == rt.CUES and set(got["CODE_CHARS"]) == rt.CODE_CHARS


def calc_inputs():
    cases = [c[0] for c in test_chatds_runtime.test_apply_calc.pytestmark[0].args[1]]
    r = random.Random(1789)
    alpha = "0123456789" * 3 + " +-*/().%" + ".."
    for _ in range(4000):
        body = "".join(r.choice(alpha) for _ in range(r.randint(1, 30)))
        cases.append(r.choice(["", " ", "\n", "\t "]) + "calc(" + body + ")" + r.choice(["", " ", "\n", ")", "x"]))
    for _ in range(500):  # well-formed, with big and fractional operands to hit rounding and overflow
        n = lambda: r.choice(["", "-"]) + str(r.randint(0, 10 ** r.randint(0, 19))) + r.choice(
            ["", "." + str(r.randint(0, 10 ** r.randint(1, 7)))])
        cases.append("calc(%s%s%s%s%s)" % (n(), r.choice("+-*/%"), n(), r.choice("+-*/%"), n()))
    return cases


def test_calc_matches_python(built):
    cases = calc_inputs()
    got = probe("calc", cases)
    bad = [(c, g, rt.apply_calc(c)) for c, g in zip(cases, got) if g != rt.apply_calc(c)]
    assert not bad, bad[:5]


def test_c_unit(built, tmp_path):
    p = subprocess.run([os.path.join(HERE, "build", "test-plumbing"), built, str(tmp_path) + "/"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr


ARMCC = os.path.join(os.environ.get("WONDERFUL_TOOLCHAIN", os.path.expanduser("~/opt/wonderful")),
                     "toolchain", "gcc-arm-none-eabi", "bin", "arm-none-eabi-gcc")


@pytest.mark.skipif(not os.path.exists(ARMCC), reason="no BlocksDS ARM toolchain")
def test_arm9_builds_with_small_static_frames():
    subprocess.run(["make", "-s", "-C", HERE, "arm"], check=True)
    frames = [l.split("\t") for f in glob.glob(os.path.join(HERE, "build", "arm", "*.su")) for l in open(f)]
    assert frames and all(kind == "static\n" for _, _, kind in frames)
    worst = max(int(size) for _, size, _ in frames)
    assert worst <= 1024, worst  # DTCM stack is ~16 KiB total; calc recursion is capped at depth 8
