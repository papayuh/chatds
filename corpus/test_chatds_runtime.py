import json, os, sys
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "kb"))
import build_kb, chatds_runtime as rt  # noqa: E402


@pytest.fixture(scope="module")
def kbp(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("kb") / "t.kb")
    build_kb.build(os.path.join(HERE, "kb", "fixtures", "mini.xml"), out, "test")
    return out


def test_retrieve(kbp):
    assert rt.retrieve("what is the capital of france", kbp).startswith("France is a country")
    assert rt.retrieve("Who is Lutetia?", kbp).startswith("Paris is the capital city")
    assert rt.retrieve("hiii how are you", kbp) is None
    assert rt.retrieve("", kbp) is None


def test_retrieve_stopwords_and_short(kbp):
    # "the"/"of"/"is" are stopwords: never looked up even if they were keys
    calls = []
    orig = rt.kb.lookup
    rt.kb.lookup = lambda p, k: calls.append(k) or orig(p, k)
    try:
        rt.retrieve("what is the of an it", kbp)
        assert calls == []
        rt.retrieve("ab", kbp)  # 1-grams < 3 chars skipped
        assert calls == []
    finally:
        rt.kb.lookup = orig


def test_longest_first(kbp):
    calls = []
    orig = rt.kb.lookup
    rt.kb.lookup = lambda p, k: calls.append(k) or orig(p, k)
    try:
        assert rt.retrieve("tell me about cafe society please", kbp).startswith("Cafe society")
        assert calls[0].count(" ") == 3  # a 4-gram before any shorter one
        assert calls[-1] == "cafe society" and all(" " in c for c in calls)  # 2-gram hit before any 1-gram
    finally:
        rt.kb.lookup = orig


def test_retrieve_try_cap(kbp):
    calls = []
    orig = rt.kb.lookup
    rt.kb.lookup = lambda p, k: calls.append(k)
    try:
        assert rt.retrieve("what " + " ".join("zzz%d" % i for i in range(30)), kbp) is None
        assert len(calls) <= rt.MAX_TRIES
    finally:
        rt.kb.lookup = orig


@pytest.mark.parametrize("out,want", [
    ("calc(38*47)", "calc(38*47) = 1786"),
    (" calc(144/12) \n", "calc(144/12) = 12"),
    ("calc(10/3)", "calc(10/3) = 3.3333"),
    ("calc(2/3)", "calc(2/3) = 0.6667"),
    ("calc(1/8)", "calc(1/8) = 0.125"),
    ("calc(2.5+3.75)", "calc(2.5+3.75) = 6.25"),
    ("calc(80*0.15)", "calc(80*0.15) = 12"),
    ("calc(2+3*4)", "calc(2+3*4) = 14"),
    ("calc((2+3)*4)", "calc((2+3)*4) = 20"),
    ("calc(-5+2)", "calc(-5+2) = -3"),
    ("calc(5-9/4)", "calc(5-9/4) = 2.75"),
    ("calc(17%5)", "calc(17%5) = 2"),
    ("calc(-5/2)", "calc(-5/2) = -2.5"),
    ("calc(-1/3)", "calc(-1/3) = -0.3333"),
    ("calc(-2/3)", "calc(-2/3) = -0.6667"),
    ("calc(0.00005)", "calc(0.00005) = 0.0001"),
    ("calc(0.000049)", "calc(0.000049) = 0"),
    ("calc(0.00005*0.5)", "calc(0.00005*0.5) = 0.0001"),  # 0.0001*0.5 = 0.00005 rounds away from zero
    ("calc(-0.0001*0.5)", "calc(-0.0001*0.5) = -0.0001"),
    ("calc(0.0001*0.4)", "calc(0.0001*0.4) = 0"),
    ("calc(-7%3)", "calc(-7%3) = -1"),   # remainder keeps the dividend's sign
    ("calc(7%-3)", "calc(7%-3) = 1"),
    ("calc(5.5%2)", "calc(5.5%2) = 1.5"),
    ("calc( ( 1 + 2 ) * - 3 )", "calc(( 1 + 2 ) * - 3) = -9"),
    ("calc(--4)", "calc(--4) = 4"),
    ("calc(007+.5+1.)", "calc(007+.5+1.) = 8.5"),
    ("calc(999999999999*999999999999)", "calc(999999999999*999999999999) = error"),
    ("calc(922337203685477*10000)", "calc(922337203685477*10000) = error"),  # 9.2e18 scaled no longer fits int64
    ("calc(99999999999999999)", "calc(99999999999999999) = error"),
    ("calc(100000*100000)", "calc(100000*100000) = 10000000000"),
    ("calc(1/0)", "calc(1/0) = error"),
    ("calc(1/0+)", "calc(1/0+)"),          # syntax error beats /0
    ("calc(1.2.3)", "calc(1.2.3)"),
    ("calc(5 5)", "calc(5 5)"),
    ("calc()", "calc()"),
    ("calc(" + "(" * 40 + "1" + ")" * 40 + ")", "calc(" + "(" * 40 + "1" + ")" * 40 + ")"),
    ("calc(" + "(" * 8 + "1" + ")" * 8 + ")", "calc(" + "(" * 8 + "1" + ")" * 8 + ") = 1"),
    ("calc(5%0)", "calc(5%0) = error"),
    ("calc(__import__('os'))", "calc(__import__('os'))"),
    ("calc(2**9)", "calc(2**9)"),
    ("calc(1+)", "calc(1+)"),
    ("calc(9)) or 1", "calc(9)) or 1"),
    ("hello", "hello"),
])
def test_apply_calc(out, want):
    assert rt.apply_calc(out) == want


def test_format_prompt():
    assert rt.format_prompt("hi", None) == "Q: hi\nA:"
    assert rt.format_prompt("hi", "Ctx.") == "C: Ctx.\nQ: hi\nA:"


@pytest.mark.parametrize("q,ok", [
    ("who was george washington", True), ("what's the capital of france", True), ("tell me about paris", True),
    ("hiii", False), ("how do i list files", False), ("what is 12 + 7", False), ("whats 38 times 47", False),
    ("what does x = 5 mean", False), ("what is 'abc'", False), ("what is a dict", False), ("where is kenya", True),
    ("make a folder called pics", False), ("what is the code for paris", False),
    ("what is five times six", False), ("what is half of eighty", False), ("whats the sum of twelve and forty two", False),
    ("what is forty two over seven", False), ("what is eight and seventy four", False), ("whats a quarter of one hundred", False), ("who was henry the eighth", True), ("what is one direction", True),
])
def test_gate(q, ok):
    assert rt.wants_context(q) is ok


def test_context_is_one_sentence(kbp):
    assert rt.retrieve("what is france", kbp) == "France is a country in western Europe."
    assert rt.retrieve("how do i sort france", kbp) is None  # gated


def test_fuzzy_typos(kbp):
    assert rt.retrieve("what is franse", kbp).startswith("France is a country")
    assert rt.retrieve("who is lutetiaa", kbp).startswith("Paris is the capital")
    assert rt.retrieve("what is fraance", kbp).startswith("France is a country")
    assert rt.retrieve("what is frnace", kbp).startswith("France is a country")
    assert rt.retrieve("what is xyzqwv", kbp) is None
    # fuzzy needs every word of the n-gram to have >= 5 chars: "cafe" has 4
    assert rt.retrieve("what is cafe soceity", kbp) is None


def test_relevance_guard(kbp):
    # the only hit ("france") covers 1 of 3 content words -> rejected
    assert rt.retrieve("who is zorvath the france painter", kbp) is None
    # strict majority: 1 of 2 content words is not enough, 2 of 3 is
    assert rt.retrieve("who is zorvath france", kbp) is None
    assert rt.retrieve("who is the cafe society zorvath", kbp).startswith("Cafe society")
    assert rt.retrieve("what is france", kbp) is not None
