import os, struct
import build_kb, kb

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "mini.xml")

def test_kb(tmp_path):
    out = str(tmp_path / "t.kb")
    st = build_kb.build(FIX, out, "test")
    hd = open(out, "rb").read(512)
    assert hd[:4] == b"KB3\0" and struct.unpack("<I", hd[4:8])[0] == 3
    assert struct.unpack("<I", hd[32:36])[0] == os.path.getsize(out)
    assert st["n_records"] == 3 and st["n_keys"] == 4  # france, paris+lutetia, cafe society
    fr = kb.lookup(out, "France")
    assert fr == "France is a country in western Europe."
    assert kb.lookup(out, "  PARIS!") == "Paris is the capital city of France."
    assert kb.lookup(out, "lutetia") == kb.lookup(out, "paris")
    assert kb.lookup(out, "cafe society") == 'Cafe society was a group of "fashionable" people - rich ones - in the 1920s.'
    for miss in ("mercury", "list of rivers", "tiny", "talk paris", "nothing here", ""):
        assert kb.lookup(out, miss) is None
    kb.sector_reads = 0; kb._last = None
    kb.lookup(out, "france")
    assert kb.sector_reads <= 3


def test_fuzzy(tmp_path):
    out = str(tmp_path / "t.kb")
    build_kb.build(FIX, out, "test")
    assert "seine" not in open(out, "rb").read(512).decode("ascii", "ignore")
    assert kb.fuzzy_words(out, "franse") == ["france"]      # substitution
    assert kb.fuzzy_words(out, "frnace") == ["france"]      # transposition
    assert kb.fuzzy_words(out, "lutetiaa") == ["lutetia"]   # extra char
    assert kb.fuzzy_words(out, "cafe") == [] and kb.fuzzy_words(out, "zzzzzz") == []
    assert kb.fuzzy_words(out, "france") == []              # exact word is not its own correction


def test_context_sentence():
    c = build_kb.context_sentence
    assert c("Kuching is the capital of Sarawak. It is big.") == "Kuching is the capital of Sarawak."
    long = "Jenins is a municipality in the district of Landquart in the canton of Graubuenden in Switzerland, near a river."
    assert c(long) == "Jenins is a municipality in the district of Landquart in the canton of Graubuenden in Switzerland."
    # clause cut: longest prefix ending at , ; : or before and/which/who/that
    x = c("George Washington was the first president of the United States from 1789 to 1797 and a leader of the revolution.")
    assert x == "George Washington was the first president of the United States from 1789 to 1797."
    # no boundary: drop trailing function words, end with a period
    y = c("Photosynthesis is a process in which green plants make their own food from sunlight using chlorophyll inside leaves")
    assert y.endswith(".") and len(y) <= 100 and y.split(" ")[-1][:-1] not in build_kb._FUNC
    z = c("A " * 10 + "word " * 40)
    assert len(z) <= 100 and z.endswith(".")
