import random, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import build_chatds as b, chatds_runtime as rt, fact_answers as fa


def test_valid():
    fa.selftest()


def test_numwords_calc_and_gate():
    rows = b.synth_numwords(random.Random(3), 3000)
    assert all(r["answer"].startswith("calc(") and not rt.wants_context(r["prompt"]) for r in rows)
    assert b.spell(342, random.Random(0)).split()[0] == "three"
