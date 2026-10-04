import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "reference", "ds-llm"))
import export_train as ex  # noqa: E402

sys.path.insert(0, os.path.join(HERE, '..', 'tools'))
from tokenizer_fixture import model_path
TOK = str(model_path())


def test_context_pair_and_python_only_bytes_unchanged():
    tok, _ = ex.load_tokenizer(TOK)
    seq, n = ex.build_example(tok, "who was george washington", "a president.", 256, "George was first.")
    assert tok.decode(seq[1:n]) == "C: George was first.\nQ: who was george washington\nA:"
    # python-only pair: identical ids to the pre-chatds encoding
    seq, n = ex.build_example(tok, "make name all caps", "name.upper()", 256)
    old = tok.encode("Q: make name all caps\nA:", bos=True, eos=False)
    assert seq[:n] == old and seq[n:] == tok.encode(" name.upper()", bos=False, eos=False) + [ex.EOS_ID]
