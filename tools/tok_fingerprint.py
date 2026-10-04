"""Tokenizer provenance: one hashing + comparison rule for every tool that
has to prove a checkpoint, a shard set and a .bin all came from the SAME
SentencePiece model.

Why this exists: tok2048 (TinyStories) and tok2048-s2 (corpus) are both 2048
pieces and completely different vocabularies. Equal vocab_size is NEVER
evidence of a pairing -- pairing them trains happily and generates noise
(the tokenizer-provenance contract). The only evidence is a content hash recorded at the
moment the tokenizer was USED, not one recomputed later from whatever file
sits at the same path.

Fingerprint shape: {"path": abs path, "sha256": 64 hex chars, "size": bytes}.

Legacy note: train_instruct.py before 2026-09-09 recorded sha256[:16].
compare() therefore matches on the shorter of the two digests, requiring at
least MIN_DIGEST hex chars (64 bits) -- deliberate, so old checkpoints stay
usable -- and treats anything shorter as no evidence at all.
"""
import hashlib
import os

MIN_DIGEST = 16  # hex chars; below this a prefix match proves nothing

MATCH = "match"
MISMATCH = "mismatch"
UNKNOWN = "unknown"


def fingerprint_bytes(blob, path=None):
    """Fingerprint of the exact bytes a tool loaded (mutation-proof)."""
    fp = {"sha256": hashlib.sha256(blob).hexdigest(), "size": len(blob)}
    if path is not None:
        fp["path"] = os.path.abspath(path)
    return fp


def fingerprint_file(path):
    """Fingerprint of a file on disk, or None if it isn't readable."""
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return fingerprint_bytes(f.read(), path)


def compare(a, b):
    """MATCH / MISMATCH / UNKNOWN for two fingerprints.

    UNKNOWN means "no provenance recorded" -- the only case an operator may
    override. MISMATCH is proof of a wrong pairing and is never overridable.
    Paths are ignored on purpose: the same tokenizer copied next to a run is
    still the same tokenizer, and the same path holding different bytes is
    still a mismatch.
    """
    sa = (a or {}).get("sha256")
    sb = (b or {}).get("sha256")
    if not sa or not sb:
        return UNKNOWN
    za, zb = (a or {}).get("size"), (b or {}).get("size")
    if za is not None and zb is not None and za != zb:
        return MISMATCH
    n = min(len(sa), len(sb))
    if n < MIN_DIGEST:
        return UNKNOWN
    return MATCH if sa[:n].lower() == sb[:n].lower() else MISMATCH


def describe(fp):
    """Short human form for error messages."""
    if not fp:
        return "<no fingerprint>"
    sha = fp.get("sha256") or "?"
    return f"{os.path.basename(fp.get('path') or '?')} sha={sha[:16]} size={fp.get('size', '?')}"


def demo():
    import tempfile

    blob = b"pretend-sentencepiece-model"
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.model")
        with open(p, "wb") as f:
            f.write(blob)
        fp = fingerprint_file(p)
        assert fp["sha256"] == hashlib.sha256(blob).hexdigest()
        assert fp["size"] == len(blob)
        assert compare(fp, fingerprint_bytes(blob)) == MATCH
        # same path, mutated content -> proven mismatch
        with open(p, "wb") as f:
            f.write(b"a completely different vocabulary")
        assert compare(fp, fingerprint_file(p)) == MISMATCH
        # legacy 16-char digest still matches its full-length self
        legacy = {"sha256": fp["sha256"][:16]}
        assert compare(legacy, fp) == MATCH
        assert compare({"sha256": fp["sha256"][:8]}, fp) == UNKNOWN
        assert compare(None, fp) == UNKNOWN
        # equal digest prefix but different size is still a mismatch
        assert compare({"sha256": fp["sha256"], "size": 1}, fp) == MISMATCH
    print("tok_fingerprint demo: PASS")


if __name__ == "__main__":
    demo()
