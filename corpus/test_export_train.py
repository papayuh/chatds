"""Self-tests for export_train.py's tokenizer provenance. stdlib unittest, no
network, no model calls.

  python3 -m unittest discover -s corpus -p 'test_export_train.py'
  python3 -m pytest corpus/test_export_train.py -q

The claim being tested: meta.json's `tokenizer_fingerprint` describes the
tokenizer bytes THIS export actually loaded. A fingerprint recomputed later
from whatever file sits at the same path would be worthless -- that is the
mutation this guards (training touchup 4).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
import export_train as et  # noqa: E402
import tok_fingerprint as tf  # noqa: E402

import numpy as np  # noqa: E402

from tokenizer_fixture import model_path
TOKENIZER = str(model_path())
PAIRS = [{"prompt": "How do I print in Python?", "answer": "print('hi')"},
         {"prompt": "How do I reverse a list?", "answer": "xs[::-1]"},
         {"prompt": "How do I add two numbers?", "answer": "a + b"}]


def read_windows(out_dir, split, seq_len):
    """(ids, tgt) reshaped to (n_windows, seq_len) -- the shape
    train_instruct.MaskedShardDataset sees."""
    ids = np.fromfile(os.path.join(out_dir, f"{split}.ids.bin"), dtype=np.uint16)
    tgt = np.fromfile(os.path.join(out_dir, f"{split}.tgt.bin"), dtype=np.int32)
    assert len(ids) == len(tgt)
    assert len(ids) % seq_len == 0
    return ids.reshape(-1, seq_len), tgt.reshape(-1, seq_len)


def expected_independent_rows(tok, seq_len):
    """Independently recompute the exact window (ids, tgt) each accepted pair
    must produce under --packing independent: input = the whole sequence
    (including EOS when it fits) followed by a PAD_ID tail, targets = next
    token from the first answer token through EOS, -1 on prompt and pad
    positions."""
    rows = []
    for rec in PAIRS:
        seq, n_prompt = et.build_example(tok, rec["prompt"], rec["answer"], seq_len)
        L = min(len(seq), seq_len)
        ids = [et.PAD_ID] * seq_len
        ids[:L] = seq[:L]
        tgt = [-1] * seq_len
        for k in range(n_prompt - 1, len(seq) - 1):
            tgt[k] = seq[k + 1]
        rows.append((tuple(ids), tuple(tgt)))
    return rows


class ExportTrainProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.tok = os.path.join(self.dir, "tok.model")
        shutil.copyfile(TOKENIZER, self.tok)
        self.pairs = os.path.join(self.dir, "pairs.jsonl")
        with open(self.pairs, "w") as f:
            for _ in range(80):  # enough rows to fill more than one window
                for rec in PAIRS:
                    f.write(json.dumps(rec) + "\n")
        self.out = os.path.join(self.dir, "shards")

    def tearDown(self):
        self.tmp.cleanup()

    def export(self, extra=()):
        return subprocess.run(
            [sys.executable, os.path.join(HERE, "export_train.py"),
             "--pairs", self.pairs, "--tokenizer", self.tok, "--out-dir", self.out,
             "--seq-len", "64", "--val-count", "4", *extra],
            capture_output=True, text=True, timeout=600)

    def test_meta_records_the_full_hash_and_size_of_the_tokenizer_used(self):
        proc = self.export(("--selftest",))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("SELFTEST PASS", proc.stdout)
        with open(os.path.join(self.out, "meta.json")) as f:
            meta = json.load(f)
        fp = meta["tokenizer_fingerprint"]
        want = tf.fingerprint_file(self.tok)
        self.assertEqual(len(fp["sha256"]), 64, "digest must not be truncated")
        self.assertEqual(fp["sha256"], want["sha256"])
        self.assertEqual(fp["size"], want["size"])
        self.assertEqual(fp["path"], os.path.abspath(self.tok))

    def test_mutation_during_export_is_caught_before_meta_is_written(self):
        """The regression: swap the tokenizer at the same path while the shards
        are being packed. Recording the pre-swap hash would be a lie, recording
        the post-swap hash would be a worse one -- so we refuse and leave no
        meta.json, which makes train_instruct.py refuse the shards too."""
        fp = tf.fingerprint_file(self.tok)
        with open(self.tok, "wb") as f:
            f.write(b"a completely different vocabulary at the same path")
        with self.assertRaises(SystemExit) as cm:
            et.assert_tokenizer_unchanged(self.tok, fp)
        self.assertIn("changed while exporting", str(cm.exception))

    def test_loaded_processor_must_match_the_hashed_bytes(self):
        # Simulate a swap-and-restore between the byte read and the processor's
        # own file read. End-of-export path hashing alone cannot catch this.
        real_tokenizer = et.Tokenizer
        other = str(model_path(1024))
        with mock.patch.object(et, "Tokenizer", side_effect=lambda _: real_tokenizer(other)):
            with self.assertRaisesRegex(SystemExit, "loaded tokenizer"):
                et.load_tokenizer(self.tok)

    def test_failed_reexport_cannot_leave_old_success_metadata(self):
        self.assertEqual(self.export().returncode, 0)
        meta = os.path.join(self.out, "meta.json")
        self.assertTrue(os.path.exists(meta))
        argv = ["export_train.py", "--pairs", self.pairs, "--tokenizer", self.tok,
                "--out-dir", self.out, "--seq-len", "64", "--val-count", "4"]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
                et, "assert_tokenizer_unchanged", side_effect=SystemExit("changed while exporting")):
            with self.assertRaisesRegex(SystemExit, "changed while exporting"):
                et.main()
        self.assertFalse(os.path.exists(meta), "stale metadata authenticates a failed re-export")

    def test_unchanged_tokenizer_passes_the_guard(self):
        et.assert_tokenizer_unchanged(self.tok, tf.fingerprint_file(self.tok))

    def test_load_tokenizer_fingerprints_the_bytes_it_loaded(self):
        tok, fp = et.load_tokenizer(self.tok)
        self.assertEqual(tok.n_words, 2048)
        self.assertEqual(fp["sha256"], tf.fingerprint_file(self.tok)["sha256"])


class ExportTrainIndependentPackingTests(unittest.TestCase):
    """--packing independent: one example per window, BOS at position 0, no
    other question in the context, every answer token INCLUDING EOS supervised,
    prompt and pad positions masked. Written BEFORE the implementation
    (R1 evidence: packed-window teacher loss .034 vs independent-prompt .432 --
    the packing is the conditioning difference being isolated here)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.tok = os.path.join(self.dir, "tok.model")
        shutil.copyfile(TOKENIZER, self.tok)
        self.pairs = os.path.join(self.dir, "pairs.jsonl")
        with open(self.pairs, "w") as f:
            for _ in range(80):
                for rec in PAIRS:
                    f.write(json.dumps(rec) + "\n")
        self.out = os.path.join(self.dir, "shards")

    def tearDown(self):
        self.tmp.cleanup()

    def export(self, extra=(), pairs=None, seq_len=64):
        return subprocess.run(
            [sys.executable, os.path.join(HERE, "export_train.py"),
             "--pairs", pairs or self.pairs, "--tokenizer", self.tok,
             "--out-dir", self.out, "--seq-len", str(seq_len),
             "--val-count", "4", *extra],
            capture_output=True, text=True, timeout=600)

    def single_pair_export(self, rec, seq_len, extra=()):
        path = os.path.join(self.dir, "one.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps(rec) + "\n")
        return self.export(extra=extra, pairs=path, seq_len=seq_len), path

    def test_default_is_stream_and_metadata_records_packing(self):
        for extra in ((), ("--packing", "stream")):
            self.assertEqual(self.export(extra).returncode, 0)
            with open(os.path.join(self.out, "meta.json")) as f:
                self.assertEqual(json.load(f)["packing"], "stream")

    def test_independent_metadata_records_packing_and_true_counts(self):
        proc = self.export(("--packing", "independent", "--selftest"))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("SELFTEST PASS", proc.stdout)
        with open(os.path.join(self.out, "meta.json")) as f:
            meta = json.load(f)
        self.assertEqual(meta["packing"], "independent")
        for split in ("train", "val"):
            n_windows = meta[f"{split}_windows"]
            unpadded = meta[f"{split}_unpadded_input_tokens"]
            supervised = meta[f"{split}_supervised_positions"]
            padded = meta[f"{split}_padded_positions"]
            self.assertEqual(padded, n_windows * 64 - unpadded)
            self.assertGreater(padded, 0, "short pairs must actually be padded")
            self.assertGreater(supervised, 0)
            self.assertLess(supervised, unpadded, "prompt positions must be masked")
        self.assertEqual(meta["train_windows"] + meta["val_windows"], 240)

    def test_every_window_is_exactly_one_example_from_its_own_bos(self):
        """Per-window exact check: ids/tgt rows must equal the independently
        recomputed rule -- BOS at 0, example1 never prefixed by example2 (or
        followed by it), prompt/pad targets -1, first answer token through EOS
        supervised, pad tail after the last input token."""
        proc = self.export(("--packing", "independent"))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        train_ids, train_tgt = read_windows(self.out, "train", 64)
        val_ids, val_tgt = read_windows(self.out, "val", 64)
        got = sorted(list(zip((tuple(r) for r in train_ids), (tuple(r) for r in train_tgt)))
                     + list(zip((tuple(r) for r in val_ids), (tuple(r) for r in val_tgt))))
        tok, _ = et.load_tokenizer(self.tok)
        want = sorted(expected_independent_rows(tok, 64) * 80)
        self.assertEqual(got, want)
        # one BOS per window, at position 0: no packed neighbours anywhere
        self.assertTrue(bool((train_ids[:, 0] == et.BOS_ID).all()))
        self.assertTrue(bool((train_ids == et.BOS_ID).sum(axis=1).max() == 1))

    def test_boundary_example_of_seq_len_plus_one_tokens(self):
        """len(seq) == seq_len + 1: EOS cannot fit in the input window but is
        still supervised as the target of the final input position; no pad."""
        tok, _ = et.load_tokenizer(self.tok)
        rec = PAIRS[0]
        seq, n_prompt = et.build_example(tok, rec["prompt"], rec["answer"], 10 ** 9)
        L = len(seq)
        proc, _ = self.single_pair_export(rec, L - 1, ("--packing", "independent", "--selftest"))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        ids, tgt = read_windows(self.out, "train", L - 1)
        self.assertEqual(ids.shape, (1, L - 1))
        self.assertEqual(int(tgt[0, -1]), et.EOS_ID,
                         "final input position must predict EOS")
        self.assertTrue(bool((ids[0] != et.PAD_ID).all()), "no padding allowed")
        self.assertEqual(int((tgt[0, :n_prompt - 1] != -1).sum()), 0,
                         "prompt positions must be masked")
        self.assertEqual(int((tgt[0] != -1).sum()), L - n_prompt,
                         "every answer token incl EOS supervised exactly once")

    def test_boundary_example_of_exactly_seq_len_tokens(self):
        """len(seq) == seq_len: EOS sits in the last input slot, supervised at
        the position before it; the EOS slot's own target is masked and one
        pad slot follows nothing (window exactly full of real tokens)."""
        tok, _ = et.load_tokenizer(self.tok)
        rec = PAIRS[1]
        seq, n_prompt = et.build_example(tok, rec["prompt"], rec["answer"], 10 ** 9)
        L = len(seq)
        proc, _ = self.single_pair_export(rec, L, ("--packing", "independent", "--selftest"))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        ids, tgt = read_windows(self.out, "train", L)
        self.assertEqual(int(tgt[0, L - 2]), et.EOS_ID,
                         "position before EOS predicts EOS")
        self.assertEqual(int(ids[0, L - 1]), et.EOS_ID, "EOS fits in the input window")
        self.assertEqual(int(tgt[0, L - 1]), -1,
                         "nothing after EOS is ever supervised")
        self.assertTrue(bool((ids[0] != et.PAD_ID).all()), "window exactly full")

    def test_too_long_example_is_rejected_not_truncated(self):
        tok, _ = et.load_tokenizer(self.tok)
        rec = PAIRS[2]
        seq, _ = et.build_example(tok, rec["prompt"], rec["answer"], 10 ** 9)
        proc, _ = self.single_pair_export(rec, len(seq) - 2, ("--packing", "independent"))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        with open(os.path.join(self.out, "meta.json")) as f:
            meta = json.load(f)
        self.assertEqual(meta["accepted"], 0)
        self.assertEqual(meta["rejected"], 1)
        self.assertEqual(meta["train_windows"], 0)
        self.assertEqual(os.path.getsize(os.path.join(self.out, "train.ids.bin")), 0)

    def test_independent_rejects_replay_before_writing_anything(self):
        proc = self.export(("--packing", "independent", "--replay-pct", "5"))
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("replay", (proc.stderr + proc.stdout).lower())
        self.assertFalse(os.path.exists(os.path.join(self.out, "meta.json")))

    def test_unknown_packing_value_is_a_cli_error(self):
        proc = self.export(("--packing", "sideways"))
        self.assertNotEqual(proc.returncode, 0)

    def test_stream_mode_still_packs_multiple_examples_per_window(self):
        """Regression guard for the default: stream keeps back-to-back packing
        (several BOS per window), and its bins differ from independent's."""
        self.assertEqual(self.export().returncode, 0)
        s_ids, _ = read_windows(self.out, "train", 64)
        self.assertGreater(int((s_ids == et.BOS_ID).sum(axis=1).max()), 1,
                           "stream mode must keep packing neighbours")
        stream_train = s_ids.tobytes()
        self.out = os.path.join(self.dir, "shards-indep")
        self.assertEqual(self.export(("--packing", "independent")).returncode, 0)
        i_ids, _ = read_windows(self.out, "train", 64)
        self.assertNotEqual(i_ids.tobytes(), stream_train)


if __name__ == "__main__":
    unittest.main()


class LoadTextRowsTests(unittest.TestCase):
    def test_rows_are_bos_text_eos_within_seq_len(self):
        tok, _ = et.load_tokenizer(TOKENIZER)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.txt")
            open(p, "w").write("\n\n".join(["The cat sat on the mat. " * 4] * 30) + "\n")
            tr, va = et.load_text_rows(tok, p, 64, val_rows=2)
        self.assertEqual(len(va), 2)
        for r in tr + va:
            self.assertEqual((r[0], r[-1]), (et.BOS_ID, et.EOS_ID))
            self.assertLessEqual(len(r), 64)
