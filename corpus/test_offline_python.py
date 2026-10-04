"""Self-tests for offline_python.py. stdlib unittest, no network, no model
calls, no snippet execution -- everything runs against the real frozen
eval suite shipped in this repo plus temp files."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import offline_python as op
import postprocess as pp
from python_tasks import TASKS


class OfflinePythonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = os.path.join(cls.tmp.name, "offline.jsonl")
        cls.pairs, cls.stats = op.generate(cls.out, seed=42)
        with open(cls.out) as f:
            cls.records = [json.loads(line) for line in f]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_nonempty_and_schema(self):
        self.assertGreater(len(self.records), 0)
        canonical = {a for _, a, _ in TASKS}
        for r in self.records:
            self.assertEqual(set(r), {"prompt", "answer", "category", "tier",
                                      "source", "task_family"})
            self.assertTrue(r["prompt"].strip() and r["answer"].strip())
            self.assertEqual(r["category"], op.CATEGORY)
            self.assertEqual(r["source"], op.SOURCE)
            self.assertIn(r["tier"], (1, 2, 3))
            self.assertTrue(r["task_family"])
            self.assertIn(r["answer"], canonical)  # answer is verbatim trusted

    def test_unique_and_modest(self):
        keys = {(pp.normalize_key(r["prompt"]), pp.normalize_key(r["answer"]))
                for r in self.records}
        self.assertEqual(len(keys), len(self.records))
        self.assertLessEqual(len(self.records), len(TASKS) * len(op.TEMPLATES))

    def test_reproducible_and_idempotent(self):
        out2 = os.path.join(self.tmp.name, "offline2.jsonl")
        op.generate(out2, seed=42)
        with open(out2) as f2, open(self.out) as f1:
            self.assertEqual(f2.read(), f1.read())
        op.generate(self.out, seed=42)  # rewrite same path: no error, same bytes
        with open(self.out) as f:
            self.assertEqual(len(f.readlines()), len(self.records))

    def test_seed_changes_order_not_content(self):
        out3 = os.path.join(self.tmp.name, "offline3.jsonl")
        op.generate(out3, seed=7)
        with open(out3) as f:
            recs3 = [json.loads(l) for l in f]
        self.assertEqual({(r["prompt"], r["answer"]) for r in recs3},
                         {(r["prompt"], r["answer"]) for r in self.records})
        self.assertNotEqual([r["prompt"] for r in recs3],
                            [r["prompt"] for r in self.records])

    def test_no_contamination_vs_frozen_suite(self):
        prompt_sets, exact_prompts, _, eval_ngrams = pp.load_eval_overlap_sets()
        for r in self.records:
            self.assertNotIn(pp.normalize_key(r["prompt"]), exact_prompts)
            self.assertFalse(pp.ngrams(pp.normalize_word_list(r["prompt"])) & eval_ngrams)
            self.assertFalse(pp.ngrams(pp.normalize_word_list(r["answer"])) & eval_ngrams)
            self.assertLess(pp.max_overlap(pp.normalize_words(r["prompt"]), prompt_sets),
                            pp.OVERLAP_THRESHOLD)

    def test_extra_eval_rejects_matching_prompt(self):
        victim = self.records[0]["prompt"]
        extra = os.path.join(self.tmp.name, "extra.jsonl")
        with open(extra, "w") as f:
            f.write(json.dumps({"id": "x-1", "prompt": victim,
                                "answer": "anything at all here"}) + "\n")
        out4 = os.path.join(self.tmp.name, "offline4.jsonl")
        _, stats = op.generate(out4, seed=42, extra_eval=extra)
        with open(out4) as f:
            kept_prompts = {json.loads(l)["prompt"] for l in f}
        self.assertNotIn(victim, kept_prompts)
        self.assertGreaterEqual(stats["drop_contamination_exact_prompt"], 1)
        # sibling phrasings of the same task may also fall to the Jaccard
        # rule (they share the task's words with the injected prompt)
        self.assertLess(len(kept_prompts), len(self.records))
        self.assertGreaterEqual(len(kept_prompts), len(self.records) - len(op.TEMPLATES))

    def test_syntax_labels_parse_only(self):
        self.assertEqual(op.syntax_label("x = 5"), "exec")
        self.assertEqual(op.syntax_label("for x in xs:"), "block")
        self.assertEqual(op.syntax_label("except ValueError:"), "fragment")
        self.assertIn(op.syntax_label("a & b"), ("exec", "eval"))
        self.assertEqual(self.stats["syntax"].get("fragment", 0)
                         + sum(v for k, v in self.stats["syntax"].items()
                               if k != "fragment"),
                         self.stats["candidates"])

    def test_code_whitespace_preserved(self):
        multiline = [r for r in self.records if "\n" in r["answer"]]
        self.assertTrue(multiline)  # e.g. with/try/class reference answers
        for r in multiline:
            self.assertIn("\n    ", r["answer"])  # 4-space indent intact
            self.assertTrue(any(r["answer"] == a for _, a, _ in TASKS))

    def test_refuses_to_mutate_full_corpus(self):
        for protected in op.PROTECTED_OUTPUTS:
            with self.assertRaises(SystemExit):
                op.generate(protected, seed=42)

    def test_never_executes_snippets(self):
        # A snippet that would hang if executed is still classified purely
        # by parsing -- generation never runs snippet code.
        self.assertIn(op.syntax_label("while True:\n    pass"), ("exec", "block"))
        self.assertEqual(op.syntax_label("while True:"), "block")
        self.assertIn("while True:", {a for _, a, _ in TASKS})


if __name__ == "__main__":
    unittest.main()
