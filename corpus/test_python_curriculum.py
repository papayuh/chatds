"""Self-tests for python_curriculum.py (R3 data-coverage experiment).

stdlib unittest only. No network, no .env, no model, no DS device. Every
generated snippet is validated by PARSING ONLY -- nothing here ever
executes or evaluates generated code. The frozen eval suite is touched
exclusively through postprocess.load_eval_overlap_sets (sha-pinned).
"""
import ast
import builtins
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import postprocess as pp
import python_curriculum as pc


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class CurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = tempfile.TemporaryDirectory(prefix="pycur-test-")
        cls.addClassCleanup(tmp.cleanup)
        cls.tmp = tmp.name
        cls.out_dir = os.path.join(cls.tmp, "run")
        cls.manifest = pc.run(cls.out_dir, seed=42)
        cls.train = load_jsonl(os.path.join(cls.out_dir, "python-curriculum-train.jsonl"))
        cls.dev = load_jsonl(os.path.join(cls.out_dir, "python-curriculum-dev.jsonl"))
        # full records (with _bindings) for structural tests
        cls.train_full = pc.build_candidates("train")
        cls.dev_full = pc.build_candidates("dev")

    # ---------------------------------------------------------- schema

    def test_train_schema(self):
        self.assertGreaterEqual(len(self.train), 25_000)
        self.assertLessEqual(len(self.train), 60_000)
        for r in self.train:
            self.assertEqual(set(r), {"id", "prompt", "answer", "category",
                                      "tier", "source", "task_family", "template_id"})
            self.assertIn(r["category"], ("python", "bugfix"))
            self.assertEqual(r["source"], pc.SOURCE)
            self.assertIn(r["task_family"], pc.FAMILY_NAMES)
            self.assertIn(r["tier"], (1, 2, 3))
            self.assertTrue(r["template_id"].startswith(r["task_family"] + "/"))
            self.assertTrue(r["id"].startswith("curr-tr-"))
            self.assertTrue(r["prompt"].strip() and r["answer"].strip())

    def test_dev_schema(self):
        self.assertGreaterEqual(len(self.dev), 500)
        ids = [r["id"] for r in self.dev]
        self.assertEqual(len(ids), len(set(ids)))
        for r in self.dev:
            for key in ("id", "category", "prompt", "answer", "accept",
                        "scoring", "max_new_tokens", "tier"):
                self.assertIn(key, r)
            self.assertEqual(r["scoring"], "python-ast")
            self.assertIsInstance(r["accept"], list)
            self.assertEqual(len(r["accept"]), 1)
            self.assertIsInstance(r["max_new_tokens"], int)
            self.assertGreater(r["max_new_tokens"], 0)
            self.assertIn(r["category"], ("python", "bugfix"))
            self.assertIn(r["tier"], (1, 2, 3))

    def test_dev_file_loads_in_eval_scorer_and_canonical_answers_pass(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "eval"))
        import score
        dev_path = os.path.join(self.out_dir, "python-curriculum-dev.jsonl")
        suite = score.load_suite(dev_path)  # same schema gate as frozen suite
        rows = [{"id": it["id"], "output": it["answer"], "tokens_generated": 0}
                for it in suite]
        by_id, viol = score.load_outputs([json.dumps(r) for r in rows],
                                         {it["id"] for it in suite})
        res = score.report(suite, by_id, {it["id"] for it in suite},
                           violations=viol)
        self.assertEqual(res["correct"], res["scored"], len(suite))
        self.assertEqual(res["correct"], len(suite))
        self.assertFalse(res["violations"])

    # ------------------------------------------------- reproducibility

    def test_byte_reproducible(self):
        out2 = os.path.join(self.tmp, "run2")
        pc.run(out2, seed=42)
        for name in ("python-curriculum-train.jsonl",
                     "python-curriculum-dev.jsonl", "manifest.json"):
            with open(os.path.join(self.out_dir, name), "rb") as f1, \
                 open(os.path.join(out2, name), "rb") as f2:
                self.assertEqual(f1.read(), f2.read(), name)

    def test_seed_changes_order_not_content(self):
        out3 = os.path.join(self.tmp, "run3")
        pc.run(out3, seed=7)
        recs = load_jsonl(os.path.join(out3, "python-curriculum-train.jsonl"))
        self.assertEqual({(r["prompt"], r["answer"]) for r in recs},
                         {(r["prompt"], r["answer"]) for r in self.train})
        self.assertNotEqual([r["prompt"] for r in recs], [r["prompt"] for r in self.train])

    # ------------------------------------------------------ uniqueness

    def test_unique_within_and_across_splits(self):
        def keys(recs):
            return {(pp.normalize_key(r["prompt"]), pp.normalize_key(r["answer"]))
                    for r in recs}
        self.assertEqual(len(keys(self.train)), len(self.train))
        self.assertEqual(len(keys(self.dev)), len(self.dev))
        self.assertFalse(keys(self.train) & keys(self.dev))
        prompts_tr = {pp.normalize_key(r["prompt"]) for r in self.train}
        prompts_dev = {pp.normalize_key(r["prompt"]) for r in self.dev}
        self.assertFalse(prompts_tr & prompts_dev)

    # ------------------------------------------------------- splitting

    def test_no_shared_template_ids(self):
        tr = {r["template_id"] for r in self.train}
        dv = {r["template_id"] for r in self.dev}
        self.assertFalse(tr & dv)

    def test_pool_parameter_disjointness(self):
        for pool in pc.POOLS.values():
            train_vals = {str(v) for v in pool["train"]}
            dev_vals = {str(v) for v in pool["dev"]}
            self.assertFalse(train_vals & dev_vals)
        for recs, split in ((self.train_full, "train"), (self.dev_full, "dev")):
            other = "dev" if split == "train" else "train"
            for rec in recs:
                slots = pc.RECIPE_BY_FAMILY[rec["task_family"]]["slots"]
                for slot, value in rec["_bindings"].items():
                    pool = pc.POOLS[slots[slot]]
                    self.assertIn(str(value), {str(v) for v in pool[split]})
                    self.assertNotIn(str(value), {str(v) for v in pool[other]})

    def test_pools_are_safe_identifiers_and_not_builtins(self):
        import keyword
        for pool in pc.POOLS.values():
            for split in ("train", "dev"):
                for v in pool[split]:
                    self.assertIsInstance(v, (str, int, float))
                    if isinstance(v, str) and pool["kind"] == "name":
                        self.assertTrue(v.isidentifier(), v)
                        self.assertFalse(keyword.iskeyword(v), v)
                        self.assertNotIn(v, dir(builtins), v)

    # --------------------------------------------- prompt/answer limits

    def test_prompt_constraints(self):
        for recs in (self.train, self.dev):
            for r in recs:
                p = r["prompt"]
                self.assertNotIn("\n", p)
                self.assertNotIn("\\", p)
                self.assertTrue(p.isascii())
                self.assertLessEqual(len(p.encode("utf-8")), 127)
                self.assertTrue(all(32 <= ord(c) < 127 for c in p))
                self.assertGreaterEqual(len(p.split()), 4)

    def test_device_budget_conservative(self):
        # Character-size sanity bound, NOT a chars/4 token estimate.
        # Export preparation checks actual serialized tokens, including
        # prompt delimiters, BOS and final EOS, with the pinned tokenizer.
        for recs in (self.train, self.dev):
            for r in recs:
                total = len(r["prompt"]) + 1 + len(r["answer"])
                self.assertLessEqual(total, 220, r["prompt"])

    def test_every_answer_parses_as_exec_snippet(self):
        for recs in (self.train, self.dev):
            for r in recs:
                ast.parse(r["answer"])  # parse-only, never exec'd
        for split in ("train", "dev"):
            self.assertEqual(pc.PIPELINE_STATS[split]["syntax_fail"], 0)
            self.assertEqual(pc.PIPELINE_STATS[split]["drop_prompt_shape"], 0)

    # --------------------------------------------- semantics by parsing

    def test_literal_and_identifier_binding(self):
        for recs in (self.train_full, self.dev_full):
            for rec in recs:
                names = pc.answer_identifiers(rec["answer"])
                consts = pc.answer_constants(rec["answer"])
                slots = pc.RECIPE_BY_FAMILY[rec["task_family"]]["slots"]
                for slot, value in rec["_bindings"].items():
                    kind = pc.POOLS[slots[slot]]["kind"]
                    if kind == "name":
                        self.assertIn(value, names, (rec["prompt"], rec["answer"], slot))
                    elif kind in ("quote", "bare"):
                        self.assertIn(value, {c for c in consts if isinstance(c, str)},
                                      (rec["prompt"], rec["answer"], slot))
                    else:
                        self.assertIn(value, {c for c in consts
                                              if isinstance(c, (int, float))
                                              and not isinstance(c, bool)},
                                      (rec["prompt"], rec["answer"], slot))

    def test_representative_known_task_references(self):
        # Reference semantics for EVERY record of every family: the answer
        # must equal the independently written expected formula for its
        # recorded bindings (AST-dumped equality, not just parse success).
        F = {
            "print_literal": lambda b: f"print({b['lit']!r})",
            "print_variable": lambda b: f"print({b['n']})",
            "print_message": lambda b: f"print({b['msg']!r})",
            "assign_literal": lambda b: f"{b['n']} = {b['k']}",
            "str_int_literal": lambda b: f"int({b['lit']!r})",
            "num_str_literal": lambda b: f"str({b['k']})",
            "abs_literal": lambda b: f"abs({b['k']})",
            "str_len_literal": lambda b: f"len({b['lit']!r})",
            "list_first": lambda b: f"{b['xs']}[0]",
            "list_last": lambda b: f"{b['xs']}[-1]",
            "list_first_n": lambda b: f"{b['xs']}[:{b['k']}]",
            "list_sorted_desc": lambda b: f"sorted({b['xs']}, reverse=True)",
            "list_max": lambda b: f"max({b['xs']})",
            "list_min": lambda b: f"min({b['xs']})",
            "list_sum": lambda b: f"sum({b['xs']})",
            "list_len": lambda b: f"len({b['xs']})",
            "list_count": lambda b: f"{b['xs']}.count({b['k']})",
            "list_append": lambda b: f"{b['xs']}.append({b['k']})",
            "list_concat": lambda b: f"{b['xs']} + {b['ys']}",
            "list_contains": lambda b: f"{b['k']} in {b['xs']}",
            "list_sorted_copy": lambda b: f"sorted({b['xs']})",
            "list_sort_inplace": lambda b: f"{b['xs']}.sort()",
            "list_reverse_slice": lambda b: f"{b['xs']}[::-1]",
            "list_reverse_inplace": lambda b: f"{b['xs']}.reverse()",
            "str_upper": lambda b: f"{b['s']}.upper()",
            "str_lower": lambda b: f"{b['s']}.lower()",
            "str_strip": lambda b: f"{b['s']}.strip()",
            "str_split": lambda b: f"{b['s']}.split()",
            "str_split_sep": lambda b: f"{b['s']}.split({b['sep']!r})",
            "str_join": lambda b: f"{b['sep']!r}.join({b['xs']})",
            "str_replace": lambda b: f"{b['s']}.replace({b['lit']!r}, {b['lit2']!r})",
            "str_startswith": lambda b: f"{b['s']}.startswith({b['lit']!r})",
            "str_contains": lambda b: f"{b['lit']!r} in {b['s']}",
            "str_len": lambda b: f"len({b['s']})",
            "str_concat": lambda b: f"{b['s']} + {b['t']}",
            "dict_get": lambda b: f"{b['d']}[{b['key']!r}]",
            "dict_get_default": lambda b: f"{b['d']}.get({b['key']!r}, {b['k']})",
            "dict_set": lambda b: f"{b['d']}[{b['key']!r}] = {b['k']}",
            "dict_has_key": lambda b: f"{b['key']!r} in {b['d']}",
            "dict_keys": lambda b: f"{b['d']}.keys()",
            "dict_values": lambda b: f"{b['d']}.values()",
            "dict_pop": lambda b: f"{b['d']}.pop({b['key']!r})",
            "dict_from_zip": lambda b: f"dict(zip({b['xs']}, {b['ys']}))",
            "set_union": lambda b: f"{b['a']} | {b['b']}",
            "set_intersection": lambda b: f"{b['a']} & {b['b']}",
            "set_difference": lambda b: f"{b['a']} - {b['b']}",
            "set_add": lambda b: f"{b['a']}.add({b['k']})",
            "set_remove": lambda b: f"{b['a']}.remove({b['k']})",
            "set_contains": lambda b: f"{b['k']} in {b['a']}",
            "set_from_list": lambda b: f"set({b['xs']})",
            "tuple_index": lambda b: f"{b['t']}[{b['k']}]",
            "tuple_len": lambda b: f"len({b['t']})",
            "tuple_unpack": lambda b: f"{b['a']}, {b['b']} = {b['t']}",
            "tuple_from_list": lambda b: f"tuple({b['xs']})",
            "str_to_int": lambda b: f"int({b['s']})",
            "str_to_float": lambda b: f"float({b['s']})",
            "num_to_str": lambda b: f"str({b['n']})",
            "float_trunc": lambda b: f"int({b['f']})",
            "abs_value": lambda b: f"abs({b['n']})",
            "round_num": lambda b: f"round({b['f']})",
            "round_places": lambda b: f"round({b['f']}, 2)",
            "floor_div": lambda b: f"{b['n']} // {b['k']}",
            "mod": lambda b: f"{b['n']} % {b['k']}",
            "power": lambda b: f"{b['n']} ** {b['k']}",
            "range_stop": lambda b: f"range({b['k']})",
            "range_start_stop": lambda b: f"range({b['a']}, {b['b']})",
            "range_step": lambda b: f"range({b['a']}, {b['b']}, {b['step']})",
            "range_to_list": lambda b: f"list(range({b['a']}, {b['b']}))",
            "enumerate_loop": lambda b: (f"for {b['i']}, {b['x']} in "
                                         f"enumerate({b['xs']}):\n"
                                         f"    print({b['i']}, {b['x']})"),
            "zip_loop": lambda b: (f"for {b['a']}, {b['b']} in "
                                   f"zip({b['xs']}, {b['ys']}):\n"
                                   f"    print({b['a']}, {b['b']})"),
            "def_add": lambda b: f"def {b['fn']}({b['a']}, {b['b']}): return {b['a']} + {b['b']}",
            "def_default": lambda b: (f"def {b['fn']}({b['a']}, {b['b']}={b['k']}): "
                                      f"return {b['a']} + {b['b']}"),
            "lambda_double": lambda b: f"{b['fn']} = lambda {b['x']}: {b['x']} * {b['k']}",
            "even_check": lambda b: f"{b['n']} % 2 == 0",
            "squares_comp": lambda b: f"[{b['x']} ** 2 for {b['x']} in range({b['k']})]",
            "filter_comp": lambda b: f"[{b['x']} for {b['x']} in {b['xs']} if {b['x']} > {b['k']}]",
            "upper_comp": lambda b: f"[{b['x']}.upper() for {b['x']} in {b['xs']}]",
            "bf_if_colon": lambda b: f"if {b['n']} > {b['k']}: print({b['n']})",
            "bf_for_colon": lambda b: f"for {b['i']} in range({b['k']}): print({b['i']})",
            "bf_def_colon": lambda b: f"def {b['fn']}({b['a']}): return {b['a']} * {b['k']}",
            "bf_false_typo": lambda b: f"{b['n']} = False",
            "bf_eq_assign": lambda b: f"if {b['n']} == {b['k']}: print({b['n']})",
            "bf_unclosed_paren": lambda b: f"print({b['lit']!r})",
            "bf_concat_int": lambda b: f"print({b['lit']!r} + str({b['k']}))",
            "bf_list_add": lambda b: f"{b['xs']}.append({b['k']})",
            "bf_append_reassign": lambda b: f"{b['xs']}.append({b['k']})",
            "bf_len_typo": lambda b: f"print(len({b['xs']}))",
            "bf_print_typo": lambda b: f"print({b['lit']!r})",
        }
        self.assertEqual(set(F), set(pc.FAMILY_NAMES))
        checked = 0
        for rec in self.train_full:
            expected = F[rec["task_family"]](rec["_bindings"])
            self.assertEqual(
                ast.dump(ast.parse(rec["answer"])),
                ast.dump(ast.parse(expected)),
                (rec["task_family"], rec["prompt"], rec["answer"], expected))
            checked += 1
        self.assertGreater(checked, 25_000)

    def test_in_place_vs_copy_distinctions(self):
        opposites = [
            ("list_sort_inplace", "sorted("),
            ("list_sorted_copy", ".sort()"),
            ("list_reverse_inplace", "[::-1]"),
            ("list_reverse_slice", ".reverse()"),
        ]
        for family, forbidden in opposites:
            recs = [r for r in self.train_full if r["task_family"] == family]
            self.assertTrue(recs, family)
            for rec in recs[:50]:
                self.assertNotIn(forbidden, rec["answer"], (family, rec["answer"]))
        # mutating recipes mutate the target directly: method-call recipes
        # never reassign the container; dict_set mutates via subscript
        # assignment, which is exactly one Assign onto the dict
        for family in ("list_sort_inplace", "list_reverse_inplace", "list_append",
                       "dict_pop", "set_add", "set_remove"):
            self.assertTrue(pc.RECIPE_BY_FAMILY[family]["mutates"])
            for r in [x for x in self.train_full if x["task_family"] == family][:20]:
                tree = ast.parse(r["answer"])
                self.assertFalse(any(isinstance(n, ast.Assign) for n in ast.walk(tree)),
                                 r["answer"])
        self.assertTrue(pc.RECIPE_BY_FAMILY["dict_set"]["mutates"])
        for r in [x for x in self.train_full if x["task_family"] == "dict_set"][:20]:
            tree = ast.parse(r["answer"])
            assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)]
            self.assertEqual(len(assigns), 1, r["answer"])
            self.assertIsInstance(assigns[0].targets[0], ast.Subscript, r["answer"])
        for family in ("list_sorted_copy", "list_reverse_slice", "list_concat"):
            self.assertFalse(pc.RECIPE_BY_FAMILY[family]["mutates"])

    # ---------------------------------------------------- contamination

    def test_zero_contamination_vs_frozen_suite(self):
        prompt_sets, exact_prompts, _, eval_ngrams = pp.load_eval_overlap_sets()
        for recs in (self.train, self.dev):
            for r in recs:
                self.assertNotIn(pp.normalize_key(r["prompt"]), exact_prompts)
                self.assertFalse(pp.ngrams(pp.normalize_word_list(r["prompt"])) & eval_ngrams)
                self.assertFalse(pp.ngrams(pp.normalize_word_list(r["answer"])) & eval_ngrams)
                self.assertLess(pp.max_overlap(pp.normalize_words(r["prompt"]), prompt_sets),
                                pp.OVERLAP_THRESHOLD)
        # the shared-filter path really ran (not skipped)
        self.assertEqual(pc.PIPELINE_STATS["train"]["candidates"],
                         sum(1 for _ in self.train_full))

    # -------------------------------------------------------- guards

    def test_refuses_existing_outputs(self):
        with self.assertRaises(SystemExit):
            pc.run(self.out_dir, seed=42)  # dir already populated

    def test_refuses_protected_paths(self):
        for bad in pc.PROTECTED_OUTPUTS:
            with self.assertRaises(SystemExit):
                pc.assert_writable(bad)
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "results", "x.jsonl")
        with self.assertRaises(SystemExit):
            pc.assert_writable(results_dir)

    def test_never_executes_generated_code(self):
        def boom(*a, **k):
            raise AssertionError("generated snippet was executed")
        out_dir = os.path.join(self.tmp, "noexec")
        with mock.patch.object(builtins, "exec", boom), \
             mock.patch.object(builtins, "eval", boom):
            pc.run(out_dir, seed=42)  # must not raise

    # ------------------------------------------------------ structure

    def test_family_balance_and_coverage(self):
        fam_train, fam_dev = {}, {}
        for r in self.train:
            fam_train[r["task_family"]] = fam_train.get(r["task_family"], 0) + 1
        for r in self.dev:
            fam_dev[r["task_family"]] = fam_dev.get(r["task_family"], 0) + 1
        self.assertEqual(set(fam_train), set(pc.FAMILY_NAMES))
        self.assertEqual(set(fam_dev), set(pc.FAMILY_NAMES))
        counts = list(fam_train.values())
        self.assertGreaterEqual(min(counts) / max(counts), 0.5,
                                f"family imbalance: {fam_train}")
        for fam in pc.FAMILY_NAMES:
            self.assertGreaterEqual(fam_dev[fam], 4, fam)

    def test_manifest_consistent(self):
        m = self.manifest
        self.assertEqual(m["train_count"], len(self.train))
        self.assertEqual(m["dev_count"], len(self.dev))
        self.assertIn("EXPERIMENTAL", m["label"])
        fam = m["families"]
        self.assertEqual(sum(f["train_count"] for f in fam.values()), len(self.train))
        self.assertEqual(sum(f["dev_count"] for f in fam.values()), len(self.dev))
        self.assertEqual(m["train_sha256"],
                         pc.sha256_file(os.path.join(self.out_dir,
                                                     "python-curriculum-train.jsonl")))
        self.assertEqual(m["dev_sha256"],
                         pc.sha256_file(os.path.join(self.out_dir,
                                                     "python-curriculum-dev.jsonl")))

    def test_post_filter_parameter_diversity_is_recorded(self):
        for split, kept, raw in (("train", self.train, self.train_full),
                                 ("dev", self.dev, self.dev_full)):
            raw_by_id = {r["id"]: r for r in raw}
            observed = {r["family"]: {slot: set() for slot in r["slots"]}
                        for r in pc.RECIPES}
            for rec in kept:
                for slot, value in raw_by_id[rec["id"]]["_bindings"].items():
                    observed[rec["task_family"]][slot].add(value)
            for family, slots in observed.items():
                counts = {slot: len(values) for slot, values in slots.items()}
                self.assertEqual(self.manifest["families"][family]
                                 ["parameter_values_" + split], counts)
                for slot, count in counts.items():
                    pool = pc.POOLS[pc.RECIPE_BY_FAMILY[family]["slots"][slot]]
                    if len(pool[split]) > 1:
                        self.assertGreater(count, 1, (split, family, slot))

    def test_categories_present(self):
        cats = {r["category"] for r in self.train}
        self.assertEqual(cats, {"python", "bugfix"})

    def test_phrase_synonymy_present(self):
        # the measured weakness: same op, different natural wording
        def blob(fam):
            return " ".join(r["prompt"].lower() for r in self.train
                            if r["task_family"] == fam)
        for word in ("largest", "maximum", "biggest", "highest"):
            self.assertIn(word, blob("list_max"))
        for word in ("reverse", "backwards"):
            self.assertIn(word, blob("list_reverse_slice"))
        for word in ("items", "count", "length"):
            self.assertIn(word, blob("list_len"))


if __name__ == "__main__":
    unittest.main()
