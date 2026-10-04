"""Parent regressions: malformed audit inputs cannot buy coverage/budget passes."""
import unittest

from diagnose_curriculum import diagnose


class DiagnosticValidationTest(unittest.TestCase):
    def fixture(self, identifier='sample'):
        suite = [{'id': identifier, 'answer': 'print(x)', 'category': 'python',
                  'task_family': 'print', 'max_new_tokens': 32}]
        output = [{'id': identifier, 'output': 'print(x)', 'stop': 'eos',
                   'tokens_generated': 4}]
        bindings = [{'id': identifier, 'identifiers': ['x'], 'literals': []}]
        return suite, output, bindings

    def test_error_word_in_id_cannot_hide_unknown_or_duplicate(self):
        for identifier in ('tokens_generated', 'bytes_read', 'output is not a string'):
            suite, rows, bindings = self.fixture()
            with self.assertRaises(ValueError):
                diagnose(suite, rows + [{**rows[0], 'id': identifier}], bindings)
            suite, rows, bindings = self.fixture(identifier)
            with self.assertRaises(ValueError):
                diagnose(suite, rows * 2, bindings)

    def test_malformed_suite_ids_raise_the_documented_error(self):
        for identifier in ([], {}, 3, None):
            suite, rows, bindings = self.fixture()
            suite[0]['id'] = identifier
            with self.assertRaises(ValueError):
                diagnose(suite, rows, bindings)

    def test_audit_requires_a_real_positive_budget(self):
        for budget in (None, True, 0, -1, '32'):
            suite, rows, bindings = self.fixture()
            suite[0]['max_new_tokens'] = budget
            with self.assertRaises(ValueError):
                diagnose(suite, rows, bindings)
        suite, rows, bindings = self.fixture()
        del suite[0]['max_new_tokens']
        with self.assertRaises(ValueError):
            diagnose(suite, rows, bindings)

    def test_empty_suite_is_not_a_completed_diagnostic(self):
        with self.assertRaises(ValueError):
            diagnose([], [], [])


if __name__ == '__main__':
    unittest.main()
