"""Whole-answer audit checks. No generated Python is executed."""
import unittest

from audit_python import audit


class AuditTest(unittest.TestCase):
    def setUp(self):
        self.item = {'id': 'py-001', 'category': 'python', 'prompt': 'Print hi.',
                     'answer': 'print("hi")', 'max_new_tokens': 40}

    def check_output(self, output, stop='eos', **extra):
        row = {'id': 'py-001', 'output': output, 'tokens_generated': 5,
               'stop': stop, **extra}
        return audit([self.item], [row])

    def test_quotes_and_whitespace_do_not_change_ast(self):
        self.assertEqual(self.check_output(" print('hi')\n")['exact_ast_and_eos'], 1)

    def test_correct_first_line_does_not_hide_wrong_extra_statement(self):
        result = self.check_output('print("hi")\nraise ValueError("oops")')
        self.assertEqual(result['complete_python'], 1)
        self.assertEqual(result['exact_reference_ast'], 0)

    def test_parseable_wrong_answer_is_not_correct(self):
        self.assertEqual(self.check_output('print("bye")')['exact_reference_ast'], 0)

    def test_unterminated_or_overbudget_answer_cannot_pass_joint_check(self):
        for stop in ('budget', 'bos', 'cancelled', ''):
            self.assertEqual(self.check_output('print("hi")', stop)['exact_ast_and_eos'], 0)
        self.assertEqual(self.check_output('print("hi")', tokens_generated=41)['exact_ast_and_eos'], 0)

    def test_empty_and_fragment_are_not_complete_programs(self):
        for text in ('', 'for i in range(3):', 'def f():\nreturn 1'):
            self.assertEqual(self.check_output(text)['complete_python'], 0)
        self.item['answer'] = 'except ValueError:'
        self.assertEqual(self.check_output('except ValueError:')['exact_reference_ast'], 0)

    def test_dangerous_looking_code_is_only_parsed(self):
        # If executed this would fail/hang the test. Syntax is not an invitation
        # to run it, and syntax alone earns no correctness pass.
        result = self.check_output('raise RuntimeError("must not run")\nwhile True: pass')
        self.assertEqual(result['complete_python'], 1)
        self.assertEqual(result['exact_reference_ast'], 0)

    def test_missing_duplicate_unknown_outputs_are_rejected(self):
        for rows in ([], [{'id': 'unknown', 'output': ''}],
                     [{'id': 'py-001', 'output': ''}] * 2):
            with self.assertRaises(ValueError):
                audit([self.item], rows)


if __name__ == '__main__':
    unittest.main()
