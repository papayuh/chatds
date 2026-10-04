"""Binding vs operation/structure diagnostic. No generated Python is executed."""
import unittest

from diagnose_curriculum import NOTE, diagnose


def suite_item(answer, **extra):
    return {'id': 'py-001', 'category': 'python', 'task_family': 'list_ops',
            'prompt': 'Do the thing.', 'answer': answer, 'max_new_tokens': 40, **extra}


class DiagnoseTest(unittest.TestCase):
    def verdict(self, answer, output, identifiers=(), literals=(), stop='eos', **extra):
        row = {'id': 'py-001', 'output': output, 'stop': stop,
               'tokens_generated': 12, **extra}
        binding = {'id': 'py-001', 'identifiers': list(identifiers),
                   'literals': list(literals)}
        report = diagnose([suite_item(answer)], [row], [binding])
        self.assertEqual(report['total'], 1)
        return report['rows'][0]['classification']

    # ------------------------------------------------------------ exact
    def test_quote_style_and_whitespace_are_exact(self):
        self.assertEqual(self.verdict('print("hi")', " print('hi')\n",
                                      literals=['hi']), 'exact')

    def test_exact_wins_over_binding_when_bindings_are_relaxable(self):
        self.assertEqual(self.verdict('xs = [1]', 'xs = [1]',
                                      identifiers=['xs'], literals=[1]), 'exact')

    # ------------------------------------------------------------ binding
    def test_wrong_parameter_identifier_is_binding(self):
        self.assertEqual(self.verdict('print(len(items))', 'print(len(values))',
                                      identifiers=['items']), 'binding')

    def test_identifier_outside_bindings_is_not_binding(self):
        self.assertEqual(self.verdict('print(len(items))', 'print(len(values))'),
                         'operation_or_structure')

    def test_function_and_argument_names_are_relaxable(self):
        self.assertEqual(self.verdict('def total(xs):\n    return sum(xs)',
                                      'def add_up(ys):\n    return sum(ys)',
                                      identifiers=['total', 'xs']), 'binding')

    def test_wrong_parameter_literal_and_sign_change_are_binding(self):
        self.assertEqual(self.verdict('print(x + 7)', 'print(x + 9)', literals=[7]),
                         'binding')
        self.assertEqual(self.verdict('print(x + -7)', 'print(x + 7)', literals=[-7]),
                         'binding')
        self.assertEqual(self.verdict('print(x - 7)', 'print(x - -7)', literals=[7]),
                         'binding')

    def test_missing_quotes_around_parameter_literal_is_binding(self):
        self.assertEqual(self.verdict('print("hi")', 'print(hi)', literals=['hi']),
                         'binding')

    def test_missing_quotes_without_a_literal_parameter_is_not_binding(self):
        self.assertEqual(self.verdict('print("hi")', 'print(hi)'),
                         'operation_or_structure')

    # ------------------------------------------------------------ operation/structure
    def test_max_never_normalises_into_min_even_if_listed(self):
        self.assertEqual(self.verdict('print(max(xs))', 'print(min(xs))',
                                      identifiers=['max', 'min', 'xs']),
                         'operation_or_structure')

    def test_method_attribute_is_never_relaxed(self):
        self.assertEqual(self.verdict('xs.sort()', 'xs.reverse()',
                                      identifiers=['xs', 'sort', 'reverse']),
                         'operation_or_structure')

    def test_fixed_stride_constant_is_not_a_binding(self):
        self.assertEqual(self.verdict('print(xs[::-1])', 'print(xs[::-2])',
                                      identifiers=['xs']), 'operation_or_structure')

    def test_fixed_modulus_of_an_even_check_is_not_a_binding(self):
        self.assertEqual(self.verdict('if n % 2 == 0:\n    print(n)',
                                      'if n % 3 == 0:\n    print(n)',
                                      identifiers=['n']), 'operation_or_structure')

    def test_operator_and_loop_body_changes_are_structural(self):
        self.assertEqual(self.verdict('print(a + b)', 'print(a - b)',
                                      identifiers=['a', 'b']), 'operation_or_structure')
        self.assertEqual(self.verdict('for x in xs:\n    total += x',
                                      'for x in xs:\n    total += 1',
                                      identifiers=['x', 'xs', 'total']),
                         'operation_or_structure')
        self.assertEqual(self.verdict('for x in xs:\n    print(x)',
                                      'while x in xs:\n    print(x)',
                                      identifiers=['x', 'xs']), 'operation_or_structure')

    def test_extra_statements_never_pass_exact_or_binding(self):
        for output in ('print(x)\nprint(x)', 'print(y)\nraise ValueError("oops")'):
            self.assertEqual(self.verdict('print(x)', output, identifiers=['x', 'y']),
                             'operation_or_structure')

    def test_true_is_not_the_literal_one(self):
        self.assertEqual(self.verdict('print(1)', 'print(True)'),
                         'operation_or_structure')

    # ------------------------------------------------------------ syntax
    def test_unparseable_or_empty_or_fenced_output_is_syntax(self):
        for output in ('', '   ', 'def f():\nreturn 1', 'for i in range(3):',
                       '```python\nprint(x)\n```', 'Here is the answer: print(x)'):
            self.assertEqual(self.verdict('print(x)', output), 'syntax')

    def test_dangerous_looking_code_is_only_parsed(self):
        # Parsed, never run: this would hang or fail the suite if executed.
        self.assertEqual(self.verdict('print(x)',
                                      'while True: pass\nraise SystemExit(1)',
                                      identifiers=['x']), 'operation_or_structure')

    def test_unparseable_reference_refuses_the_suite(self):
        with self.assertRaises(ValueError):
            diagnose([suite_item('def f():\nreturn 1')],
                     [{'id': 'py-001', 'output': 'print(1)', 'stop': 'eos',
                       'tokens_generated': 3}],
                     [{'id': 'py-001', 'identifiers': [], 'literals': []}])

    # ------------------------------------------------------------ termination/budget
    def test_non_eos_stop_outranks_a_perfect_answer(self):
        for stop in ('budget', 'cancelled', '', None):
            self.assertEqual(self.verdict('print(x)', 'print(x)', stop=stop),
                             'termination_budget')

    def test_bad_generated_count_is_termination_budget(self):
        for tokens in (41, -1, True, 'many', None):
            self.assertEqual(self.verdict('print(x)', 'print(x)',
                                          tokens_generated=tokens),
                             'termination_budget')
        self.assertEqual(self.verdict('print(x)', 'print(x)', tokens_generated=40),
                         'exact')

    # ------------------------------------------------------------ coverage refusals
    def test_missing_duplicate_and_unknown_output_ids_are_refused(self):
        good_binding = [{'id': 'py-001', 'identifiers': [], 'literals': []}]
        row = {'id': 'py-001', 'output': 'print(x)', 'stop': 'eos', 'tokens_generated': 3}
        for rows in ([], [row, dict(row)], [{**row, 'id': 'py-999'}], [row, 'nonsense'],
                     [{'output': 'print(x)'}]):
            with self.assertRaises(ValueError):
                diagnose([suite_item('print(x)')], rows, good_binding)

    def test_missing_duplicate_unknown_and_malformed_bindings_are_refused(self):
        row = [{'id': 'py-001', 'output': 'print(x)', 'stop': 'eos', 'tokens_generated': 3}]
        binding = {'id': 'py-001', 'identifiers': [], 'literals': []}
        for bindings in ([], [binding, dict(binding)],
                         [{**binding, 'id': 'py-999'}, binding],
                         [{**binding, 'identifiers': 'xs'}],
                         [{**binding, 'literals': [{'not': 'scalar'}]}]):
            with self.assertRaises(ValueError):
                diagnose([suite_item('print(x)')], row, bindings)

    def test_duplicate_suite_ids_are_refused(self):
        with self.assertRaises(ValueError):
            diagnose([suite_item('print(x)'), suite_item('print(y)')],
                     [{'id': 'py-001', 'output': 'print(x)', 'stop': 'eos',
                       'tokens_generated': 3}],
                     [{'id': 'py-001', 'identifiers': [], 'literals': []}])

    # ------------------------------------------------------------ report shape
    def test_report_carries_totals_family_counts_rows_and_the_note(self):
        suite = [suite_item('print(x)'),
                 suite_item('xs.sort()', id='py-002', category='bugfix',
                            task_family='sorting')]
        rows = [{'id': 'py-001', 'output': 'print(y)', 'stop': 'eos', 'tokens_generated': 3},
                {'id': 'py-002', 'output': 'xs.reverse()', 'stop': 'eos', 'tokens_generated': 4}]
        bindings = [{'id': 'py-001', 'identifiers': ['x'], 'literals': []},
                    {'id': 'py-002', 'identifiers': ['xs'], 'literals': []}]
        report = diagnose(suite, rows, bindings)
        self.assertEqual(report['total'], 2)
        self.assertEqual(report['counts']['binding'], 1)
        self.assertEqual(report['counts']['operation_or_structure'], 1)
        self.assertEqual(report['families']['sorting']['categories']['bugfix']
                         ['operation_or_structure'], 1)
        self.assertEqual(report['families']['list_ops']['total'], 1)
        self.assertEqual(report['note'], NOTE)
        first = report['rows'][0]
        for key in ('id', 'prompt', 'reference', 'output', 'classification'):
            self.assertIn(key, first)
        self.assertEqual(first['reference'], 'print(x)')
        self.assertEqual(first['output'], 'print(y)')


if __name__ == '__main__':
    unittest.main()
