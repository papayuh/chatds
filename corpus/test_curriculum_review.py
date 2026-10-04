"""Parent review regressions: actual parameter coverage and basic recipe gaps."""
import ast
import unittest
import python_curriculum as pc


class CurriculumReviewTest(unittest.TestCase):
    def test_no_parameter_axis_accidentally_frozen(self):
        for split in ('train','dev'):
            for recipe in pc.RECIPES:
                rows=pc._combos(recipe,split)
                for slot,pool in recipe['slots'].items():
                    possible=set(pc.POOLS[pool][split])
                    if len(possible)>1:
                        self.assertGreater(len({r[slot] for r in rows}),1,
                            f'{split}/{recipe["family"]}/{slot} is frozen')

    def test_foundational_recipes_render_real_literals_and_bindings(self):
        cases=[
            ('print_literal',{'lit':"'blue sky'"},"print('blue sky')"),
            ('print_variable',{'n':'amount'},'print(amount)'),
            ('print_message',{'msg':"'good morning'"},"print('good morning')"),
            ('assign_literal',{'n':'count','k':'-8'},'count = -8'),
            ('str_int_literal',{'lit':"'-8'"},"int('-8')"),
            ('num_str_literal',{'k':'-8'},'str(-8)'),
            ('abs_literal',{'k':'-8'},'abs(-8)'),
            ('str_len_literal',{'lit':"'blue sky'"},"len('blue sky')"),
            ('list_first',{'xs':'records'},'records[0]'),
            ('list_last',{'xs':'records'},'records[-1]'),
            ('list_first_n',{'xs':'records','k':'4'},'records[:4]'),
            ('list_sorted_desc',{'xs':'records'},'sorted(records, reverse=True)'),
        ]
        for family,bindings,expected in cases:
            self.assertIn(family,pc.FAMILY_NAMES)
            actual=pc.RECIPE_BY_FAMILY[family]['answer'](bindings)
            self.assertEqual(ast.dump(ast.parse(actual)),ast.dump(ast.parse(expected)),family)

    def test_parameter_pool_values_are_disjoint_across_roles_too(self):
        values = {'train': set(), 'dev': set()}
        for pool in pc.POOLS.values():
            for split in values:
                for value in pool[split]:
                    # Numeric text and actual numbers share a held-out value;
                    # a quoted '11' is not new if 11 was already a parameter.
                    canonical = str(value)
                    values[split].add(canonical)
        self.assertFalse(values['train'] & values['dev'])

    def test_one_error_comparison_has_its_colon(self):
        recipe=pc.RECIPE_BY_FAMILY['bf_eq_assign']
        self.assertEqual(recipe['prompt_extra']({'n':'count','k':'2'})['bug'],
                         'if count = 2: print(count)')

    def test_ambiguous_references_are_not_taught_as_facts(self):
        text=lambda f:' '.join(pc.RECIPE_BY_FAMILY[f]['cores']+pc.RECIPE_BY_FAMILY[f]['dev_cores'])
        self.assertNotIn('running total',text('list_sum'))
        self.assertNotIn('{k}th element',text('tuple_index'))
        self.assertNotIn('whole-number part',text('floor_div'))
        for core in pc.RECIPE_BY_FAMILY['bf_list_add']['cores']+pc.RECIPE_BY_FAMILY['bf_list_add']['dev_cores']:
            self.assertIn('list',core,'sets have a valid add(); input type must be stated')


if __name__=='__main__':
    unittest.main()
