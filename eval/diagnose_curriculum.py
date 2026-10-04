#!/usr/bin/env python3
"""Split curriculum failures into binding vs operation/structure mistakes.

Parse only: generated code is NEVER imported, executed, eval'd or literal_eval'd.
A 'binding' verdict means the answer had the reference's exact operations and
control structure but differed at a position the generator itself treated as a
parameter (an identifier or literal from --bindings). A 'operation_or_structure'
verdict means the difference was somewhere else -- that is NOT proof the model
picked the wrong operation, only that the difference is not a binding slip.
Nothing here certifies semantics: AST similarity is not correctness, and the
relaxed compare is coarse (it ignores identifier ORDER and scope identity, so
two answers that swap two relaxable names still compare equal).

Usage: python3 eval/diagnose_curriculum.py --suite SUITE.jsonl \\
           --outputs RUN.jsonl --bindings BINDINGS.jsonl [--out REPORT.json]
Exit 0 means a complete/well-formed diagnostic, NOT a good model.
"""
import argparse
import ast
import json
from collections import Counter, defaultdict
from pathlib import Path

import score

CLASSES = ('exact', 'binding', 'operation_or_structure', 'syntax', 'termination_budget')

NOTE = 'parse-only heuristic; no code executed; not semantic certification'

# Operation names are never treated as bindings even if a generator listed them
# as parameters: max must never normalise into min, nor .sort into .reverse.
NEVER_RELAX = frozenset((
    'abs', 'all', 'any', 'bool', 'dict', 'divmod', 'enumerate', 'filter', 'float',
    'format', 'int', 'len', 'list', 'map', 'max', 'min', 'print', 'range', 'repr',
    'reversed', 'round', 'set', 'sorted', 'str', 'sum', 'tuple', 'zip',
))

# ---------------------------------------------------------------- parsing

def parsed(text):
    """Parse whole text; None unless it is a complete, non-empty Python module."""
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        tree = ast.parse(text.strip())
    except (SyntaxError, ValueError, RecursionError):
        return None
    return tree if tree.body else None


def dumped(tree):
    try:
        return ast.dump(tree, include_attributes=False)
    except RecursionError:
        return None


# ---------------------------------------------------------------- loading

def load_suite(rows):
    """Schema-check the curriculum suite. A malformed suite is fatal: an
    unparseable reference answer would make every verdict meaningless."""
    items, seen, bad = [], set(), []
    for n, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            bad.append('suite line %d: not a JSON object' % n); continue
        for key in ('id', 'answer', 'task_family', 'category'):
            if not isinstance(row.get(key), str) or not row[key].strip():
                bad.append('suite line %d: missing/non-string %r' % (n, key))
        mnt = row.get('max_new_tokens')
        if isinstance(mnt, bool) or not isinstance(mnt, int) or mnt <= 0:
            bad.append('suite line %d: max_new_tokens must be a positive int, got %r' % (n, mnt))
        identifier = row.get('id')
        if isinstance(identifier, str):
            if identifier in seen:
                bad.append('suite line %d: duplicate id %r' % (n, identifier))
            seen.add(identifier)
        if isinstance(row.get('answer'), str) and parsed(row['answer']) is None:
            bad.append('suite line %d (%s): reference answer is not complete Python'
                       % (n, row.get('id')))
        items.append(row)
    if not items:
        bad.append('suite is empty')
    if bad:
        raise ValueError('; '.join(bad))
    return items


def load_rows(rows, known_ids):
    """Check IDs structurally before reusing score's counter sanitation.
    Never classify errors by matching words inside their message: an unknown
    ID can itself contain 'tokens_generated' or any other diagnostic text.
    """
    rows = list(rows)
    seen = set()
    for n, row in enumerate(rows, 1):
        if not isinstance(row, dict) or not isinstance(row.get('id'), str):
            raise ValueError('outputs line %d: missing/non-string id' % n)
        identifier = row['id']
        if identifier not in known_ids:
            raise ValueError('outputs line %d: unknown id %r' % (n, identifier))
        if identifier in seen:
            raise ValueError('outputs line %d: duplicate id %r' % (n, identifier))
        seen.add(identifier)
    missing = known_ids - seen
    if missing:
        raise ValueError('missing outputs: %s' % sorted(missing))
    # Remaining violations describe row data, not coverage. Invalid counters
    # are stripped by the shared validator and fail termination/budget checks;
    # non-string output fails the whole-answer syntax check.
    return score.load_outputs((json.dumps(row) for row in rows), known_ids)


def load_bindings(rows, known_ids):
    """Trusted generator parameters: {id, identifiers: [str], literals: [scalar]}."""
    by_id, bad = {}, []
    for n, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            bad.append('bindings line %d: not a JSON object' % n); continue
        bid = row.get('id')
        if not isinstance(bid, str):
            bad.append('bindings line %d: missing/non-string id' % n); continue
        if bid not in known_ids:
            bad.append('bindings line %d: unknown id %r' % (n, bid)); continue
        if bid in by_id:
            bad.append('bindings line %d: duplicate id %r' % (n, bid)); continue
        idents, lits = row.get('identifiers', []), row.get('literals', [])
        if not isinstance(idents, list) or not all(isinstance(i, str) for i in idents):
            bad.append('bindings line %d (%s): identifiers must be a list of strings' % (n, bid))
            idents = []
        if not isinstance(lits, list) or not all(
                l is None or isinstance(l, (str, int, float, bool)) for l in lits):
            bad.append('bindings line %d (%s): literals must be a list of JSON scalars' % (n, bid))
            lits = []
        by_id[bid] = {'identifiers': frozenset(idents) - NEVER_RELAX, 'literals': list(lits)}
    missing = known_ids - by_id.keys()
    if missing:
        bad.append('missing bindings: %s' % sorted(missing))
    if bad:
        raise ValueError('; '.join(bad))
    return by_id


# ---------------------------------------------------------------- binding compare

def _same_scalar(left, right):
    """Type-strict: True must never pass for 1, nor 2 for 2.0."""
    return type(left) is type(right) and left == right


def _relaxable_literal(value, binding):
    return any(_same_scalar(value, lit) for lit in binding['literals'])


def _number(node):
    """(is_number, value) for a scalar number, including -1 as UnaryOp(USub)."""
    if isinstance(node, ast.Constant) and not isinstance(node.value, bool) \
            and isinstance(node.value, (int, float, complex)):
        return True, node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        ok, value = _number(node.operand)
        if ok:
            return True, (-value if isinstance(node.op, ast.USub) else value)
    return False, None


def _binding_number(exp, got, binding):
    """A parameter number may change value or sign; a fixed one may not.
    xs[::-1] vs xs[::-2] stays operation_or_structure with no literal parameter."""
    ok, value = _number(exp)
    return ok and _relaxable_literal(value, binding) and _number(got)[0]


def _binding_unquoted(exp, got, binding):
    """print(hi) for print("hi"): a dropped quote around a parameter literal."""
    return (isinstance(exp, ast.Constant) and isinstance(exp.value, str)
            and _relaxable_literal(exp.value, binding) and isinstance(got, ast.Name))


def _binding_identifier(expected, actual, binding, relax=True):
    """Equal names always pass; a different name passes only when the expected
    one is a generator parameter (and never in a call's function position)."""
    if _same_scalar(expected, actual):
        return True
    return (relax and isinstance(expected, str) and isinstance(actual, str)
            and expected in binding['identifiers'])


def relaxed_equal(exp, got, binding, func_pos=False):
    """Equal when the only differences sit at expected parameter positions.
    Operators, control flow, statement count, attribute/keyword names, Load/Store
    contexts and fixed constants must all match exactly."""
    if isinstance(exp, list) or isinstance(got, list):
        return (isinstance(exp, list) and isinstance(got, list)
                and len(exp) == len(got)
                and all(relaxed_equal(e, g, binding) for e, g in zip(exp, got)))
    if not isinstance(exp, ast.AST) or not isinstance(got, ast.AST):
        return not isinstance(exp, ast.AST) and not isinstance(got, ast.AST) \
            and _same_scalar(exp, got)
    if _binding_number(exp, got, binding) or _binding_unquoted(exp, got, binding):
        return True
    if type(exp) is not type(got):
        return False
    if isinstance(exp, ast.Constant):
        return _relaxable_literal(exp.value, binding) or (
            _same_scalar(exp.value, got.value) and _same_scalar(exp.kind, got.kind))
    if isinstance(exp, ast.Name):
        return type(exp.ctx) is type(got.ctx) \
            and _binding_identifier(exp.id, got.id, binding, relax=not func_pos)
    defines_name = isinstance(exp, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    for field in exp._fields:
        left, right = getattr(exp, field, None), getattr(got, field, None)
        # Definition and argument names are relaxable; Attribute.attr, keyword.arg
        # and import aliases fall through to the strict scalar compare below.
        if (field == 'name' and defines_name) or (field == 'arg' and isinstance(exp, ast.arg)):
            if not _binding_identifier(left, right, binding):
                return False
            continue
        if not relaxed_equal(left, right, binding,
                             func_pos=field == 'func' and isinstance(exp, ast.Call)):
            return False
    return True


# ---------------------------------------------------------------- classification

def classify(item, row, binding):
    """Exclusive priority: termination/budget, syntax, exact, binding, else."""
    tokens, mnt = row.get('tokens_generated'), item.get('max_new_tokens')
    budget_ok = (isinstance(tokens, int) and not isinstance(tokens, bool)
                 and tokens >= 0 and (mnt is None or tokens <= mnt))
    if row.get('stop') != 'eos' or not budget_ok:
        return 'termination_budget'
    actual, expected = parsed(row.get('output', '')), parsed(item['answer'])
    if actual is None or expected is None:
        return 'syntax'
    got, exp = dumped(actual), dumped(expected)
    if exp is not None and got == exp:
        return 'exact'
    try:
        if relaxed_equal(expected, actual, binding):
            return 'binding'
    except RecursionError:
        pass
    return 'operation_or_structure'


def diagnose(suite, output_rows, binding_rows):
    """Return the report dict. Raises ValueError for any coverage problem."""
    suite = load_suite(suite)
    known = {item['id'] for item in suite}
    by_id, violations = load_rows(output_rows, known)
    bindings = load_bindings(binding_rows, known)
    rows, families = [], defaultdict(lambda: defaultdict(Counter))
    for item in suite:
        row = by_id[item['id']]
        verdict = classify(item, row, bindings[item['id']])
        families[item['task_family']][item['category']][verdict] += 1
        rows.append({
            'id': item['id'], 'task_family': item['task_family'],
            'category': item['category'], 'prompt': item.get('prompt', ''),
            'reference': item['answer'],
            'output': row.get('output', '') if isinstance(row.get('output'), str) else '',
            'stop': row.get('stop'), 'tokens_generated': row.get('tokens_generated'),
            'classification': verdict,
        })
    totals = Counter(r['classification'] for r in rows)
    return {
        'total': len(rows),
        'counts': {name: totals.get(name, 0) for name in CLASSES},
        'families': {
            family: {
                'total': sum(sum(c.values()) for c in cats.values()),
                'counts': {name: sum(c.get(name, 0) for c in cats.values())
                           for name in CLASSES},
                'categories': {cat: {name: counts.get(name, 0) for name in CLASSES}
                               for cat, counts in sorted(cats.items())},
            } for family, cats in sorted(families.items())},
        'data_violations': violations,
        'note': NOTE,
        'rows': rows,
    }


def read_jsonl(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--suite', required=True)
    ap.add_argument('--outputs', required=True)
    ap.add_argument('--bindings', required=True)
    ap.add_argument('--out')
    args = ap.parse_args()
    try:
        report = diagnose(read_jsonl(args.suite), read_jsonl(args.outputs),
                          read_jsonl(args.bindings))
    except ValueError as error:
        ap.error(str(error))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'rows'},
                     indent=2))


if __name__ == '__main__':
    main()
