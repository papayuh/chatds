#!/usr/bin/env python3
"""Versioned, whole-answer AST scoring and failure feedback. NEVER executes answers.

Accepts formatting/quote differences and explicitly authored alternative answers;
not a semantic equivalence checker. Do not compare these counts to old regex scores.
"""
import argparse
import ast
from collections import Counter
import copy
import json
from pathlib import Path

VERSION = 'whole-ast-eos-v2'


def tree(text):
    if not isinstance(text, str) or not text.strip() or len(text) > 8192:
        return None
    text = text.strip()
    # A header-only answer is legitimate for our one-line snippet task. Never
    # discard an actual body or extra statements to force a match.
    if '\n' not in text and text.endswith(':'):
        text += ' pass'
    try:
        return ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return None


def signature(node):
    return ast.dump(node, include_attributes=False) if node is not None else None


def binding_shape(node):
    """Lossy diagnostic ONLY: never used to award correctness."""
    class Erase(ast.NodeTransformer):
        def visit_Name(self, node):
            return ast.copy_location(ast.Name(id='_', ctx=node.ctx), node)

        def visit_Constant(self, node):
            return ast.copy_location(ast.Constant(value=type(node.value).__name__), node)
    return signature(Erase().visit(copy.deepcopy(node)))


def audit(suite, outputs):
    expected = {r['id']: r for r in suite}
    actual = {r['id']: r for r in outputs}
    if len(expected) != len(suite) or len(actual) != len(outputs) or expected.keys() != actual.keys():
        raise ValueError('duplicate, missing or unknown IDs')
    details = []
    for identifier, item in expected.items():
        answers = [item['answer'], *item.get('alternative_answers', [])]
        references = [tree(answer) for answer in answers]
        if any(r is None for r in references):
            raise ValueError(f'invalid reference: {identifier}')
        row = actual[identifier]
        output = row.get('output')
        if not isinstance(output, str):
            raise ValueError(f'invalid output: {identifier}')
        tokens = row.get('tokens_generated')
        valid_count = type(tokens) is int and tokens >= 0
        if not valid_count:
            raise ValueError(f'invalid token count: {identifier}')
        parsed = tree(output)
        matches = parsed is not None and signature(parsed) in [signature(r) for r in references]
        if row.get('stop') != 'eos' or tokens > item['max_new_tokens']:
            reason = 'termination_or_budget'
        elif parsed is None:
            reason = 'syntax'
        elif matches:
            reason = 'correct'
        elif any(binding_shape(parsed) == binding_shape(r) for r in references):
            reason = 'possible_binding_error'
        else:
            reason = 'operation_or_structure'
        details.append({'id': identifier, 'category': item['category'], 'tier': item['tier'],
                        'prompt': item['prompt'], 'reference': item['answer'], 'output': output,
                        'reason': reason, 'correct': reason == 'correct',
                        'tokens_generated': tokens, 'stop': row.get('stop')})
    return {'version': VERSION, 'correct': sum(d['correct'] for d in details), 'total': len(details),
            'failure_counts': dict(Counter(d['reason'] for d in details)),
            'per_category': {c: {'correct': sum(d['correct'] for d in details if d['category'] == c),
                                 'n': sum(d['category'] == c for d in details)}
                             for c in sorted({d['category'] for d in details})},
            'per_tier': {str(t): {'correct': sum(d['correct'] for d in details if d['tier'] == t),
                                  'n': sum(d['tier'] == t for d in details)}
                         for t in sorted({d['tier'] for d in details})},
            'note': 'Parse-only; AST mismatch may be valid alternative. Binding labels are heuristic.',
            'answers': details}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--suite', required=True, type=Path)
    ap.add_argument('--outputs', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    load = lambda p: [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    report = audit(load(args.suite), load(args.outputs))
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'answers'}, indent=2))
