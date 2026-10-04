#!/usr/bin/env python3
"""Audit WHOLE Python answers, not only suite-v1.1's first-line regex score.

Parse only: generated code is NEVER executed. Syntax success is not semantic
correctness. Exact-reference AST equality is a conservative diagnostic: it
rejects extra statements but also rejects valid alternative implementations.
Reference fragments cannot get an exact-AST pass. This is not Diego's personal
acceptance set, nor a substitute for reviewing generated answers.

Usage: python3 eval/audit_python.py --outputs RUN.jsonl [--out REPORT.json]
Exit 0 means a complete/well-formed audit, NOT a useful or shippable model.
"""
import argparse
import ast
import json
from pathlib import Path

import score


def parsed_ast(text):
    if not text.strip():
        return None
    try:
        return ast.dump(ast.parse(text.strip()), include_attributes=False)
    except (SyntaxError, ValueError, RecursionError):
        return None


def audit(suite, rows):
    by_id, violations = score.load_outputs(
        (json.dumps(row) for row in rows), {item['id'] for item in suite})
    if violations:
        raise ValueError('; '.join(violations))
    missing = {item['id'] for item in suite} - by_id.keys()
    if missing:
        raise ValueError(f'missing outputs: {sorted(missing)}')
    details = []
    for item in suite:
        if item['category'] not in ('python', 'bugfix'):
            continue
        row = by_id[item['id']]
        output = row.get('output', '')
        actual = parsed_ast(output)
        expected = parsed_ast(item['answer'])
        tokens = row.get('tokens_generated')
        budget_ok = (isinstance(tokens, int) and not isinstance(tokens, bool)
                     and 0 <= tokens <= item['max_new_tokens'])
        eos = row.get('stop') == 'eos'
        details.append({
            'id': item['id'], 'prompt': item['prompt'],
            'reference': item['answer'], 'output': output,
            'stop': row.get('stop'), 'tokens_generated': tokens,
            'within_budget': budget_ok, 'eos': eos,
            'complete_python': actual is not None,
            'reference_complete_python': expected is not None,
            'exact_reference_ast': expected is not None and actual == expected,
            'exact_ast_and_eos': expected is not None and actual == expected and eos and budget_ok,
        })
    keys = ('within_budget', 'eos', 'complete_python', 'reference_complete_python',
            'exact_reference_ast', 'exact_ast_and_eos')
    return {'total': len(details),
            **{key: sum(row[key] for row in details) for key in keys},
            'note': 'Parse-only diagnostic; no code execution, no semantic/usefulness certification.',
            'answers': details}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--suite', default=str(Path(__file__).with_name('suite-v1.1.jsonl')))
    ap.add_argument('--outputs', required=True)
    ap.add_argument('--out')
    args = ap.parse_args()
    suite = score.load_suite(args.suite)
    with open(args.outputs) as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    try:
        report = audit(suite, rows)
    except ValueError as error:
        ap.error(str(error))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'answers'}, indent=2))


if __name__ == '__main__':
    main()
