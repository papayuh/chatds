#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Capture answer-only results through the independent clean host product.

Build with `make host`. Requires DSQ8 model and matching tokenizer; KB optional.
This is host quality evaluation, not DS performance measurement.
"""
import argparse
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
import engine_golden as golden
import score


def run_capture(args, suite):
    with args.model.open('rb') as f:
        vocab = struct.unpack_from('<I', f.read(40), 36)[0]
    rows = []
    with tempfile.TemporaryDirectory(prefix='chatds-eval-') as td:
        work = Path(td)
        run = work / 'run.txt'
        for item in suite:
            question, steps = item['prompt'], item['max_new_tokens']
            if (not question.isascii() or len(question) > 127
                    or any(c in question for c in '\n\r\0') or not 1 <= steps <= 256):
                raise ValueError('item is outside the product input limits: ' + item['id'])
            # No expected answer or oracle context reaches the product.
            run.write_text(f'steps={steps}\nkbd=0\ninstruct=1\nprompt={question}\n')
            subprocess.run([str(args.host), str(args.model), str(args.tokenizer),
                            str(args.chatds_kb) if args.chatds_kb else '-',
                            str(run), str(work) + '/'], check=True, timeout=args.timeout)
            result = golden.parse_result((work / 'out.txt').read_text(),
                                         (work / 'ids.txt').read_text(),
                                         (work / 'ctx.txt').read_text(),
                                         vocab_size=vocab, allow_context=True)
            raw = result['text']
            # out.txt preserves raw generated text. The independently evaluated
            # display/tool result is published separately, even when it is empty.
            calc = (work / 'calc.txt').read_text()
            if not calc.startswith('calc=') or not calc.endswith('\n'):
                raise ValueError('malformed calculator publication: ' + item['id'])
            display = calc[5:-1] or raw
            rows.append({'id': item['id'], 'output': display, 'raw_output': raw,
                         'tokens_generated': len(result['ids']), 'stop': result['stop']})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--suite', type=Path, default=ROOT / 'eval/suite-v1.1.jsonl')
    ap.add_argument('--host', type=Path, default=ROOT / 'clean/product/build/chatds-host')
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--chatds-kb', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--limit', type=int, help='development subset only, not a full-suite result')
    ap.add_argument('--timeout', type=float, default=120)
    args = ap.parse_args()
    if args.timeout <= 0 or not math.isfinite(args.timeout):
        ap.error('--timeout must be finite and positive')
    if args.limit is not None and args.limit <= 0:
        ap.error('--limit must be positive')
    for name in ('host', 'model', 'tokenizer', 'chatds_kb'):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, path.resolve(strict=True))
    suite = score.load_suite(args.suite)
    if args.limit:
        print('DEVELOPMENT SUBSET: not a full-suite quality result', file=sys.stderr)
        suite = suite[:args.limit]
    rows = run_capture(args, suite)
    lines = [json.dumps(row) + '\n' for row in rows]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(''.join(lines))
    ids = {item['id'] for item in suite}
    by_id, violations = score.load_outputs(lines, ids)
    score.report(suite, by_id, ids, violations=violations)
    return int(bool(violations))


if __name__ == '__main__':
    raise SystemExit(main())
