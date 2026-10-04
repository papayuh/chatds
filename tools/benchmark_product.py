#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure prebuilt ROMs against frozen goldens, always headless.

Uses engine_golden's isolated boot/cleanup boundary. Historical comparison
measurements are retained as observations, not an executable legacy lane.
"""
import argparse
import json
from pathlib import Path
import subprocess

import engine_golden as golden
from product_parity import load_exceptions, verdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, default=golden.DEFAULT_FIXTURES)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--kb', type=Path, required=True)
    parser.add_argument('--rom', type=Path, required=True)
    parser.add_argument('--known-divergences', type=Path)
    parser.add_argument('--card-dir', choices=('chatds',), default='chatds')
    parser.add_argument('--cases', nargs='*')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=120)
    args = parser.parse_args()
    fixture = json.loads(args.fixtures.read_text())
    golden.validate_fixture_set(fixture)
    known = load_exceptions(args.known_divergences, args.fixtures)
    card_dir = args.card_dir
    names = {c['name'] for c in fixture['cases']}
    if args.cases and set(args.cases) - names:
        parser.error('unknown case name')
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('--out must be new or empty; preserve existing evidence')
    for name, path in [('model', args.model), ('tokenizer', args.tokenizer), ('kb', args.kb)]:
        if golden.sha256(path) != fixture['provenance'][name + '_sha256']:
            parser.error(name + ' does not match the frozen fixture')
    args.out.mkdir(parents=True, exist_ok=True)
    report = {'rom_sha256': golden.sha256(args.rom), 'baseline': False,
              'fixture_sha256': golden.sha256(args.fixtures),
              'assets': {n: golden.sha256(p) for n, p in [('model', args.model),
                         ('tokenizer', args.tokenizer), ('kb', args.kb)]},
              'clock_hz': 33513982, 'card_dir': card_dir,
              'known_divergences_sha256': golden.sha256(args.known_divergences) if args.known_divergences else None,
              'cases': []}
    for case in fixture['cases']:
        if args.cases and case['name'] not in args.cases:
            continue
        work = args.out / case['name']
        actual = golden.run_rom(args.rom, args.model, args.tokenizer, args.kb,
                                case, work, args.timeout, card_dir=card_dir)
        result = verdict(case, actual, known)
        row = {'name': case['name'], 'actual': actual, 'verdict': result,
               'parity': result in ('exact', 'safety-pass'), 'accepted': result != 'FAIL'}
        if not case.get('class'):
            row['mismatches'] = golden.compare(case['expected'], actual)
        if actual.get('status') != 'FAIL':
            subprocess.run(['mcopy', '-o', '-i', str(work / 'sd.img'),
                            f'::/{card_dir}/heap.txt', str(work / 'heap.txt')], check=True, timeout=10,
                           capture_output=True)
            heap = dict(line.split('=', 1) for line in (work / 'heap.txt').read_text().splitlines()
                        if '=' in line)
            row['heap'] = heap
            row['total_ms'] = int(heap['total_ticks']) * 1000 / 33513982
            first_ticks = int(heap['first_piece_ticks']) or int(heap['first_logit_ticks'])
            row['first_ms'] = first_ticks * 1000 / 33513982
            row['first_kind'] = 'first streamed piece (or first logit if no piece)'
            row['first_logit_ms'] = int(heap['first_logit_ticks']) * 1000 / 33513982
            row['ram_break_bytes'] = int(heap['ram_break_bytes'])
            count = len(actual['ids'])
            # Post-first forwards include EOS for EOS-stopped runs; budget runs
            # perform count-1. State the denominator rather than hide prefill.
            forwards = count if actual['stop'] in ('eos', 'bos') else max(0, count - 1)
            decode_ms = row['total_ms'] - row['first_ms']
            row['decode_steps_s'] = forwards * 1000 / decode_ms if forwards and decode_ms > 0 else None
        report['cases'].append(row)
        golden.write_json(args.out / 'measurements.json', report)
        print(json.dumps(row), flush=True)
        # Keep small raw evidence; each FAT image is ~80 MB and no longer needed.
        (work / 'sd.img').unlink(missing_ok=True)
        (work / 'one/sd.img').unlink(missing_ok=True)
    return 0 if report['cases'] and all(c['accepted'] for c in report['cases']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
