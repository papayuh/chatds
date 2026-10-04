#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Narrow product parity policy; the frozen golden suite stays strict/unmodified."""
import json
from pathlib import Path

import engine_golden as golden


def load_exceptions(path, fixtures_path):
    if path is None:
        return {}
    data = json.loads(Path(path).read_text())
    if data.get('version') != 1 or data.get('fixture_sha256') != golden.sha256(fixtures_path):
        raise ValueError('known-divergence policy does not match these frozen fixtures')
    cases = {c['name']: c for c in json.loads(Path(fixtures_path).read_text())['cases']}
    exceptions = data['cases']
    for name, actual in exceptions.items():
        if name not in cases or cases[name].get('class'):
            raise ValueError('exception must name a successful-input fixture')
        golden.validate_actual(actual, cases[name])
    return exceptions


def verdict(case, actual, exceptions=None):
    """Only the explicitly recorded output is accepted; no blanket prompt exclusion."""
    golden.validate_actual(actual, case)
    if case.get('class') == 'baseline-fails':
        return 'safety-pass'
    if not golden.compare(case['expected'], actual):
        return 'exact'
    known = (exceptions or {}).get(case['name'])
    if known is not None and not golden.compare(known, actual):
        return 'known-divergence'
    return 'FAIL'
