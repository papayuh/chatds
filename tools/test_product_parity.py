"""Product exceptions are exact, fixture-pinned records, never skipped prompts."""
import copy
import json
from pathlib import Path

import pytest

from product_parity import load_exceptions, verdict

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / 'ds/engine-golden/fixtures.json'
POLICY = ROOT / 'ds/engine-golden/clean-known-divergences.json'
CASES = {c['name']: c for c in json.loads(FIXTURES.read_text())['cases']}


def test_strict_default_still_rejects_both_divergences():
    known = load_exceptions(POLICY, FIXTURES)
    assert set(known) == {'near-tie-uppercase', 'raw-length-0'}
    for name, actual in known.items():
        assert verdict(CASES[name], actual) == 'FAIL'
        assert verdict(CASES[name], actual, known) == 'known-divergence'
        changed = copy.deepcopy(actual)
        changed['ids'][0] += 1
        assert verdict(CASES[name], changed, known) == 'FAIL'


def test_other_fixture_must_remain_exact():
    known = load_exceptions(POLICY, FIXTURES)
    case = CASES['smoke-math']
    assert verdict(case, case['expected'], known) == 'exact'
    changed = copy.deepcopy(case['expected'])
    changed['ids'][0] += 1
    assert verdict(case, changed, known) == 'FAIL'


def test_known_case_can_improve_to_exact():
    assert verdict(CASES['raw-length-0'], CASES['raw-length-0']['expected'],
                   load_exceptions(POLICY, FIXTURES)) == 'exact'


def test_policy_cannot_silently_follow_changed_goldens(tmp_path):
    changed = tmp_path / 'fixtures.json'
    changed.write_text(FIXTURES.read_text() + '\n')
    with pytest.raises(ValueError, match='does not match'):
        load_exceptions(POLICY, changed)


def test_safety_fixture_needs_a_real_failure_record():
    case = CASES['raw-length-128']
    with pytest.raises(ValueError):
        verdict(case, {'status': 'FAIL', 'failure': ''})
    assert verdict(case, {'status': 'FAIL', 'failure': 'prompt too long'}) == 'safety-pass'
