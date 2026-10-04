"""Exercise clean answer capture through the real host executable and CLI."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import run_eval

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('product_assets', ROOT / 'clean/product/test_product.py')
product = importlib.util.module_from_spec(spec)
spec.loader.exec_module(product)


@pytest.fixture(scope='module')
def host():
    subprocess.run(['make', '-C', ROOT / 'clean/product', 'host'], check=True)
    return ROOT / 'clean/product/build/chatds-host'


def test_real_capture_excludes_question_and_expected_answer(host, tmp_path):
    model, tokenizer = product.assets(tmp_path)
    args = SimpleNamespace(host=host, model=model, tokenizer=tokenizer,
                           chatds_kb=None, timeout=10)
    suite = [{'id': 'a', 'prompt': 'secret question', 'answer': 'oracle canary',
              'max_new_tokens': 2}, {'id': 'b', 'prompt': 'another', 'max_new_tokens': 1}]
    rows = run_eval.run_capture(args, suite)
    assert [r['id'] for r in rows] == ['a', 'b']
    assert rows[0]['tokens_generated'] == 2 and rows[0]['stop'] == 'budget'
    assert rows[1]['output'] and rows[0]['output'] == rows[1]['output'] * 2
    assert all('secret question' not in r['output'] and 'oracle canary' not in r['output']
               for r in rows)
    assert rows[0]['raw_output'] == rows[0]['output']


@pytest.mark.parametrize('prompt,steps', [('x' * 128, 1), ('bad\nline', 1), ('é', 1), ('ok', 257)])
def test_input_limits_fail_before_inference(host, tmp_path, prompt, steps):
    model, tokenizer = product.assets(tmp_path)
    args = SimpleNamespace(host=host, model=model, tokenizer=tokenizer,
                           chatds_kb=None, timeout=10)
    with pytest.raises(ValueError, match='input limits'):
        run_eval.run_capture(args, [{'id': 'bad', 'prompt': prompt, 'max_new_tokens': steps}])


def test_cli_writes_jsonl_and_labels_subset(host, tmp_path):
    model, tokenizer = product.assets(tmp_path)
    suite = tmp_path / 'suite.jsonl'
    rows = [dict(id=str(n), category='python', prompt='a', answer='<unk>',
                 accept=['<unk>'], scoring='exact', max_new_tokens=1) for n in range(2)]
    suite.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    out = tmp_path / 'answers.jsonl'
    result = subprocess.run([sys.executable, run_eval.__file__, '--model', model,
                             '--tokenizer', tokenizer, '--host', host, '--suite', suite,
                             '--out', out, '--limit', '1'], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert 'DEVELOPMENT SUBSET' in result.stderr
    captured = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(captured) == 1 and captured[0]['id'] == '0'
    assert captured[0]['tokens_generated'] == 1 and captured[0]['output']


def test_calculator_capture_preserves_raw_tool_evidence(tmp_path):
    model, tokenizer = product.assets(tmp_path)
    host = tmp_path / 'host'
    files = {'out.txt': ('calc(38*47)\ntokens=2\nprompt_tokens=1\n'
                         'generated=2\nstop=budget\nms=1\nstatus=OK\n'),
             'ids.txt': '3\n4\n', 'ctx.txt': 'ctx=\n',
             'calc.txt': 'calc=calc(38*47) = 1786\n'}
    host.write_text('#!/usr/bin/env python3\nimport pathlib,sys\n'
                    'out=pathlib.Path(sys.argv[5])\n'
                    f'files={files!r}\n'
                    'for name,text in files.items(): (out/name).write_text(text)\n')
    host.chmod(0o755)
    args = SimpleNamespace(host=host, model=model, tokenizer=tokenizer,
                           chatds_kb=None, timeout=10)
    row = run_eval.run_capture(args, [{'id': 'calc', 'prompt': 'whats 38 times 47',
                                     'max_new_tokens': 2}])[0]
    assert row['output'] == 'calc(38*47) = 1786'
    assert row['raw_output'] == 'calc(38*47)'
    assert row['tokens_generated'] == 2
