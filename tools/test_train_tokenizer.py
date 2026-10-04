"""Train/export a new tokenizer through its public CLI, without pretrained assets."""
import hashlib
from pathlib import Path
import subprocess
import sys

import sentencepiece as spm

from tokenizer_fixture import model_directory

ROOT = Path(__file__).resolve().parent.parent


def test_cli_trains_compatible_ids_whitespace_and_exports(tmp_path):
    prefix = tmp_path / 'custom'
    command = [sys.executable, ROOT / 'tools/train_tokenizer.py', '--input',
               model_directory() / 'synthetic.txt', '--prefix', prefix, '--vocab-size', '512']
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    model = Path(str(prefix) + '.model')
    sp = spm.SentencePieceProcessor(model_file=str(model))
    assert (sp.unk_id(), sp.bos_id(), sp.eos_id(), sp.get_piece_size()) == (0, 1, 2, 512)
    text = 'hello  world\n  print(x)'
    assert sp.decode(sp.encode(text)) == text
    binary = tmp_path / 'tok.bin'
    exported = subprocess.run([sys.executable, ROOT / 'ds/tokenizer/tokbin.py', model, binary],
                              capture_output=True, text=True, timeout=30)
    assert exported.returncode == 0 and binary.stat().st_size > 0
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    refused = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert refused.returncode != 0 and 'refusing to replace' in refused.stderr
    assert hashlib.sha256(model.read_bytes()).hexdigest() == digest


def test_cli_rejects_vocab_below_byte_fallback_floor(tmp_path):
    result = subprocess.run([sys.executable, ROOT / 'tools/train_tokenizer.py',
                             '--input', tmp_path / 'missing', '--prefix', tmp_path / 'tok',
                             '--vocab-size', '32'], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and '260..65536' in result.stderr
    assert not Path(str(tmp_path / 'tok') + '.model').exists()
