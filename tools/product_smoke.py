#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Headless smoke of an exact clean kit against its clean host runtime."""
import argparse
import json
from pathlib import Path
import shutil
import struct
import tempfile

import engine_golden as golden

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('kit', type=Path)
    p.add_argument('--one-token', action='store_true')
    p.add_argument('--rom', type=Path)
    args = p.parse_args()
    kit = args.kit.resolve()
    manifest = json.loads((kit / 'kit.json').read_text())
    if manifest.get('engine') != 'clean-dsq8-kv16':
        p.error('expected a clean product kit')
    for name, digest in manifest['sha256'].items():
        path = (kit / name).resolve()
        if not path.is_relative_to(kit) or golden.sha256(path) != digest:
            p.error('kit hash mismatch: ' + name)
    config = dict(line.split('=', 1) for line in (kit / 'chatds/run.txt').read_text().splitlines()
                  if line and not line.startswith('#'))
    model, tok = kit / 'chatds/model.bin', kit / 'chatds/tok.bin'
    kb = kit / 'chatds/kb.bin'
    kb = kb if kb.exists() else None
    instruct = config.get('instruct', '1') != '0'
    with model.open('rb') as f:
        vocab = struct.unpack_from('<I', f.read(40), 36)[0]
    questions = ([config['prompt']] if args.one_token or not kb else
                 ['who was george washington', 'whats 38 times 47', 'sort scores biggest first'])
    golden.command(['make', '-C', ROOT / 'clean/product', 'host'], timeout=120)
    work_root = ROOT / 'build/product-smoke'
    work_root.mkdir(parents=True, exist_ok=True)
    for question in questions:
        work = Path(tempfile.mkdtemp(prefix='case-', dir=work_root))
        host = work / 'host'
        host.mkdir()
        steps = 1 if args.one_token else int(config['steps'])
        (host / 'run.txt').write_text(f'steps={steps}\nkbd=0\ninstruct={int(instruct)}\nprompt={question}\n')
        try:
            golden.command([ROOT / 'clean/product/build/chatds-host', model, tok, kb or '-',
                            host / 'run.txt', str(host) + '/'], timeout=120)
            expected = golden.parse_result((host / 'out.txt').read_text(),
                                           (host / 'ids.txt').read_text(), (host / 'ctx.txt').read_text(),
                                           vocab_size=vocab, allow_context=True)
            case = {'question': question, 'steps': steps, 'instruct': instruct}
            actual = golden.run_rom(args.rom or kit / 'chatds.nds', model, tok, kb, case,
                                    work / 'device', 120, card_dir='chatds')
            mismatch = golden.compare(expected, actual)
            if mismatch:
                raise RuntimeError('host/device mismatch: ' + ','.join(mismatch))
            if instruct:
                golden.command(['mcopy', '-o', '-i', work / 'device/sd.img', '::/chatds/calc.txt',
                                work / 'device/calc.txt'], capture_output=True)
                if (host / 'calc.txt').read_bytes() != (work / 'device/calc.txt').read_bytes():
                    raise RuntimeError('host/device calculator mismatch')
            print(f'PASS {question!r}: exact clean host/device IDs, stop, context, text and calc', flush=True)
        except Exception:
            print(f'Artifacts retained: {work}', flush=True)
            raise
        else:
            shutil.rmtree(work)


if __name__ == '__main__':
    main()
