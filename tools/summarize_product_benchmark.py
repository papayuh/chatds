#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Archive completed measurements with clean source and toolchain identities."""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--int16', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    reports = {name: json.loads(getattr(args, name).read_text()) for name in ('baseline', 'int16')}
    for name, report in reports.items():
        if len(report['cases']) != 33 or len({c['name'] for c in report['cases']}) != 33:
            p.error(name + ' is not a completed 33-case run')
    if not all(c['accepted'] for c in reports['int16']['cases']):
        p.error('int16 has an unaccepted regression')
    if not all(c['parity'] for c in reports['baseline']['cases']):
        p.error('baseline did not reproduce its fixtures')
    if digest(ROOT / 'clean/product/build/chatds.nds') != reports['int16']['rom_sha256']:
        p.error('current ROM differs from measured ROM')
    sources = {ROOT / 'clean/product/main.c', ROOT / 'clean/product/Makefile',
               ROOT / 'ds/tokenizer/tok.c', ROOT / 'ds/tokenizer/tok.h',
               ROOT / 'clean/ui/adapter/ui_engine_gen.c', ROOT / 'clean/ui/shell/ui_script.c'}
    for directory, patterns in [('clean/src', ('*.c', '*.h')), ('clean/include', ('*.h',)),
                                ('clean/ui/src', ('*.c', '*.h')), ('clean/ui/include', ('*.h',))]:
        for pattern in patterns:
            sources.update((ROOT / directory).glob(pattern))
    sources.update(ROOT / ('clean/plumbing/' + name) for name in
                   ('calc.c', 'generate.c', 'io.c', 'kb.c', 'retrieve.c', 'plumbing.h'))
    result = {'version': 1, 'date': datetime.now(timezone.utc).date().isoformat(),
              'method': 'one full run per golden input; baseline additionally boots a one-token run per successful input; DS timers, not host wall time',
              'memory': 'largest observed main-RAM base-to-heap-break; not stack/live-allocation high water',
              'sources_sha256': {str(path.relative_to(ROOT)): digest(path) for path in sorted(sources)},
              'reports': reports}
    toolchain = Path(os.environ.get('WONDERFUL_TOOLCHAIN', Path.home() / 'opt/wonderful'))
    compiler = toolchain / 'toolchain/gcc-arm-none-eabi/bin/arm-none-eabi-gcc'
    result['compiler'] = subprocess.check_output([compiler, '--version'], text=True).splitlines()[0]
    result['melonds_flatpak_commit'] = subprocess.check_output(
        ['flatpak', 'info', '--show-commit', 'net.kuribo64.melonDS'], text=True).strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for name, report in reports.items():
        measured = [c for c in report['cases'] if 'ram_break_bytes' in c]
        print(name, 'exact/safety=', sum(c['parity'] for c in report['cases']),
              'accepted=', sum(c['accepted'] for c in report['cases']),
              'peak_break=', max(c['ram_break_bytes'] for c in measured),
              'ROM=', report['rom_sha256'])


if __name__ == '__main__':
    main()
