"""Keep completed checkpoints only; never copy on the pre-save eval line.

Copied out of train/out/python-agent-20260910-r7-v2/source/retain-checkpoints.py
(originally /tmp/ds-agent-retain-checkpoints.py) as a real repo file for R8+ so
Checkpoint retention has no dependency on scratch files surviving a reboot.
Usage unchanged: `python3 train/retain-checkpoints.py <run-out-dir> <parent-pid> <max-iter>`.
"""
import os
from pathlib import Path
import re
import shutil
import sys
import time
import torch

run = Path(sys.argv[1])
parent_pid = int(sys.argv[2])
maximum = int(sys.argv[3])
snapshots = run / 'snapshots'
snapshots.mkdir(exist_ok=True)
pattern = re.compile(r'^train_instruct: saved .*/ckpt\.pt \(iter (\d+)\)$', re.M)
while True:
    try:
        os.kill(parent_pid, 0)
    except ProcessLookupError:
        break
    log = run / 'train.log'
    hits = pattern.findall(log.read_text()) if log.exists() else []
    if hits:
        n = int(hits[-1])
        dest = snapshots / f'iter-{n:06d}.pt'
        if not dest.exists():
            tmp = snapshots / '.pending.pt'
            try:
                shutil.copyfile(run / 'ckpt.pt', tmp)
                ck = torch.load(tmp, map_location='cpu', weights_only=False)
                if ck['iter_num'] != n:
                    raise ValueError(f'expected iter {n}, copied {ck["iter_num"]}')
                del ck
                os.replace(tmp, dest)
                print(f'RETAINED iter={n} path={dest}', flush=True)
            except Exception as exc:
                tmp.unlink(missing_ok=True)
                print(f'RETAIN_RETRY iter={n}: {exc}', flush=True)
        if n >= maximum and dest.exists():
            break
    time.sleep(2)
