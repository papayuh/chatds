"""Choose by training's logged validation loss, never frozen-suite accuracy.

Policy fixed before final evaluations: lowest logged val loss, earliest update
on a tie. Metrics are logged to four decimals. Random-pair validation is optimistic
about unseen task families; this selects a candidate, not a capability claim.

Copied out of train/out/python-agent-20260910-r7-v2/source/select-checkpoint.py
(originally /tmp/ds-agent-select-python.py) as a real repo file for R8+ so
Checkpoint selection has no dependency on scratch files surviving a reboot.
Usage unchanged: `python3 train/select-checkpoint.py <run-out-dir>`.
"""
import json
from pathlib import Path
import re
import shutil
import sys
import torch

root = Path(sys.argv[1])
assert (root / 'finished.txt').is_file(), 'training/export/eval pipeline incomplete'
pattern = re.compile(r'^step (\d+): train loss ([0-9.]+), val loss ([0-9.]+)$', re.M)
rows = [{'iter_num': int(n), 'train_loss': float(t), 'val_loss': float(v)}
        for n, t, v in pattern.findall((root / 'train.log').read_text())]
assert rows, 'no validation history'
best = min(rows, key=lambda r: (r['val_loss'], r['iter_num']))
source = root / 'snapshots' / f'iter-{best["iter_num"]:06d}.pt'
ck = torch.load(source, map_location='cpu', weights_only=False)
assert ck['iter_num'] == best['iter_num']
del ck
dest = root / f'selected-iter-{best["iter_num"]}'
dest.mkdir()  # refuses replacement of any prior selection
shutil.copyfile(source, dest / 'ckpt.pt')
for name in ('tokenizer.model', 'tokenizer.bin'):
    shutil.copyfile(root / name, dest / name)
report = {'criterion': 'lowest logged validation loss; earliest on tie',
          'warning': 'same-pool random validation, not unseen-family/personal acceptance',
          'frozen_suite_used_for_selection': False, 'history': rows,
          'selected': best, 'directory': str(dest)}
(root / 'selection.json').write_text(json.dumps(report, indent=2) + '\n')
print(dest)
