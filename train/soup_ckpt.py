#!/usr/bin/env python3
"""Uniform weight average of finished checkpoints that share an init (model soup). usage: soup_ckpt.py OUT_DIR RUN_DIR [RUN_DIR ...]
Writes OUT_DIR/ckpt.pt (metadata from the first run), tokenizer files copied from it; no device-format exporter is included."""
import os, shutil, sys, torch

out, runs = sys.argv[1], sys.argv[2:]
cks = [torch.load(os.path.join(r, "ckpt.pt"), map_location="cpu", weights_only=False) for r in runs]
avg = {k: sum(c["model"][k].float() for c in cks) / len(cks) for k in cks[0]["model"]}
cks[0]["model"] = avg
os.makedirs(out, exist_ok=True)
assert not os.path.exists(os.path.join(out, "ckpt.pt"))
torch.save(cks[0], os.path.join(out, "ckpt.pt"))
for f in ("tokenizer.model", "tokenizer.bin"):
    shutil.copy(os.path.join(runs[0], f), out)
open(os.path.join(out, "SOUP.txt"), "w").write("\n".join(runs) + "\n")
