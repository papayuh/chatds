"""D16 deliverable 2: instruct fine-tune on export_train.py's masked shards.

Derived from reference/ds-llm/train.py (train/train_qat.py doesn't exist in
this repo -- checked). Same configurator.py override style:

  python3 train/train_instruct.py --data_dir=train/data/instruct2048 \\
      --out_dir=train/out/instruct-2048 \\
      --vocab_size=2048 --dim=192 --n_layers=6 --n_heads=6 --n_kv_heads=2 \\
      --max_iters=6000 --device=cuda

Run it from anywhere: the configurator is located next to this file (or in
reference/ds-llm), NOT in the cwd, so `--flag=value` is honoured from the
repo root exactly as documented above. If no configurator can be found and
CLI arguments were passed, we abort instead of silently ignoring them.

What's different from train.py:
  * Data: memmapped (ids.bin uint16, tgt.bin int32) pairs from
    export_train.py, not TinyStories's PretokDataset -- targets already
    carry -1 (ignore_index) on every non-supervised position, no on-the-fly
    masking here. model.py's forward() already does
    F.cross_entropy(..., ignore_index=-1), so no model change was needed.
  * init_from adds "weights_only": loads model weights ONLY (fresh
    optimizer, iter_num=0, fresh lr schedule) -- "resume" restores
    iter/optimizer/best_val_loss, which is what you want to continue an
    interrupted run and is NOT what you want to start a fine-tune.
  * Tokenizer pairing is verified by CONTENT HASH, not by path or by
    vocab_size. corpus/export_train.py records the sha256+size of the
    tokenizer bytes it actually loaded into meta.json; every checkpoint we
    write copies that fingerprint into its config; load_data_meta() re-reads
    the tokenizer file and refuses if it no longer matches what the shards
    were exported with (an in-place tokenizer swap is otherwise invisible).
    Pairing a checkpoint with shards from a different tokenizer of the same
    vocab size trains happily and produces garbage.

    --allow_unverified_tokenizer=True covers exactly one case: MISSING
    provenance (a legacy checkpoint or legacy meta.json that never recorded a
    fingerprint). It never overrides a PROVEN mismatch -- two recorded
    fingerprints that disagree are a hard error with no escape hatch.
  * No DDP / wandb / MFU logging -- single-GPU, one run, ponytail cut.

Loop contract (both were bugs, both are pinned by tests in
train/test_train_instruct.py):
  * exactly `max_iters - iter_num` optimizer updates, no off-by-one extra;
  * a final checkpoint is always written when the loop ends, not only when
    the last iteration happened to land on an eval boundary.

--selftest runs a few iterations against a synthetic in-memory tiny shard
(honors --device=cpu) and asserts the loss is finite, the update count is
exact, lr(0) is in the warmup ramp, and a final checkpoint landed on disk.
"""
import json
import math
import os
import sys
from contextlib import nullcontext

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, os.path.join(HERE, "..", "reference", "ds-llm"))
import qat4  # noqa: E402
from model import Transformer, ModelArgs  # noqa: E402
from export import model_export  # noqa: E402
from tok_fingerprint import (fingerprint_file, compare, describe,  # noqa: E402
                             MATCH, MISMATCH)

# ----------------------------------------------------------------- config
out_dir = "train/out/instruct"
data_dir = "train/data/instruct2048"  # export_train.py's --out-dir
eval_interval = 500
log_interval = 1
eval_iters = 50
always_save_checkpoint = True
init_from = "scratch"       # scratch | resume | weights_only
init_checkpoint = ""        # resume: defaults to out_dir/ckpt.pt; weights_only: required
allow_unverified_tokenizer = False  # see check_tokenizer_pairing()
batch_size = 128
max_seq_len = 256
dim = 192
n_layers = 6
n_heads = 6
n_kv_heads = 2
vocab_size = 2048
multiple_of = 32
dropout = 0.0
gradient_accumulation_steps = 1
trim_padding = False  # independent windows only; crop future masked tail per batch
learning_rate = 3e-4
max_iters = 6000
weight_decay = 1e-1
beta1, beta2 = 0.9, 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 100
device = "cuda"
dtype = "bfloat16"
compile = True
seed = 1337
qat_bits = 0          # 4 = DSQ4 fake-quant (train/qat4.py, E5); 0 = off
qat_start_frac = 0.0  # QAT switches on at this fraction of max_iters
selftest = False
# -----------------------------------------------------------------------
config_keys = [k for k, v in globals().items()
               if not k.startswith("_") and isinstance(v, (int, float, bool, str))]

# Derived values live in globals too, but are recomputed by refresh_derived()
# after any override pass -- otherwise --max_iters=N would leave lr_decay_iters
# and --device=cpu would leave device_type/ctx at their pre-override values.
lr_decay_iters = max_iters
min_lr = 0.0
device_type = "cuda"
ptdtype = torch.bfloat16
ctx = nullcontext()
config = {}


def refresh_derived():
    g = globals()
    g["lr_decay_iters"] = max_iters
    g["device_type"] = "cuda" if "cuda" in device else "cpu"
    g["ptdtype"] = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                    "float16": torch.float16}[dtype]
    g["ctx"] = (nullcontext() if g["device_type"] == "cpu"
                else torch.amp.autocast(device_type=g["device_type"], dtype=g["ptdtype"]))
    g["config"] = {k: g[k] for k in config_keys}
    torch.manual_seed(seed)


def find_configurator():
    """configurator.py is opened relative to cwd by llama2.c's train.py, which
    means the documented `python3 train/train_instruct.py --flag=x` from the
    repo root silently dropped every flag. Look next to this file first, then
    at the vendored reference copy, and only then in the cwd."""
    for cand in (os.path.join(HERE, "configurator.py"),
                 os.path.join(HERE, "..", "reference", "ds-llm", "configurator.py"),
                 os.path.join(os.getcwd(), "configurator.py")):
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return None


def apply_cli_overrides(argv=None):
    """Runs configurator.py against this module's globals (llama2.c style),
    then recomputes derived config. Aborts loudly if there are arguments to
    apply but no configurator to apply them with."""
    argv = list(sys.argv[1:] if argv is None else argv)
    path = find_configurator()
    if path is None:
        if argv:
            raise SystemExit(
                "train_instruct: CLI overrides given but no configurator.py found "
                f"(looked in {HERE}, reference/ds-llm, and {os.getcwd()}); "
                "refusing to run with the flags silently ignored: " + " ".join(argv))
        return
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]] + argv
    try:
        with open(path) as f:
            src = f.read()
        exec(src, globals())  # noqa: S102 -- this IS the configurator contract
    finally:
        sys.argv = saved_argv
    refresh_derived()


def configure(**overrides):
    """Programmatic equivalent of the CLI overrides (used by the tests), with
    the same unknown-key / type-mismatch strictness as configurator.py."""
    g = globals()
    for k, v in overrides.items():
        if k not in config_keys:
            raise ValueError(f"Unknown config key: {k}")
        if type(v) is not type(g[k]):
            raise TypeError(f"{k}: expected {type(g[k]).__name__}, got {type(v).__name__}")
        g[k] = v
    refresh_derived()


refresh_derived()


# ------------------------------------------------------- tokenizer pairing
def tokenizer_fingerprint(path):
    """{path, sha256, size} for a .model file, or None if unreadable.

    Full digest since 2026-09-09; checkpoints written before that carry
    sha256[:16] and tok_fingerprint.compare() matches them on the prefix."""
    return fingerprint_file(path)


def check_tokenizer_pairing(ckpt_meta, data_meta, what):
    """Hard-fail on a checkpoint/dataset pair that cannot be shown to share a
    tokenizer. Same vocab_size is NOT evidence -- tok2048 (TinyStories) and
    tok2048-s2 (corpus) are both 2048 and completely different vocabularies,
    and training one against the other's shards produces a checkpoint that
    looks fine everywhere except in its output.

    Two recorded fingerprints that disagree are PROOF of a wrong pairing:
    that is a hard error, and --allow_unverified_tokenizer does not override
    it. The flag only covers missing provenance."""
    if data_meta is None:
        raise SystemExit(
            f"train_instruct: {what} needs the dataset's meta.json to verify tokenizer "
            f"pairing; {data_dir}/meta.json is missing. Re-run corpus/export_train.py, "
            "or pass --allow_unverified_tokenizer=True if you are certain.")
    ckpt_tok = (ckpt_meta or {}).get("tokenizer")
    data_tok = data_meta.get("tokenizer_fingerprint")
    ckpt_vocab = (ckpt_meta or {}).get("vocab_size")
    data_vocab = data_meta.get("vocab_size")
    if ckpt_vocab is not None and data_vocab is not None and ckpt_vocab != data_vocab:
        raise SystemExit(
            f"train_instruct: {what} vocab_size={ckpt_vocab} but {data_dir} was tokenized "
            f"with vocab_size={data_vocab}. These are different tokenizers; refusing.")
    verdict = compare(ckpt_tok, data_tok)
    if verdict == MATCH:
        print(f"train_instruct: tokenizer pairing OK ({describe(data_tok)})")
        return
    if verdict == MISMATCH:
        raise SystemExit(
            f"train_instruct: {what} was trained with tokenizer {describe(ckpt_tok)} "
            f"but {data_dir} was tokenized with {describe(data_tok)}. These are "
            "provably different tokenizers; refusing to pair them. Re-export the "
            "shards with the checkpoint's tokenizer, or train from scratch. "
            "(--allow_unverified_tokenizer does NOT override a proven mismatch.)")
    missing = "the checkpoint" if not ckpt_tok else f"{data_dir}"
    if allow_unverified_tokenizer:
        print(f"train_instruct: WARNING {missing} carries no tokenizer fingerprint; "
              "--allow_unverified_tokenizer=True, continuing on vocab_size alone.")
        return
    raise SystemExit(
        f"train_instruct: {missing} carries no tokenizer fingerprint, so {what} cannot "
        f"be shown to match the tokenizer behind {data_dir}. Matching vocab_size is not "
        "evidence (tok2048 vs tok2048-s2). Train from scratch, re-export the shards, or "
        "pass --allow_unverified_tokenizer=True to accept the risk.")


# --------------------------------------------------------- masked dataset
class MaskedShardDataset:
    """Memmaps export_train.py's {split}.ids.bin/{split}.tgt.bin. A batch is
    `batch_size` random non-overlapping seq_len windows -- the shards are
    already packed to seq_len with no padding, so this is a straight
    reshape-and-index, no shifting (targets already hold the shifted,
    masked value at each position).

    Every failure mode here is a wrong-shards mistake that would otherwise
    show up as a silently bad model, so each one gets its own message."""

    def __init__(self, data_dir, split, seq_len, vocab_size=None, meta=None,
                 trim_padding=False):
        if trim_padding and (meta or {}).get("packing") != "independent":
            raise SystemExit("train_instruct: trim_padding requires independent packing metadata")
        self.trim_padding = trim_padding
        ids_path = os.path.join(data_dir, f"{split}.ids.bin")
        tgt_path = os.path.join(data_dir, f"{split}.tgt.bin")
        for p in (ids_path, tgt_path):
            if not os.path.exists(p):
                raise SystemExit(
                    f"train_instruct: {p} not found. Build the shards first:\n"
                    f"  python3 corpus/export_train.py --pairs corpus/results/full-train.jsonl "
                    f"--tokenizer path/to/your-tokenizer.model --out-dir {data_dir} --seq-len {seq_len}")
        ids = np.memmap(ids_path, dtype=np.uint16, mode="r")
        tgt = np.memmap(tgt_path, dtype=np.int32, mode="r")
        if len(ids) != len(tgt):
            raise SystemExit(
                f"train_instruct: {ids_path} has {len(ids)} tokens but {tgt_path} has "
                f"{len(tgt)} -- the two files are from different export_train.py runs.")
        if meta is not None:
            meta_seq = meta.get("seq_len")
            if meta_seq is not None and meta_seq != seq_len:
                raise SystemExit(
                    f"train_instruct: shards in {data_dir} were packed at seq_len={meta_seq} "
                    f"but max_seq_len={seq_len}. Windows would be sliced mid-example; "
                    f"pass --max_seq_len={meta_seq} or re-export.")
            meta_vocab = meta.get("vocab_size")
            if vocab_size is not None and meta_vocab is not None and meta_vocab != vocab_size:
                raise SystemExit(
                    f"train_instruct: shards in {data_dir} are vocab_size={meta_vocab} "
                    f"but --vocab_size={vocab_size}.")
        n_windows = len(ids) // seq_len
        if n_windows == 0:
            raise SystemExit(
                f"train_instruct: {data_dir}/{split}: {len(ids)} tokens is less than one "
                f"{seq_len}-token window.")
        self.n_windows = n_windows
        self.ids = ids[: n_windows * seq_len].reshape(n_windows, seq_len)
        self.tgt = tgt[: n_windows * seq_len].reshape(n_windows, seq_len)
        self.seq_len = seq_len
        if vocab_size is not None:
            self.validate_range(split, vocab_size)

    def validate_range(self, split, vocab_size):
        """Out-of-range ids are a CUDA device-side assert 20 minutes into a
        run; out-of-range targets are a silent NaN. Check both up front --
        a full pass over a few hundred MB of memmap is seconds."""
        id_max = int(self.ids.max())
        if id_max >= vocab_size:
            raise SystemExit(
                f"train_instruct: {data_dir}/{split}.ids.bin holds token id {id_max} but "
                f"vocab_size={vocab_size}. The shards were tokenized with a larger vocab.")
        tgt_min, tgt_max = int(self.tgt.min()), int(self.tgt.max())
        if tgt_max >= vocab_size:
            raise SystemExit(
                f"train_instruct: {data_dir}/{split}.tgt.bin holds target {tgt_max} but "
                f"vocab_size={vocab_size}.")
        if tgt_min < -1:
            raise SystemExit(
                f"train_instruct: {data_dir}/{split}.tgt.bin holds target {tgt_min}; the "
                "only legal negative target is -1 (ignore_index).")
        n_sup = int((np.asarray(self.tgt) != -1).sum())
        if n_sup == 0:
            raise SystemExit(
                f"train_instruct: {data_dir}/{split}: every position is masked (-1), the "
                "loss would be NaN. The exporter masked the whole split.")
        print(f"train_instruct: {split}: {self.n_windows} windows x {self.seq_len}, "
              f"{n_sup} supervised positions ({100.0 * n_sup / self.tgt.size:.1f}%)")

    def get_batch(self, batch_size, device, rng):
        ix = rng.integers(0, self.n_windows, size=batch_size)
        x_np = self.ids[ix].astype(np.int64)
        y_np = self.tgt[ix].astype(np.int64)
        if self.trim_padding:
            columns = np.flatnonzero((y_np != -1).any(axis=0))
            if not len(columns):
                raise SystemExit("train_instruct: sampled batch has no supervised targets")
            end = int(columns[-1]) + 1
            # Only FUTURE masked positions are removed. Causal attention at
            # every supervised position sees the exact same prefix; keep the
            # final EOS target, even when EOS itself is not an input token.
            x_np = np.ascontiguousarray(x_np[:, :end])
            y_np = np.ascontiguousarray(y_np[:, :end])
        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)
        if device_type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y


def load_data_meta(data_dir):
    """export_train.py's meta.json, with its EXPORT-TIME tokenizer fingerprint
    re-verified against the file that fingerprint names.

    The fingerprint is evidence only because export_train.py recorded it from
    the bytes it loaded. Recomputing one here from whatever file now sits at
    meta["tokenizer"] would invent provenance: swapping the tokenizer in place
    after the shards were written would then look perfectly consistent. So:

      * recorded and present disagree -> hard error, ALWAYS (this is proof the
        shards and the file no longer describe the same tokenizer, and
        --allow_unverified_tokenizer does not cover proof);
      * recorded present, file gone -> keep the recorded fingerprint (it is
        still what produced the shards) and warn;
      * nothing recorded (legacy meta.json) -> no fingerprint, no invention.
        check_tokenizer_pairing() then demands --allow_unverified_tokenizer.
    """
    path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        meta = json.load(f)

    recorded = meta.get("tokenizer_fingerprint")
    tok_path = meta.get("tokenizer")
    if not recorded:
        if not allow_unverified_tokenizer:
            raise SystemExit(
                f"train_instruct: {path} has no tokenizer fingerprint. Re-export "
                "with corpus/export_train.py, or explicitly accept missing "
                "provenance with --allow_unverified_tokenizer=True.")
        meta["tokenizer_fingerprint"] = None
        print(f"train_instruct: WARNING {path} predates export-time tokenizer "
              "fingerprints, so these shards carry no provenance at all. Re-run "
              "corpus/export_train.py to record it.")
        return meta

    present = fingerprint_file(tok_path)
    verdict = compare(recorded, present)
    if verdict == MISMATCH:
        raise SystemExit(
            f"train_instruct: {path} says its shards were tokenized with "
            f"{describe(recorded)}, but {tok_path} now holds {describe(present)}. "
            "The tokenizer file was replaced after the shards were exported, so the "
            "ids in these shards mean something else. Re-run corpus/export_train.py "
            "against the tokenizer you intend to use. (This is proof, not a "
            "heuristic: --allow_unverified_tokenizer does not override it.)")
    if verdict != MATCH:
        print(f"train_instruct: WARNING {tok_path} is missing; keeping the export-time "
              f"fingerprint {describe(recorded)} from {path} as the pairing evidence.")
    meta["tokenizer_fingerprint"] = recorded
    return meta


# ------------------------------------------------------------ model / opt
def strip_compile_prefix(state_dict):
    for k in list(state_dict.keys()):
        if k.startswith("_orig_mod."):
            state_dict[k[len("_orig_mod."):]] = state_dict.pop(k)
    return state_dict


def make_model(model_args_cli, data_meta=None):
    """Returns (model, model_args, iter_num, best_val_loss, optim_state).

    optim_state is None for scratch/weights_only (fresh AdamW moments, fresh
    lr schedule) and the checkpoint's optimizer state for resume -- resume
    that restores iter_num but not the optimizer throws away the Adam moments
    and silently changes what the next step does."""
    iter_num, best_val_loss, optim_state = 0, 1e9, None

    if init_from == "scratch":
        print("Initializing a new model from scratch")
        model_args = dict(model_args_cli)
        model = Transformer(ModelArgs(**model_args))
    elif init_from in ("resume", "weights_only"):
        ckpt_path = init_checkpoint or (os.path.join(out_dir, "ckpt.pt")
                                        if init_from == "resume" else "")
        if not ckpt_path:
            raise SystemExit("train_instruct: --init_from=weights_only needs "
                             "--init_checkpoint=PATH")
        if not os.path.exists(ckpt_path):
            raise SystemExit(f"train_instruct: --init_from={init_from} but {ckpt_path} "
                             "does not exist")
        print(f"{'Resuming training from' if init_from == 'resume' else 'Loading weights ONLY from'}"
              f" {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        check_tokenizer_pairing(checkpoint.get("config"), data_meta, ckpt_path)
        model_args = dict(model_args_cli)
        for k in ("dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size",
                  "multiple_of", "max_seq_len"):
            if k in checkpoint["model_args"]:
                model_args[k] = checkpoint["model_args"][k]
        model = Transformer(ModelArgs(**model_args))
        model.load_state_dict(strip_compile_prefix(checkpoint["model"]))
        if init_from == "resume":
            iter_num = checkpoint["iter_num"]
            best_val_loss = checkpoint["best_val_loss"]
            optim_state = checkpoint.get("optimizer")
            if optim_state is None:
                print("train_instruct: WARNING resume checkpoint has no optimizer state; "
                      "AdamW moments start from zero.")
        else:
            # The fine-tune path: weights in, everything else fresh.
            print("train_instruct: fresh optimizer / iter_num=0 / fresh lr schedule")
    else:
        raise SystemExit(f"train_instruct: unknown init_from={init_from!r}")

    model.to(device)
    return model, model_args, iter_num, best_val_loss, optim_state


def build_optimizer(model, optim_state):
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2),
                                           device_type)
    if optim_state is not None:
        optimizer.load_state_dict(optim_state)
        print("train_instruct: restored optimizer state from checkpoint")
    return optimizer


def maybe_compile(model):
    """Returns (step_model, raw_model). raw_model is ALWAYS the uncompiled
    module: taking state_dict()/model_export() off the torch.compile wrapper
    writes `_orig_mod.`-prefixed keys and hands export.py an OptimizedModule
    instead of a Transformer."""
    raw_model = model
    if compile and device_type == "cuda":
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(raw_model)
    return model, raw_model


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, (lr_decay_iters - warmup_iters))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss(model, train_ds, val_ds, rng):
    out = {}
    model.eval()
    for name, ds in (("train", train_ds), ("val", val_ds)):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = ds.get_batch(batch_size, device, rng)
            with ctx:
                model(X, Y)
                losses[k] = model.last_loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(raw_model, optimizer, model_args, iter_num, best_val_loss,
                    data_meta=None, tag="ckpt"):
    """Writes out_dir/{tag}.pt plus the v0 float export. The checkpoint's
    config records the tokenizer fingerprint so a later resume/fine-tune can
    refuse a mismatched dataset instead of training garbage."""
    os.makedirs(out_dir, exist_ok=True)
    state = strip_compile_prefix(dict(raw_model.state_dict()))
    assert not any(k.startswith("_orig_mod.") for k in state), "compiled state_dict leaked"
    cfg = dict(config)
    if data_meta is not None:
        cfg["tokenizer"] = data_meta.get("tokenizer_fingerprint")
        cfg["data_meta"] = {k: v for k, v in data_meta.items() if k != "tokenizer_fingerprint"}
    ckpt = {"model": state, "optimizer": optimizer.state_dict(),
            "model_args": model_args, "iter_num": iter_num,
            "best_val_loss": best_val_loss, "config": cfg}
    path = os.path.join(out_dir, f"{tag}.pt")
    torch.save(ckpt, path)
    model_export(raw_model, os.path.join(out_dir, f"{tag if tag != 'ckpt' else 'model'}.bin"),
                 version=0)
    print(f"train_instruct: saved {path} (iter {iter_num})")
    return path


def train(model, model_args, iter_num, best_val_loss, train_ds, val_ds, data_meta=None,
          optim_state=None):
    """Runs exactly max_iters - iter_num optimizer updates, then saves.

    Returns {"n_updates", "iter_num", "last_loss", "ckpt_path"}."""
    os.makedirs(out_dir, exist_ok=True)
    optimizer = build_optimizer(model, optim_state)
    scaler = torch.amp.GradScaler(device_type, enabled=(dtype == "float16"))
    model, raw_model = maybe_compile(model)

    rng = np.random.default_rng(seed)
    X, Y = train_ds.get_batch(batch_size, device, rng)
    last_loss, n_updates = None, 0

    while iter_num < max_iters:
        assert qat_bits in (0, 4), "qat_bits must be 0 or 4"
        qat4.enable(qat_bits == 4 and iter_num >= qat_start_frac * max_iters)
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if iter_num % eval_interval == 0 and iter_num > 0:
            losses = estimate_loss(model, train_ds, val_ds, rng)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, "
                  f"val loss {losses['val']:.4f}")
            if losses["val"] < best_val_loss or always_save_checkpoint:
                best_val_loss = min(best_val_loss, losses["val"])
                save_checkpoint(raw_model, optimizer, model_args, iter_num,
                                best_val_loss, data_meta)

        for _ in range(gradient_accumulation_steps):
            with ctx:
                model(X, Y)
                loss = raw_model.last_loss / gradient_accumulation_steps
            X, Y = train_ds.get_batch(batch_size, device, rng)
            scaler.scale(loss).backward()
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        last_loss = loss.item() * gradient_accumulation_steps
        if iter_num % log_interval == 0:
            print(f"{iter_num} | loss {last_loss:.4f} | lr {lr:e}")
        iter_num += 1
        n_updates += 1

    # Always land a checkpoint at the end of the run: the eval-boundary save
    # above only fires when max_iters happens to be a multiple of
    # eval_interval, which silently threw away the last N iterations.
    ckpt_path = save_checkpoint(raw_model, optimizer, model_args, iter_num,
                                best_val_loss, data_meta)
    qat4.enable(False)
    print(f"train_instruct: done -- {n_updates} optimizer updates, iter_num={iter_num}")
    return {"n_updates": n_updates, "iter_num": iter_num, "last_loss": last_loss,
            "ckpt_path": ckpt_path, "optimizer": optimizer, "raw_model": raw_model}


def run(train_ds, val_ds, model_args_cli, data_meta=None):
    """make_model + train, wired so the optimizer state travels with it."""
    model, model_args, iter_num, best_val_loss, optim_state = make_model(model_args_cli,
                                                                        data_meta)
    return train(model, model_args, iter_num, best_val_loss, train_ds, val_ds, data_meta,
                 optim_state=optim_state)


# ------------------------------------------------------------- selftest
def build_synthetic_shard(rng, n_windows, seq_len, vocab):
    """A --selftest shard: valid uint16 ids, targets -1 on a random prefix
    of each row (like real prompt masking) then shifted ids after that."""
    ids = rng.integers(3, vocab, size=(n_windows, seq_len)).astype(np.uint16)
    tgt = np.full((n_windows, seq_len), -1, dtype=np.int32)
    for r in range(n_windows):
        split = rng.integers(1, seq_len - 1)
        tgt[r, split:seq_len - 1] = ids[r, split + 1:seq_len]
    return ids, tgt


class InMemoryShard:
    def __init__(self, ids, tgt):
        self.ids, self.tgt = ids, tgt
        self.n_windows = ids.shape[0]
        self.seq_len = ids.shape[1]

    def get_batch(self, batch_size, device, rng):
        ix = rng.integers(0, self.n_windows, size=batch_size)
        x = torch.from_numpy(self.ids[ix].astype(np.int64)).to(device)
        y = torch.from_numpy(self.tgt[ix].astype(np.int64)).to(device)
        return x, y


def tiny_setup(seq_len=32, n_windows=64, vocab=None):
    """Synthetic shards + a ~10k-param model config, shared by --selftest and
    the tests."""
    vocab = vocab_size if vocab is None else vocab
    rng = np.random.default_rng(seed)
    ids, tgt = build_synthetic_shard(rng, n_windows, seq_len, vocab)
    ds = InMemoryShard(ids, tgt)
    args = dict(dim=32, n_layers=2, n_heads=2, n_kv_heads=2, vocab_size=vocab,
                multiple_of=32, max_seq_len=seq_len, dropout=0.0)
    return ds, args


def run_selftest():
    if device_type != "cpu":
        raise SystemExit("train_instruct: --selftest is meant to run with --device=cpu")
    configure(compile=False, batch_size=min(batch_size, 8),
              eval_interval=max(eval_interval, 10 ** 6))
    if max_iters > 20:
        configure(max_iters=3)
    ds, args = tiny_setup(seq_len=min(max_seq_len, 32))
    want_updates = max_iters
    res = run(ds, ds, args, data_meta=None)

    assert math.isfinite(res["last_loss"]), f"loss not finite: {res['last_loss']}"
    assert res["n_updates"] == want_updates, \
        f"ran {res['n_updates']} updates, expected exactly {want_updates}"
    assert os.path.exists(res["ckpt_path"]), f"no final checkpoint at {res['ckpt_path']}"
    lr0 = get_lr(0)
    assert lr0 < learning_rate * 0.5, f"schedule doesn't start at warmup: lr(0)={lr0}"
    print(f"SELFTEST PASS: {res['n_updates']} iters, final loss={res['last_loss']:.4f}, "
          f"lr(0)={lr0:.2e} (warmup), final ckpt {res['ckpt_path']}")


def main():
    apply_cli_overrides()
    if selftest:
        run_selftest()
        return 0

    tokens_per_iter = gradient_accumulation_steps * batch_size * max_seq_len
    print(f"{'maximum window positions' if trim_padding else 'tokens'} per iteration: {tokens_per_iter:,}")

    data_meta = load_data_meta(data_dir)
    if data_meta is None and not allow_unverified_tokenizer:
        raise SystemExit(
            f"train_instruct: {data_dir}/meta.json is missing -- it is what pins which "
            "tokenizer produced these shards. Re-run corpus/export_train.py, or pass "
            "--allow_unverified_tokenizer=True.")
    train_ds = MaskedShardDataset(data_dir, "train", max_seq_len, vocab_size, data_meta,
                                 trim_padding=trim_padding)
    val_ds = MaskedShardDataset(data_dir, "val", max_seq_len, vocab_size, data_meta,
                               trim_padding=trim_padding)

    cli_model_args = dict(dim=dim, n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv_heads,
                          vocab_size=vocab_size, multiple_of=multiple_of,
                          max_seq_len=max_seq_len, dropout=dropout)
    run(train_ds, val_ds, cli_model_args, data_meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
