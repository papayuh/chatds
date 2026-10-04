"""CPU smoke tests for train/train_instruct.py -- one test per bug that was
actually in the file, so a regression fails here instead of 6000 GPU steps
later. Everything runs on a ~90k-param model against synthetic shards in
tmp_path; the whole file is a few seconds.

  python3 -m pytest train/test_train_instruct.py -q      # from the repo root
  python3 -m pytest test_train_instruct.py -q            # from train/

Covered:
  * CLI overrides are honoured from the repo root AND from an unrelated cwd
    (the configurator used to be loaded from cwd only, so the documented
    root invocation silently ignored every flag);
  * exactly max_iters optimizer updates, no off-by-one extra;
  * a final checkpoint always lands, including when max_iters is not a
    multiple of eval_interval;
  * init_from=weights_only -> fresh optimizer/iter/lr schedule;
  * init_from=resume -> optimizer moments AND iter_num restored;
  * torch.compile wrapper never reaches state_dict()/model_export();
  * dataset shape/vocab/mask validation fails with a useful message;
  * checkpoint<->shards tokenizer pairing cannot be mismatched silently,
    including a tokenizer swapped IN PLACE after the shards were exported,
    legacy metadata that records no fingerprint at all, and the old
    truncated 16-char checkpoint digests.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))

import tok_fingerprint as tf  # noqa: E402
import train_instruct as ti  # noqa: E402

VOCAB = 64
SEQ = 32
TINY = dict(dim=32, n_layers=2, n_heads=2, n_kv_heads=2, vocab_size=VOCAB,
            multiple_of=32, max_seq_len=SEQ, dropout=0.0)


@pytest.fixture(autouse=True)
def clean_config():
    """train_instruct keeps its config in module globals (llama2.c style), so
    restore them between tests."""
    saved = {k: getattr(ti, k) for k in ti.config_keys}
    ti.configure(device="cpu", dtype="float32", compile=False, seed=1337,
                 batch_size=4, max_seq_len=SEQ, vocab_size=VOCAB,
                 log_interval=1000, eval_interval=1000, eval_iters=2,
                 warmup_iters=2, learning_rate=1e-3, max_iters=3,
                 init_from="scratch", init_checkpoint="",
                 allow_unverified_tokenizer=False)
    yield
    ti.configure(**saved)


# ------------------------------------------------------------- fixtures
def write_tokenizer(path, body=b"fake-sentencepiece-model"):
    with open(path, "wb") as f:
        f.write(body)
    return str(path)


def write_shards(dir_path, tokenizer_path, n_train=64, n_val=8, seq_len=SEQ,
                 vocab=VOCAB, legacy_meta=False, **meta_over):
    """Synthetic stand-in for corpus/export_train.py's output.

    Records the export-time tokenizer fingerprint exactly like the real
    exporter does. legacy_meta=True reproduces a meta.json written before
    fingerprints existed (path only, no provenance)."""
    os.makedirs(dir_path, exist_ok=True)
    rng = np.random.default_rng(0)
    for split, n in (("train", n_train), ("val", n_val)):
        ids, tgt = ti.build_synthetic_shard(rng, n, seq_len, vocab)
        ids.reshape(-1).tofile(os.path.join(dir_path, f"{split}.ids.bin"))
        tgt.reshape(-1).tofile(os.path.join(dir_path, f"{split}.tgt.bin"))
    meta = {"seq_len": seq_len, "vocab_size": vocab,
            "tokenizer": str(tokenizer_path),
            "train_windows": n_train, "val_windows": n_val}
    if not legacy_meta:
        meta["tokenizer_fingerprint"] = tf.fingerprint_file(str(tokenizer_path))
    meta.update(meta_over)
    with open(os.path.join(dir_path, "meta.json"), "w") as f:
        json.dump(meta, f)
    return str(dir_path)


@pytest.fixture
def shards(tmp_path):
    tok = write_tokenizer(tmp_path / "tok.model")
    data = write_shards(tmp_path / "data", tok)
    return data, tok


def datasets(data_dir):
    meta = ti.load_data_meta(data_dir)
    train_ds = ti.MaskedShardDataset(data_dir, "train", ti.max_seq_len, ti.vocab_size, meta)
    val_ds = ti.MaskedShardDataset(data_dir, "val", ti.max_seq_len, ti.vocab_size, meta)
    return train_ds, val_ds, meta


# ------------------------------------------- 1. CLI overrides from any cwd
@pytest.mark.parametrize("cwd", [REPO, HERE, "/tmp"])
def test_cli_overrides_are_honoured_from_any_cwd(cwd, tmp_path):
    out = tmp_path / f"out-{os.path.basename(cwd) or 'root'}"
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "train_instruct.py"),
         "--selftest=True", "--device=cpu", "--dtype=float32", "--compile=False",
         f"--out_dir={out}", "--max_iters=2", "--batch_size=4", "--warmup_iters=2",
         f"--vocab_size={VOCAB}", f"--max_seq_len={SEQ}"],
        cwd=cwd, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Overriding: max_iters = 2" in proc.stdout, "flags were silently ignored"
    assert "2 optimizer updates" in proc.stdout
    assert "SELFTEST PASS" in proc.stdout
    assert os.path.exists(out / "ckpt.pt")


def test_configurator_is_found_next_to_the_script():
    found = ti.find_configurator()
    assert found and os.path.exists(found)
    assert os.path.dirname(found) in (HERE, os.path.join(REPO, "reference", "ds-llm"))


def test_unknown_flag_is_rejected(tmp_path):
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "train_instruct.py"), "--nonsense=1"],
        cwd=REPO, capture_output=True, text=True, timeout=600)
    assert proc.returncode != 0
    assert "Unknown config key" in proc.stderr


# --------------------------------------------------- 2. exact update count
@pytest.mark.parametrize("max_iters", [1, 3, 5])
def test_exact_update_count(shards, tmp_path, max_iters):
    data, _ = shards
    ti.configure(out_dir=str(tmp_path / "out"), max_iters=max_iters)
    train_ds, val_ds, meta = datasets(data)
    res = ti.run(train_ds, val_ds, dict(TINY), meta)
    assert res["n_updates"] == max_iters
    assert res["iter_num"] == max_iters
    assert torch.load(res["ckpt_path"], weights_only=False)["iter_num"] == max_iters


# ------------------------------------------------- 3. final checkpoint save
def test_final_checkpoint_saved_off_eval_boundary(shards, tmp_path):
    """max_iters=5 with eval_interval=2 never lands an eval on the last step;
    the old loop only ever saved at eval boundaries, so iterations 4 and 5
    were thrown away."""
    data, _ = shards
    out = tmp_path / "out"
    ti.configure(out_dir=str(out), max_iters=5, eval_interval=2)
    train_ds, val_ds, meta = datasets(data)
    res = ti.run(train_ds, val_ds, dict(TINY), meta)
    ckpt = torch.load(out / "ckpt.pt", weights_only=False)
    assert ckpt["iter_num"] == 5, "final checkpoint is not from the last iteration"
    assert res["ckpt_path"] == str(out / "ckpt.pt")
    assert (out / "model.bin").exists(), "v0 export missing next to the checkpoint"


def test_final_checkpoint_has_optimizer_and_matches_model(shards, tmp_path):
    data, _ = shards
    out = tmp_path / "out"
    ti.configure(out_dir=str(out), max_iters=3)
    train_ds, val_ds, meta = datasets(data)
    res = ti.run(train_ds, val_ds, dict(TINY), meta)
    ckpt = torch.load(out / "ckpt.pt", weights_only=False)
    assert ckpt["optimizer"]["state"], "checkpoint carries no optimizer state"
    live = res["raw_model"].state_dict()
    for k, v in ckpt["model"].items():
        assert torch.equal(v, live[k].cpu()), f"{k} differs from the in-memory model"


# ------------------------------------------ 4. weights_only vs 5. resume
def test_weights_only_gives_fresh_optimizer_and_schedule(shards, tmp_path):
    data, _ = shards
    out = tmp_path / "out"
    ti.configure(out_dir=str(out), max_iters=4)
    train_ds, val_ds, meta = datasets(data)
    ti.run(train_ds, val_ds, dict(TINY), meta)

    ti.configure(init_from="weights_only", init_checkpoint=str(out / "ckpt.pt"),
                 out_dir=str(tmp_path / "out2"), max_iters=2)
    model, _, iter_num, best_val_loss, optim_state = ti.make_model(dict(TINY), meta)
    assert iter_num == 0, "weights_only must not restore iter_num"
    assert optim_state is None, "weights_only must not restore optimizer moments"
    assert best_val_loss == 1e9
    assert ti.build_optimizer(model, optim_state).state_dict()["state"] == {}
    assert ti.get_lr(0) < ti.learning_rate, "lr schedule must restart in warmup"

    # ...and the weights really did come from the checkpoint.
    saved = torch.load(out / "ckpt.pt", weights_only=False)["model"]
    got = model.state_dict()
    assert torch.equal(saved["tok_embeddings.weight"], got["tok_embeddings.weight"].cpu())


def test_resume_restores_optimizer_state_and_iter(shards, tmp_path):
    data, _ = shards
    out = tmp_path / "out"
    ti.configure(out_dir=str(out), max_iters=4)
    train_ds, val_ds, meta = datasets(data)
    ti.run(train_ds, val_ds, dict(TINY), meta)
    saved = torch.load(out / "ckpt.pt", weights_only=False)

    ti.configure(init_from="resume", init_checkpoint=str(out / "ckpt.pt"), max_iters=6)
    model, _, iter_num, _, optim_state = ti.make_model(dict(TINY), meta)
    assert iter_num == 4
    assert optim_state is not None, "resume dropped the optimizer state"
    state = ti.build_optimizer(model, optim_state).state_dict()["state"]
    assert state, "optimizer moments were not loaded"
    assert int(state[0]["step"]) == 4, "Adam step counter did not carry over"
    assert torch.equal(state[0]["exp_avg"], saved["optimizer"]["state"][0]["exp_avg"])


def test_resume_runs_only_the_remaining_iters(shards, tmp_path):
    data, _ = shards
    out = tmp_path / "out"
    ti.configure(out_dir=str(out), max_iters=4)
    train_ds, val_ds, meta = datasets(data)
    ti.run(train_ds, val_ds, dict(TINY), meta)

    ti.configure(init_from="resume", init_checkpoint=str(out / "ckpt.pt"), max_iters=6)
    res = ti.run(train_ds, val_ds, dict(TINY), meta)
    assert res["n_updates"] == 2 and res["iter_num"] == 6


def test_resume_default_checkpoint_missing_is_a_clear_error(tmp_path):
    ti.configure(init_from="resume", out_dir=str(tmp_path / "nope"))
    with pytest.raises(SystemExit, match="does not exist"):
        ti.make_model(dict(TINY), None)


def test_weights_only_without_checkpoint_is_a_clear_error():
    ti.configure(init_from="weights_only", init_checkpoint="")
    with pytest.raises(SystemExit, match="needs --init_checkpoint"):
        ti.make_model(dict(TINY), None)


# ------------------------------------------------ 6. compile wrapper export
class _FakeCompiled(torch.nn.Module):
    """Stands in for torch._dynamo OptimizedModule: same `_orig_mod` attribute,
    so state_dict() keys come out `_orig_mod.`-prefixed."""

    def __init__(self, mod):
        super().__init__()
        self._orig_mod = mod


def test_maybe_compile_keeps_the_raw_model_for_export(monkeypatch):
    ti.configure(compile=True, device="cuda")  # device_type=="cuda" gates compile
    monkeypatch.setattr(ti.torch, "compile", _FakeCompiled)
    inner = torch.nn.Linear(2, 2)
    step_model, raw_model = ti.maybe_compile(inner)
    assert isinstance(step_model, _FakeCompiled), "compile was not applied"
    assert raw_model is inner, "raw_model must stay the uncompiled module"
    assert not any(k.startswith("_orig_mod.") for k in raw_model.state_dict())
    assert any(k.startswith("_orig_mod.") for k in step_model.state_dict()), \
        "the fake wrapper should prefix keys -- otherwise this test proves nothing"


def test_save_checkpoint_strips_a_compile_prefix(shards, tmp_path):
    """Belt and braces for checkpoints written before the fix / by other tools."""
    data, _ = shards
    out = tmp_path / "out"
    ti.configure(out_dir=str(out), max_iters=1)
    train_ds, val_ds, meta = datasets(data)
    res = ti.run(train_ds, val_ds, dict(TINY), meta)
    ckpt = torch.load(res["ckpt_path"], weights_only=False)
    assert not any(k.startswith("_orig_mod.") for k in ckpt["model"])
    ckpt["model"] = {"_orig_mod." + k: v for k, v in ckpt["model"].items()}
    torch.save(ckpt, out / "prefixed.pt")
    ti.configure(init_from="weights_only", init_checkpoint=str(out / "prefixed.pt"))
    ti.make_model(dict(TINY), meta)  # must not raise on unexpected keys


# ------------------------------------------------- 7. dataset validation
def test_missing_shards_message_names_the_exporter(tmp_path):
    with pytest.raises(SystemExit, match="export_train.py"):
        ti.MaskedShardDataset(str(tmp_path), "train", SEQ, VOCAB, None)


def test_ids_tgt_length_mismatch(tmp_path, shards):
    data, _ = shards
    np.zeros(SEQ * 3, dtype=np.int32).tofile(os.path.join(data, "train.tgt.bin"))
    with pytest.raises(SystemExit, match="different export_train.py runs"):
        ti.MaskedShardDataset(data, "train", SEQ, VOCAB, None)


def test_token_id_out_of_vocab(tmp_path, shards):
    data, _ = shards
    ids = np.fromfile(os.path.join(data, "train.ids.bin"), dtype=np.uint16)
    ids[7] = VOCAB + 3
    ids.tofile(os.path.join(data, "train.ids.bin"))
    with pytest.raises(SystemExit, match=r"token id 67 but vocab_size=64"):
        ti.MaskedShardDataset(data, "train", SEQ, VOCAB, None)


def test_illegal_negative_target(tmp_path, shards):
    data, _ = shards
    tgt = np.fromfile(os.path.join(data, "train.tgt.bin"), dtype=np.int32)
    tgt[5] = -100  # the other common ignore_index
    tgt.tofile(os.path.join(data, "train.tgt.bin"))
    with pytest.raises(SystemExit, match="only legal negative target is -1"):
        ti.MaskedShardDataset(data, "train", SEQ, VOCAB, None)


def test_fully_masked_split_is_rejected(tmp_path, shards):
    data, _ = shards
    n = np.fromfile(os.path.join(data, "train.ids.bin"), dtype=np.uint16).size
    np.full(n, -1, dtype=np.int32).tofile(os.path.join(data, "train.tgt.bin"))
    with pytest.raises(SystemExit, match="every position is masked"):
        ti.MaskedShardDataset(data, "train", SEQ, VOCAB, None)


def test_seq_len_mismatch_against_meta(shards):
    data, _ = shards
    meta = ti.load_data_meta(data)
    with pytest.raises(SystemExit, match="packed at seq_len=32"):
        ti.MaskedShardDataset(data, "train", 16, VOCAB, meta)


def test_vocab_mismatch_against_meta(shards):
    data, _ = shards
    meta = ti.load_data_meta(data)
    with pytest.raises(SystemExit, match="vocab_size=64.*--vocab_size=128"):
        ti.MaskedShardDataset(data, "train", SEQ, 128, meta)


def test_too_few_tokens_for_one_window(tmp_path):
    d = tmp_path / "short"
    os.makedirs(d)
    np.zeros(SEQ - 1, dtype=np.uint16).tofile(d / "train.ids.bin")
    np.zeros(SEQ - 1, dtype=np.int32).tofile(d / "train.tgt.bin")
    with pytest.raises(SystemExit, match="less than one 32-token window"):
        ti.MaskedShardDataset(str(d), "train", SEQ, VOCAB, None)


def test_batch_shapes_and_dtypes(shards):
    data, _ = shards
    train_ds, _, meta = datasets(data)
    rng = np.random.default_rng(0)
    x, y = train_ds.get_batch(4, "cpu", rng)
    assert x.shape == y.shape == (4, SEQ)
    assert x.dtype == y.dtype == torch.int64
    assert int(x.max()) < VOCAB and int(y.min()) >= -1


# --------------------------------------------- 8. tokenizer <-> checkpoint
def test_checkpoint_records_the_tokenizer_fingerprint(shards, tmp_path):
    data, tok = shards
    ti.configure(out_dir=str(tmp_path / "out"), max_iters=1)
    train_ds, val_ds, meta = datasets(data)
    res = ti.run(train_ds, val_ds, dict(TINY), meta)
    cfg = torch.load(res["ckpt_path"], weights_only=False)["config"]
    assert cfg["tokenizer"]["sha256"] == ti.tokenizer_fingerprint(tok)["sha256"]
    assert cfg["tokenizer"]["path"] == os.path.abspath(tok)


def test_finetune_across_a_tokenizer_change_is_refused(shards, tmp_path):
    """Same vocab_size, different tokenizer: this is the failure that produces
    a checkpoint which looks fine and generates noise."""
    data, _ = shards
    out = tmp_path / "out"
    ti.configure(out_dir=str(out), max_iters=1)
    train_ds, val_ds, meta = datasets(data)
    ti.run(train_ds, val_ds, dict(TINY), meta)

    other_tok = write_tokenizer(tmp_path / "other.model", b"a completely different vocab")
    other_data = write_shards(tmp_path / "data2", other_tok)
    other_meta = ti.load_data_meta(other_data)
    ti.configure(init_from="weights_only", init_checkpoint=str(out / "ckpt.pt"))
    with pytest.raises(SystemExit, match="refusing to pair them"):
        ti.make_model(dict(TINY), other_meta)


def test_matching_tokenizer_is_accepted(shards, tmp_path):
    data, _ = shards
    out = tmp_path / "out"
    ti.configure(out_dir=str(out), max_iters=1)
    train_ds, val_ds, meta = datasets(data)
    ti.run(train_ds, val_ds, dict(TINY), meta)
    ti.configure(init_from="weights_only", init_checkpoint=str(out / "ckpt.pt"))
    model, _, iter_num, _, optim_state = ti.make_model(dict(TINY), meta)
    assert iter_num == 0 and optim_state is None and model is not None


def test_foreign_checkpoint_without_fingerprint_is_refused(shards, tmp_path):
    """A pretrained checkpoint from before this script (e.g. pipe3m-gqa2, which
    was trained on TinyStories' tok2048, not the corpus tok2048-s2) has no
    fingerprint at all -- matching vocab_size must not be treated as proof."""
    data, _ = shards
    out = tmp_path / "out"
    ti.configure(out_dir=str(out), max_iters=1)
    train_ds, val_ds, meta = datasets(data)
    res = ti.run(train_ds, val_ds, dict(TINY), meta)
    ckpt = torch.load(res["ckpt_path"], weights_only=False)
    ckpt["config"].pop("tokenizer")
    torch.save(ckpt, out / "foreign.pt")

    ti.configure(init_from="weights_only", init_checkpoint=str(out / "foreign.pt"))
    with pytest.raises(SystemExit, match="no tokenizer fingerprint"):
        ti.make_model(dict(TINY), meta)
    ti.configure(allow_unverified_tokenizer=True)
    ti.make_model(dict(TINY), meta)  # explicit opt-in works


def test_vocab_size_mismatch_between_checkpoint_and_shards(shards, tmp_path):
    data, _ = shards
    out = tmp_path / "out"
    ti.configure(out_dir=str(out), max_iters=1)
    train_ds, val_ds, meta = datasets(data)
    res = ti.run(train_ds, val_ds, dict(TINY), meta)
    ckpt = torch.load(res["ckpt_path"], weights_only=False)
    ckpt["config"]["vocab_size"] = 2048
    ckpt["config"].pop("tokenizer")
    torch.save(ckpt, out / "v2048.pt")
    ti.configure(init_from="weights_only", init_checkpoint=str(out / "v2048.pt"),
                 allow_unverified_tokenizer=True)
    with pytest.raises(SystemExit, match="different tokenizers"):
        ti.make_model(dict(TINY), meta)


def test_tokenizer_swapped_in_place_after_export_is_rejected(shards):
    """THE regression this file exists for: meta.json's fingerprint is recorded
    at export time, so replacing the tokenizer at the same path afterwards is
    provable -- and proof is never overridable."""
    data, tok = shards
    assert ti.load_data_meta(data)["tokenizer_fingerprint"]["sha256"] == \
        tf.fingerprint_file(tok)["sha256"]

    write_tokenizer(tok, b"a completely different vocabulary, same path")
    with pytest.raises(SystemExit, match="tokenizer file was replaced"):
        ti.load_data_meta(data)
    ti.configure(allow_unverified_tokenizer=True)
    with pytest.raises(SystemExit, match="does not override it"):
        ti.load_data_meta(data)


def test_missing_tokenizer_file_keeps_the_export_time_fingerprint(shards, tmp_path):
    """The file being gone doesn't unmake the provenance: the shards were still
    exported with that tokenizer, and a checkpoint carrying the same
    fingerprint still pairs."""
    data, tok = shards
    out = tmp_path / "out"
    ti.configure(out_dir=str(out), max_iters=1)
    train_ds, val_ds, meta = datasets(data)
    res = ti.run(train_ds, val_ds, dict(TINY), meta)

    recorded = meta["tokenizer_fingerprint"]
    os.remove(tok)
    meta2 = ti.load_data_meta(data)
    assert meta2["tokenizer_fingerprint"] == recorded
    ti.configure(init_from="weights_only", init_checkpoint=res["ckpt_path"])
    ti.make_model(dict(TINY), meta2)  # still provably the same tokenizer


def test_legacy_meta_without_fingerprint_invents_nothing(tmp_path):
    """A meta.json from before export-time fingerprints must NOT be upgraded by
    hashing whatever file its "tokenizer" path points at now -- that would
    manufacture the exact evidence the check is supposed to demand."""
    tok = write_tokenizer(tmp_path / "tok.model")
    data = write_shards(tmp_path / "data", tok, legacy_meta=True)
    # Scratch must require opt-in too, not only resume/weights_only.
    with pytest.raises(SystemExit, match="no tokenizer fingerprint"):
        ti.load_data_meta(data)
    ti.configure(allow_unverified_tokenizer=True)
    meta = ti.load_data_meta(data)
    assert meta["tokenizer_fingerprint"] is None

    out = tmp_path / "out"
    ti.configure(out_dir=str(out), max_iters=1, allow_unverified_tokenizer=True)
    train_ds, val_ds, _ = datasets(data)
    res = ti.run(train_ds, val_ds, dict(TINY), meta)

    ti.configure(init_from="weights_only", init_checkpoint=res["ckpt_path"],
                 allow_unverified_tokenizer=False)
    with pytest.raises(SystemExit, match="no tokenizer fingerprint"):
        ti.make_model(dict(TINY), meta)
    ti.configure(allow_unverified_tokenizer=True)
    ti.make_model(dict(TINY), meta)  # explicit opt-in is the only way through


def test_proven_mismatch_is_not_overridable(shards, tmp_path):
    """allow_unverified_tokenizer covers MISSING provenance, never a mismatch,
    and the message must not offer it as a way out."""
    data, _ = shards
    out = tmp_path / "out"
    ti.configure(out_dir=str(out), max_iters=1)
    train_ds, val_ds, meta = datasets(data)
    ti.run(train_ds, val_ds, dict(TINY), meta)

    other_tok = write_tokenizer(tmp_path / "other.model", b"different vocabulary")
    other_meta = ti.load_data_meta(write_shards(tmp_path / "data2", other_tok))
    ti.configure(init_from="weights_only", init_checkpoint=str(out / "ckpt.pt"),
                 allow_unverified_tokenizer=True)
    with pytest.raises(SystemExit) as e:
        ti.make_model(dict(TINY), other_meta)
    assert "does NOT override a proven mismatch" in str(e.value)


def test_legacy_truncated_checkpoint_fingerprint_still_pairs(shards, tmp_path):
    """Checkpoints written before 2026-09-09 stored sha256[:16]. A 64-bit
    prefix match is accepted deliberately; a shorter one is not evidence."""
    data, tok = shards
    out = tmp_path / "out"
    ti.configure(out_dir=str(out), max_iters=1)
    train_ds, val_ds, meta = datasets(data)
    res = ti.run(train_ds, val_ds, dict(TINY), meta)

    full = tf.fingerprint_file(tok)["sha256"]
    ckpt = torch.load(res["ckpt_path"], weights_only=False)
    ckpt["config"]["tokenizer"] = {"path": tok, "sha256": full[:16],
                                   "size": os.path.getsize(tok)}
    torch.save(ckpt, out / "legacy.pt")
    ti.configure(init_from="weights_only", init_checkpoint=str(out / "legacy.pt"))
    ti.make_model(dict(TINY), meta)  # 16 hex chars is accepted

    ckpt["config"]["tokenizer"]["sha256"] = full[:8]
    torch.save(ckpt, out / "tooshort.pt")
    ti.configure(init_checkpoint=str(out / "tooshort.pt"))
    with pytest.raises(SystemExit, match="no tokenizer fingerprint"):
        ti.make_model(dict(TINY), meta)


def test_truncated_fingerprint_that_disagrees_is_a_mismatch(shards, tmp_path):
    data, tok = shards
    out = tmp_path / "out"
    ti.configure(out_dir=str(out), max_iters=1)
    train_ds, val_ds, meta = datasets(data)
    res = ti.run(train_ds, val_ds, dict(TINY), meta)

    ckpt = torch.load(res["ckpt_path"], weights_only=False)
    ckpt["config"]["tokenizer"] = {"path": tok, "sha256": "0" * 16,
                                   "size": os.path.getsize(tok)}
    torch.save(ckpt, out / "legacy-bad.pt")
    ti.configure(init_from="weights_only", init_checkpoint=str(out / "legacy-bad.pt"))
    with pytest.raises(SystemExit, match="provably different tokenizers"):
        ti.make_model(dict(TINY), meta)


def test_missing_meta_json_blocks_a_finetune(shards, tmp_path):
    data, _ = shards
    out = tmp_path / "out"
    ti.configure(out_dir=str(out), max_iters=1)
    train_ds, val_ds, meta = datasets(data)
    res = ti.run(train_ds, val_ds, dict(TINY), meta)

    os.remove(os.path.join(data, "meta.json"))
    assert ti.load_data_meta(data) is None
    ti.configure(init_from="weights_only", init_checkpoint=res["ckpt_path"])
    with pytest.raises(SystemExit, match="meta.json is missing"):
        ti.make_model(dict(TINY), None)
