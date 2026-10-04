#!/usr/bin/env python3
"""D16 deliverable 1: pairs (JSONL prompt/answer) -> packed training shards.

  corpus/export_train.py --pairs corpus/results/full-train.jsonl \\
      --tokenizer path/to/your-tokenizer.model --out-dir train/data/instruct1024

Prompt serialization is pinned in tools/prompt_format.py and used verbatim
here -- see tools/prompt_format.py. For each pair:

    prompt_ids = [BOS] + encode("Q: " + question + "\\nA:")
    answer_ids = encode(" " + answer) + [EOS]
    seq        = prompt_ids + answer_ids

Rejected outright (never packed) if len(prompt_ids) - 1 + len(answer_ids) >
--seq-len (this is exactly "prompt_tokens - 1 + answer_tokens + 1", since
prompt_ids already carries BOS and answer_ids already carries EOS). Note
what that ACCEPTS: len(seq) - 1 <= seq_len, i.e. len(seq) may be seq_len+1.
In independent packing that is the boundary case where EOS cannot fit in
the input window but is still supervised as the target of the window's
final input position -- build_example must never require len(seq) <=
seq_len, or the last answer token would be silently dropped.

Sequences are packed back-to-back into one flat token stream per split
(train/val), then chopped into fixed --seq-len windows -- no padding, the
trailing partial window is dropped. Two parallel flat arrays are written:

  <out>/{split}.ids.bin   uint16   token ids
  <out>/{split}.tgt.bin   int32    -1 (ignore_index) on every non-supervised
                                    position, else the next token id

--packing independent (the R2 experiment) instead gives EVERY example its
own window, exactly as standalone inference conditions on it: BOS at
position 0, no other question anywhere in the context, the example's own
tokens at absolute positions 0..len(seq)-2, and the remaining input slots
padded with PAD_ID. Same two flat files, same window count == example
count, so train_instruct.py loads both packings identically. See the PAD_ID
comment for why the pad is harmless.

<out>/meta.json records `tokenizer_fingerprint`: the sha256 + size of the
tokenizer bytes AS LOADED HERE, at export time. That is the pairing evidence
train/train_instruct.py checks; recomputing a hash later from whatever file
sits at the same path proves nothing, because the file can be replaced in
place after the shards are written. We re-read the tokenizer before writing
meta.json and abort if it changed mid-export.

A position is supervised iff the NEXT token is part of an answer (including
its terminating EOS) -- i.e. answer-only loss, first answer token through
EOS. The boundary position at the end of every packed example (predicting
across into the next example's BOS) is always masked, instruct or replay
alike (ponytail: one rule, no special-casing replay's "what follows a
story" -- we never train on predicting BOS from context).

--replay mixes in TinyStories rows as FULLY-supervised rows (every position
unmasked): reads uint16 shards from --replay-dir (default
train/data/tok<vocab_size>, i.e. wherever TinyStories was pretokenized with
THIS SAME tokenizer -- D16's re-pretrain step produces exactly that
directory; pointing this at shards tokenized with a different tokenizer
silently corrupts training, nothing here can detect that).
"""
import argparse, glob, json, os, random, struct, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, os.path.join(HERE, "..", "reference", "ds-llm"))
from prompt_format import format_prompt, format_answer, BOS_ID, EOS_ID  # noqa: E402
from tok_fingerprint import fingerprint_bytes, compare, MATCH  # noqa: E402
from tokenizer import Tokenizer  # noqa: E402

# Input-slot pad for --packing independent. Id 0 (the UNK piece): the one id
# whose presence in a window can never be confused with example structure
# (BOS stays unique to position 0). It is harmless by construction, not by
# luck: pads are written ONLY after the example's last input token, so under
# causal attention no supervised position -- all of which precede them -- can
# ever attend to a pad, and every pad position's target is -1, so no pad is
# ever a label either. They exist purely to keep the rectangular window.
PAD_ID = 0


def load_tokenizer(path):
    """(tok, fingerprint) where the fingerprint covers the exact bytes on disk
    at load time -- not a later re-read of the same path."""
    with open(path, "rb") as f:
        blob = f.read()
    tok = Tokenizer(path)
    # Tokenizer opens the path itself. Prove that processor loaded the bytes
    # we hashed, including a swap-and-restore between these two reads.
    if tok.sp_model.serialized_model_proto() != blob:
        sys.exit(f"export_train: loaded tokenizer differs from hashed bytes at {path}")
    return tok, fingerprint_bytes(blob, path)


def assert_tokenizer_unchanged(path, fp):
    """Abort if the tokenizer file was replaced while we were packing. The
    recorded fingerprint has to describe the tokenizer that actually produced
    these ids; writing it after an in-place swap would record a lie that every
    downstream pairing check would then happily accept."""
    with open(path, "rb") as f:
        now = fingerprint_bytes(f.read(), path)
    if compare(fp, now) != MATCH:
        sys.exit(f"export_train: {path} changed while exporting (sha "
                 f"{fp['sha256'][:16]} -> {now['sha256'][:16]}). The shards in "
                 "--out-dir are unusable; delete them and re-run with a stable "
                 "tokenizer file.")


def build_example(tok, question, answer, seq_len, context=None):
    """Returns (seq, n_prompt) or None if it doesn't fit seq_len. `context`
    (chatds) adds a leading "C: ...\\n" line; None keeps the D16 bytes."""
    prompt_ids = tok.encode(format_prompt(question, context), bos=True, eos=False)
    answer_ids = tok.encode(format_answer(answer), bos=False, eos=False) + [EOS_ID]
    if len(prompt_ids) - 1 + len(answer_ids) > seq_len:
        return None
    return prompt_ids + answer_ids, len(prompt_ids)


def targets_for(seq, n_prompt, fully_supervised=False):
    """target[k] = seq[k+1] if supervised else -1 (ignore_index)."""
    n = len(seq)
    tgt = [-1] * n
    for k in range(n - 1):
        if fully_supervised or k + 1 >= n_prompt:
            tgt[k] = seq[k + 1]
    return tgt


def pack_independent(rows, seq_len):
    """One example per window: ids = seq[:min(len(seq), seq_len)] + PAD_ID
    tail, targets = next token from the first answer token through EOS,
    -1 on prompt and pad positions.

    When len(seq) <= seq_len the WHOLE sequence including its EOS is in the
    input (the EOS slot's own target is -1: nothing after EOS is ever
    supervised, so no cross-example targets by construction). The boundary
    case len(seq) == seq_len+1 -- which build_example deliberately accepts --
    is the one place EOS cannot fit in the input: the window is exactly full
    and its FINAL position's target is EOS, i.e. the model predicts EOS from
    the last answer token without EOS ever sitting in the input window.

    Supervised positions are k in [n_prompt-1, len(seq)-2] with
    tgt[k] = seq[k+1] -- every answer token INCLUDING EOS is a target exactly
    once, and len(seq)-1 <= seq_len guarantees the range fits the window.
    PAD_ID slots exist only after the example's last real token, so under
    causal attention no supervised position can ever attend to a pad (see
    PAD_ID's comment).

    Returns (ids, tgt, unpadded, supervised) as (n, seq_len) arrays written
    row-major -- byte-identical layout to the stream path's flat files."""
    n = len(rows)
    ids = np.full((n, seq_len), PAD_ID, dtype=np.uint16)
    tgt = np.full((n, seq_len), -1, dtype=np.int32)
    unpadded = supervised = 0
    for i, (seq, n_prompt) in enumerate(rows):
        L = min(len(seq), seq_len)
        ids[i, :L] = seq[:L]
        tgt[i, n_prompt - 1:len(seq) - 1] = seq[n_prompt:len(seq)]
        unpadded += L
        supervised += len(seq) - n_prompt
    return ids, tgt, unpadded, supervised


def load_replay_rows(replay_dir, seq_len, rng, want):
    """Split TinyStories dataNN.bin shards (uint16, BOS-separated) into
    per-story rows, return up to `want` of them shuffled, each as a plain
    python list of ints, truncated at seq_len (dropped if it doesn't start
    with BOS -- can't happen with well-formed shards)."""
    shards = sorted(glob.glob(os.path.join(replay_dir, "*.bin")))
    if not shards:
        print(f"export_train: --replay-dir {replay_dir} has no .bin shards", file=sys.stderr)
        return []
    rows = []
    for shard in shards:
        arr = np.memmap(shard, dtype=np.uint16, mode="r")
        bos_pos = np.flatnonzero(arr == BOS_ID)
        for i, start in enumerate(bos_pos):
            end = bos_pos[i + 1] if i + 1 < len(bos_pos) else len(arr)
            if end - start <= seq_len:
                rows.append(arr[start:end].astype(np.int64).tolist())
        if len(rows) >= want * 3:  # plenty to sample from without reading every shard
            break
    rng.shuffle(rows)
    return rows[:want]


def load_text_rows(tok, path, seq_len, val_rows=2000):
    """Plain text (blank-line separated paragraphs) -> fully supervised rows
    BOS + paragraphs back to back (each paragraph tokenized alone) + EOS, at most
    seq_len tokens (a single longer paragraph is kept whole). Returns (train, val)."""
    rows, cur = [], []
    def flush():
        if cur:
            rows.append([BOS_ID] + cur + [EOS_ID])
    for para in open(path).read().split("\n\n"):
        ids = tok.encode(para.strip(), bos=False, eos=False) if para.strip() else []
        if cur and len(cur) + len(ids) > seq_len - 2:
            flush(); cur = []
        cur = cur + ids
    flush()
    random.Random(0).shuffle(rows)
    return rows[val_rows:], rows[:val_rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True, help="JSONL with prompt/answer fields")
    ap.add_argument("--tokenizer", required=True, help="path to .model")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--packing", choices=("stream", "independent"), default="stream",
                    help="stream: back-to-back packing into shared windows "
                         "(default, unchanged semantics). independent: one "
                         "example per window from its own BOS/position 0, pad "
                         "tail (R2 experiment)")
    ap.add_argument("--val-count", type=int, default=2000,
                     help="hold out this many accepted pairs (last N) for val")
    ap.add_argument("--replay-pct", type=float, default=0.0,
                     help="0-100, TinyStories rows as %% of final row count")
    ap.add_argument("--replay-dir", default=None,
                     help="default: train/data/tok<vocab_size> next to this repo's train/")
    ap.add_argument("--text", default=None, help="plain text file (pretraining): fully supervised rows, stream packing only")
    ap.add_argument("--text-share", type=float, default=80.0, help="with --text: text's share of train tokens, the rest are the pairs (cycled/cut to fit)")
    ap.add_argument("--limit", type=int, default=None, help="only read the first N pairs")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--selftest", action="store_true",
                     help="after writing, run the round-trip + one-batch checks and exit")
    a = ap.parse_args()

    # Independent windows carry one example each; TinyStories replay rows are
    # full stories with no prompt/answer mask, and splicing one into a window
    # would put OTHER content into an example's context -- the exact thing
    # this mode exists to remove. R2 uses no replay anyway.
    if a.packing == "independent" and a.replay_pct > 0:
        sys.exit("export_train: --packing independent does not support "
                 "--replay-pct > 0 (replay rows would reintroduce packed "
                 "neighbours into a mode whose point is their absence). "
                 "Re-export with --replay-pct 0.")

    tok, tok_fp = load_tokenizer(a.tokenizer)
    assert tok.bos_id == BOS_ID and tok.eos_id == EOS_ID, (
        f"pinned special ids violated: bos={tok.bos_id} eos={tok.eos_id}")

    rng = random.Random(a.seed)
    examples = []  # list of (seq, n_prompt, formatted prompt, answer)
    n_rejected = 0
    with open(a.pairs) as f:
        for n, line in enumerate(f):
            if a.limit and n >= a.limit:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            built = build_example(tok, rec["prompt"], rec["answer"], a.seq_len, rec.get("context"))
            if built is None:
                n_rejected += 1
                continue
            seq, n_prompt = built
            examples.append((seq, n_prompt, format_prompt(rec["prompt"], rec.get("context")), rec["answer"]))

    print(f"export_train: {len(examples)} accepted, {n_rejected} rejected (overlong)")
    rng.shuffle(examples)

    val_n = min(a.val_count, max(0, len(examples) // 20))
    val_examples, train_examples = examples[:val_n], examples[val_n:]

    replay_dir = a.replay_dir or os.path.join(HERE, "..", "train", "data", f"tok{tok.n_words}")
    replay_rows = []
    if a.replay_pct > 0:
        n_train = len(train_examples)
        want = int(round(n_train * a.replay_pct / (100.0 - a.replay_pct)))
        replay_rows = load_replay_rows(replay_dir, a.seq_len, rng, want)
        print(f"export_train: replay wanted {want}, got {len(replay_rows)} rows from {replay_dir}")

    val_replay = []
    if a.text:
        if a.packing != "stream" or a.replay_pct > 0:
            sys.exit("export_train: --text needs --packing stream and no --replay-pct")
        replay_rows, val_replay = load_text_rows(tok, a.text, a.seq_len)
        text_tok = sum(map(len, replay_rows))
        want_tok = text_tok * (100 - a.text_share) / a.text_share
        pairs, got = [], 0
        while got < want_tok:  # cycle the pairs if there are too few, cut at the budget
            for e in train_examples:
                pairs.append(e); got += len(e[0])
                if got >= want_tok:
                    break
        train_examples = pairs
        print(f"export_train: text rows {len(replay_rows)} ({text_tok} tokens), pair rows {len(pairs)} ({got} tokens)")

    os.makedirs(a.out_dir, exist_ok=True)
    # Invalidate a prior successful export BEFORE replacing any shard. If this
    # re-export fails, an old manifest must not authenticate newly written ids.
    meta_path = os.path.join(a.out_dir, "meta.json")
    if os.path.exists(meta_path):
        os.unlink(meta_path)

    def pack_and_write(split_examples, split_replay, split_name):
        counts = {}
        if a.packing == "independent":
            rows = [(seq, n_prompt) for seq, n_prompt, _, _ in split_examples]
            rng.shuffle(rows)
            ids_arr, tgt_arr, unpadded, supervised = pack_independent(rows, a.seq_len)
            padded = ids_arr.size - unpadded
            counts = {"unpadded_input_tokens": unpadded,
                      "supervised_positions": supervised,
                      "padded_positions": padded}
            print(f"export_train: {split_name}: {len(rows)} rows -> {len(rows)} windows "
                  f"x {a.seq_len} (unpadded {unpadded}, supervised {supervised}, "
                  f"padded {padded})")
        else:
            rows = [(seq, n_prompt, False) for seq, n_prompt, _, _ in split_examples]
            rows += [(row, 0, True) for row in split_replay]
            rng.shuffle(rows)
            ids_stream, tgt_stream = [], []
            for seq, n_prompt, fully in rows:
                ids_stream.extend(seq)
                tgt_stream.extend(targets_for(seq, n_prompt, fully))

            n_windows = len(ids_stream) // a.seq_len
            n_keep = n_windows * a.seq_len
            ids_arr = np.array(ids_stream[:n_keep], dtype=np.uint16)
            tgt_arr = np.array(tgt_stream[:n_keep], dtype=np.int32)
            counts = {"unpadded_input_tokens": int(n_keep),
                      "supervised_positions": int((tgt_arr != -1).sum()),
                      "padded_positions": 0}
            print(f"export_train: {split_name}: {len(rows)} rows -> {n_windows} windows "
                  f"of {a.seq_len} ({n_keep} tokens, "
                  f"{counts['supervised_positions']} supervised)")

        ids_arr.tofile(os.path.join(a.out_dir, f"{split_name}.ids.bin"))
        tgt_arr.tofile(os.path.join(a.out_dir, f"{split_name}.tgt.bin"))
        return ids_arr, tgt_arr, counts

    train_ids, train_tgt, train_counts = pack_and_write(train_examples, replay_rows, "train")
    val_ids, val_tgt, val_counts = pack_and_write(val_examples, val_replay, "val")

    assert_tokenizer_unchanged(a.tokenizer, tok_fp)
    meta = {
        "seq_len": a.seq_len,
        "packing": a.packing,
        "vocab_size": tok.n_words,
        "tokenizer": os.path.abspath(a.tokenizer),
        "tokenizer_fingerprint": tok_fp,
        "accepted": len(examples), "rejected": n_rejected,
        "replay_rows": len(replay_rows), "replay_pct_requested": a.replay_pct,
        "train_windows": train_ids.size // a.seq_len,
        "val_windows": val_ids.size // a.seq_len,
        # Real signal, not padded positions: unpadded counts actual input
        # tokens, supervised counts loss positions -- an epoch over these
        # windows is NOT windows*seq_len tokens of training signal when
        # padding is present.
        "train_unpadded_input_tokens": train_counts["unpadded_input_tokens"],
        "train_supervised_positions": train_counts["supervised_positions"],
        "train_padded_positions": train_counts["padded_positions"],
        "val_unpadded_input_tokens": val_counts["unpadded_input_tokens"],
        "val_supervised_positions": val_counts["supervised_positions"],
        "val_padded_positions": val_counts["padded_positions"],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print("export_train: meta ->", meta)

    if a.selftest:
        sys.exit(selftest(tok, examples, train_ids, train_tgt, a.seq_len, a.packing))


def selftest(tok, examples, ids_arr, tgt_arr, seq_len, packing="stream"):
    ok = True

    # 1) decode round-trip on 50 random accepted pairs.
    sample = random.Random(0).sample(examples, min(50, len(examples)))
    for seq, n_prompt, want_prompt, answer in sample:
        answer_ids = seq[n_prompt:-1]  # exclude the trailing EOS
        decoded_prompt = tok.decode(seq[1:n_prompt])   # skip BOS, sp.decode ignores it anyway
        decoded_answer = tok.decode(answer_ids)
        want_answer = format_answer(answer)
        if decoded_prompt != want_prompt or decoded_answer != want_answer:
            ok = False
            print(f"SELFTEST FAIL round-trip: prompt {decoded_prompt!r} != {want_prompt!r}"
                  f" or answer {decoded_answer!r} != {want_answer!r}", file=sys.stderr)
    if ok:
        print(f"SELFTEST round-trip: {len(sample)}/{len(sample)} decoded exactly")

    if packing == "independent":
        # Per-WINDOW alignment: reshape first, then shift inside each row.
        # The flattened stream check below would compare across window
        # boundaries (last slot of window i vs first of window i+1) and could
        # pass on data whose windows are internally misaligned.
        assert ids_arr.size % seq_len == 0
        W = ids_arr.size // seq_len
        iw = ids_arr.reshape(W, seq_len)
        tw = tgt_arr.reshape(W, seq_len)

        bos_first = bool((iw[:, 0] == BOS_ID).all())
        bos_once = bool((iw == BOS_ID).sum(axis=1).max() == 1) if W else True
        valid = tw[:, :-1] != -1
        shifted_ok = np.array_equal(
            tw[:, :-1][valid].astype(np.int64), iw[:, 1:][valid].astype(np.int64))

        # EOS is the last supervised target of every window (every answer
        # token INCLUDING EOS is supervised), and everything AFTER that last
        # supervised slot -- the EOS-in-input slot plus the pad tail -- must
        # be masked: nothing after EOS is ever a target. A window whose
        # FINAL slot is supervised is the len(seq) == seq_len+1 boundary
        # case and must target EOS there.
        sup = tw != -1
        any_sup = sup.any(axis=1)
        last_sup_idx = seq_len - 1 - np.argmax(sup[:, ::-1], axis=1)
        last_tgt = tw[np.arange(W), last_sup_idx]
        eos_last = bool((last_tgt[any_sup] == EOS_ID).all()) if W else True
        cols = np.arange(seq_len)
        tail = cols[None, :] > last_sup_idx[:, None]
        tail_masked = bool((tw[tail] == -1).all()) if W else True

        print(f"SELFTEST independent: {W} windows; first-slot BOS "
              f"{'PASS' if bos_first else 'FAIL'}, single BOS/window "
              f"{'PASS' if bos_once else 'FAIL'}, per-window shift "
              f"{'PASS' if shifted_ok else 'FAIL'} ({int(valid.sum())} "
              f"supervised / {tw.size} positions), EOS-final "
              f"{'PASS' if eos_last else 'FAIL'}, tail-masked "
              f"{'PASS' if tail_masked else 'FAIL'}")
        ok = ok and bos_first and bos_once and shifted_ok and eos_last and tail_masked
    else:
        # 2) one-batch assertion: targets are input ids shifted by one on
        # answer positions, -1 elsewhere -- checked over the whole built
        # stream, which is exactly what any batch drawn from it is a slice
        # of. (Stream mode only: independent windows must never be checked
        # flattened across their padded boundaries -- see above.)
        valid = tgt_arr[:-1] != -1
        shifted_ok = np.array_equal(tgt_arr[:-1][valid].astype(np.int64),
                                    ids_arr[1:][valid].astype(np.int64))
        print(f"SELFTEST shift check: {'PASS' if shifted_ok else 'FAIL'} "
          f"({int(valid.sum())} supervised / {len(tgt_arr)} total positions)")
    ok = ok and bool(shifted_ok)

    if ok:
        print("SELFTEST PASS")
        return 0
    print("SELFTEST FAIL", file=sys.stderr)
    return 1


if __name__ == "__main__":
    main()
