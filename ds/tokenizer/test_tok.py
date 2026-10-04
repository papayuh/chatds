#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""C tokenizer (tok.c) vs sentencepiece, byte for byte. One command:

  python3 ds/tokenizer/test_tok.py

CHATDS_KB=<path to KB3 .kb> (default corpus/kb/out/simplewiki.kb when present)
adds every KB context sentence plus KB-retrieved prompts. Builds tok.c for the
host with ASan/UBSan, and for the DS (arm946e-s) when the BlocksDS toolchain is
installed. Exits nonzero on any mismatch."""
import json
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile

import sentencepiece as spm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "corpus"))
sys.path.insert(0, os.path.join(REPO, "tools"))
from tokenizer_fixture import model_path
import chatds_runtime as rt  # noqa: E402
from tokbin import tokbin  # noqa: E402

SMOKE = ["who was george washington", "whats 38 times 47", "sort scores biggest first"]
CC_FLAGS = ["-std=c99", "-O2", "-Wall", "-Wextra", "-Werror"]


def build(tmp):
    exe = os.path.join(tmp, "tok_host")
    subprocess.run(["cc", *CC_FLAGS, "-g", "-fsanitize=address,undefined",
                    "-fno-sanitize-recover=all", os.path.join(HERE, "tok.c"),
                    os.path.join(HERE, "tok_host.c"), "-o", exe], check=True)
    wf = os.environ.get("WONDERFUL_TOOLCHAIN", os.path.expanduser("~/opt/wonderful"))
    arm = os.path.join(wf, "toolchain", "gcc-arm-none-eabi", "bin", "arm-none-eabi-gcc")
    if os.path.exists(arm):
        subprocess.run([arm, *CC_FLAGS, "-mthumb", "-mcpu=arm946e-s", "-c",
                        os.path.join(HERE, "tok.c"), "-o", os.path.join(tmp, "tok.arm.o")],
                       check=True)
        print("test_tok: DS build (arm946e-s thumb) ok")
    else:
        print("test_tok: no BlocksDS toolchain, DS build skipped")
    return exe


def run(exe, tok_path, reqs):
    out = subprocess.run([exe, tok_path], input="\n".join(reqs) + "\n", text=True,
                         capture_output=True, check=True).stdout.split("\n")[:-1]
    assert len(out) == len(reqs), (len(out), len(reqs))
    return out


def kb_records(path):
    b = open(path, "rb").read()
    magic, _, _, _, n_rec, _, _, text_off = struct.unpack_from("<4s7I", b)
    assert magic == b"KB3\0", magic
    off, recs = text_off, []
    for _ in range(n_rec):
        recs.append(b[off + 1:off + 1 + b[off]])
        off += 1 + b[off]
    return recs


def corpus(sp, kb):
    """name -> list of byte strings."""
    rnd = random.Random(1234)
    c = {}
    uni = ["é ü ñ ç", "日本語のテキスト", "🙂👍🏽 👨‍👩‍👧", "é ä", "שלום مرحبا",
           "▁", "a▁▁b", "　x y", "ﬁ Ω", "﻿bom", "\U0010ffff"]
    bad = [b"\xe2\x82", b"\xed\xa0\x80", b"\xc0\x80", b"\xf4\x90\x80\x80", b"\x80", b"\xf0\x9f\x99",
           b"\xf8\x88\x80\x80\x80", b"\xe0\x80\x80", b"\xc2", b"\xe2\x82A", b"\xff\xfe", b"a\xffb"]
    c["edge"] = [b"", b" ", b"  ", b" " * 300, b"\t", b"\n", b"\r\n", b"a\x00b", b"<unk>", b"<s>",
                 b"</s>", b"<0x41>", b"\n<s>\n", b"Q:", b"\nA:", b" a", b"a ", b"x" * 500,
                 b"3.14159265358979 * 2", b"don't won't can't"] + [bytes([i]) for i in range(256)]
    c["unicode"] = [s.encode() for s in uni] + bad
    long = " ".join(r.decode() for r in kb_records(kb)[:400]) if kb else \
        " ".join(sp.id_to_piece(i).replace("▁", " ") for i in range(259, sp.get_piece_size()))
    c["max-length"] = [long[:4096].encode(), (long * 3)[:12000].encode()]
    pieces = [sp.id_to_piece(i).replace("▁", " ") for i in range(sp.get_piece_size())]
    c["pieces"] = [p.encode() for p in pieces] + [(pieces[i] + pieces[i + 1]).encode()
                                                  for i in range(259, len(pieces) - 1)]
    qa = [(q, None) for q in SMOKE]
    for f in sorted(os.listdir(os.path.join(REPO, "eval"))):
        if f.endswith(".jsonl"):
            for line in open(os.path.join(REPO, "eval", f)):
                r = json.loads(line)
                qa.append((r["prompt"], r.get("answer")))
    prompts = []
    for q, a in qa:
        ctx = rt.retrieve(q, kb) if kb else None
        prompts += [q, rt.format_prompt(q), rt.format_prompt(q, ctx or "a short fact.")]
        if a:
            prompts += [a, " " + a]
    c["prompts"] = [p.encode() for p in prompts]
    if kb:
        recs = kb_records(kb)
        c["kb"] = recs + [rt.format_prompt(SMOKE[0], r.decode()).encode() for r in recs[::50]]
    alphabet = [p for p in pieces[259:] if len(p) == 1] + list(" \t\n.,!?'\"-") + ["▁"]
    fuzz = []
    for _ in range(20000):
        n, k = rnd.randrange(0, 60), rnd.random()
        if k < 0.6:
            fuzz.append("".join(rnd.choice(alphabet) for _ in range(n)).encode())
        elif k < 0.85:
            cps = [rnd.choice([rnd.randrange(0x20, 0x7f), rnd.randrange(0xa0, 0xd800),
                               rnd.randrange(0xe000, 0x110000)]) for _ in range(n)]
            fuzz.append("".join(map(chr, cps)).encode())
        else:
            fuzz.append(bytes(rnd.randrange(256) for _ in range(n)))
    c["fuzz"] = fuzz
    return c


def check_model(exe, tmp, model, kb):
    sp = spm.SentencePieceProcessor(model_file=model)
    blob = tokbin(model)
    tok_path = os.path.join(tmp, os.path.basename(model) + ".bin")
    open(tok_path, "wb").write(blob)
    fails = 0
    seen = set()

    def fail(what, got, want):
        nonlocal fails
        fails += 1
        if fails <= 10:
            print(f"  FAIL {model} {what}\n    got  {got}\n    want {want}")

    # encode
    texts_by_name = corpus(sp, kb)
    for name, texts in texts_by_name.items():
        texts = list(dict.fromkeys(texts))
        reqs = [f"E {b} {4 * len(t) + 8} {t.hex()}" for t in texts for b in (0, 1)]
        for t, (r0, r1) in zip(texts, zip(*[iter(run(exe, tok_path, reqs))] * 2)):
            want = sp.encode(t)
            for got, w in ((r0, want), (r1, [1] + want)):
                got = [int(x) for x in got.split()]
                if got != [len(w)] + w:
                    fail(f"encode {t!r}", got[1:], w)
            seen.add(tuple(want))
        print(f"test_tok: {model} encode {name}: {len(texts)} texts")
    # truncation: full count returned, first max ids written
    long = texts_by_name["max-length"][1]
    want = sp.encode(long)
    got = [int(x) for x in run(exe, tok_path, [f"E 1 256 {long.hex()}"])[0].split()]
    if got != [len(want) + 1] + ([1] + want)[:256]:
        fail("encode max=256", got[:4], [len(want) + 1, 1] + want[:2])

    # decode
    rnd = random.Random(99)
    v = sp.get_piece_size()
    seqs = [list(s) for s in seen] + [[i] for i in range(v)] + [[1, i] for i in range(v)] + [[]]
    for _ in range(20000):
        pool = rnd.choice([range(v), range(3, 259), [0, 1, 2] + list(range(3, 259)) + list(range(259, v))])
        seqs.append([rnd.choice(pool) for _ in range(rnd.randrange(1, 12))])
    reqs = [f"D {64 * len(s) + 8} {' '.join(map(str, s))}" for s in seqs]
    for s, r in zip(seqs, run(exe, tok_path, reqs)):
        want = sp.decode(s).encode()
        ret, _, h = r.partition(" ")
        if int(ret) != len(want) or bytes.fromhex(h) != want:
            fail(f"decode {s}", (ret, bytes.fromhex(h)), want)
    print(f"test_tok: {model} decode: {len(seqs)} id sequences")
    # bad ids, snprintf-style truncation
    r = run(exe, tok_path, [f"D 100 5 {v}", "D 100 -1", f"D 4 {' '.join(map(str, sp.encode('hello world')))}"])
    if r[0] != "-1 " or r[1] != "-1 " or r[2] != "11 " + b"hel".hex():
        fail("decode bad id / cap", r, ["-1 ", "-1 ", "11 " + b"hel".hex()])

    # tok_piece
    reqs = [f"P {i}" for i in range(-1, v + 1)]
    for i, r in zip(range(-1, v + 1), run(exe, tok_path, reqs)):
        if i < 0 or i >= v:
            want = "null"
        elif i in (1, 2):
            want = ""
        elif i == 0:
            want = sp.decode([0]).encode().hex()
        elif i < 259:
            want = bytes([i - 3]).hex()
        else:
            want = sp.id_to_piece(i).replace("▁", " ").encode().hex()
        if r != want:
            fail(f"piece {i}", r, want)

    # malformed tok.bin must be refused
    swapped = bytearray(blob)
    swapped[swapped.index(b"<0x00>") + 4] = ord("1")
    for what, data in [("empty", b""), ("truncated", blob[:-3]), ("max_len 0", b"\0\0\0\0" + blob[4:]),
                       ("byte piece", bytes(swapped)), ("short vocab", blob[:200])]:
        p = os.path.join(tmp, "bad.bin")
        open(p, "wb").write(data)
        rc = subprocess.run([exe, p], input="", capture_output=True).returncode
        if rc != 2:
            fail(f"load {what}", rc, 2)
    return fails


def main():
    kb = os.environ.get("CHATDS_KB") or os.path.join(REPO, "corpus", "kb", "out", "simplewiki.kb")
    if not os.path.exists(kb):
        print(f"test_tok: no KB at {kb}, KB corpus skipped (set CHATDS_KB)")
        kb = None
    tmp = tempfile.mkdtemp(prefix="test_tok.")
    try:
        exe = build(tmp)
        models = [str(model_path(size)) for size in (2048, 1024)]
        fails = sum(check_model(exe, tmp, m, kb if i == 0 else None)
                    for i, m in enumerate(models))
    finally:
        shutil.rmtree(tmp)
    print(f"test_tok: {'FAIL ' + str(fails) if fails else 'PASS'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
