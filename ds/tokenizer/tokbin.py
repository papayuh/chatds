#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export a SentencePiece .model to tok.bin (layout in tok.h).

  python3 ds/tokenizer/tokbin.py build/my-tokenizer.model build/tok.bin

Pieces store U+2581 as ' '; BOS/EOS are written as "\\n<s>\\n" / "\\n</s>\\n"."""
import struct
import sys

import sentencepiece as spm


def tokbin(model_path):
    sp = spm.SentencePieceProcessor(model_file=model_path)
    pieces = []
    for i in range(sp.get_piece_size()):
        p = sp.id_to_piece(i)
        if i == sp.bos_id():
            p = "\n<s>\n"
        elif i == sp.eos_id():
            p = "\n</s>\n"
        pieces.append((sp.get_score(i), p.replace("\u2581", " ").encode()))
    return struct.pack("<I", max(len(b) for _, b in pieces)) + b"".join(
        struct.pack("<fI", s, len(b)) + b for s, b in pieces)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    with open(sys.argv[2], "wb") as f:
        f.write(tokbin(sys.argv[1]))
