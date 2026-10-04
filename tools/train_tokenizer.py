#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Train a ChatDS-compatible SentencePiece BPE model from your own text."""
import argparse
from pathlib import Path

import sentencepiece as spm


def train(input_path, prefix, vocab_size):
    """Keep token IDs, whitespace and unknown surface compatible with tok.c."""
    prefix = Path(prefix)
    if any(Path(str(prefix) + suffix).exists() for suffix in ('.model', '.vocab')):
        raise ValueError('refusing to replace an existing tokenizer')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(input_path), model_prefix=str(prefix), model_type='bpe',
        vocab_size=vocab_size, character_coverage=1.0, byte_fallback=True,
        normalization_rule_name='identity', remove_extra_whitespaces=False,
        add_dummy_prefix=True, split_by_whitespace=False, split_by_number=False,
        split_digits=False, allow_whitespace_only_pieces=True,
        unk_id=0, bos_id=1, eos_id=2, pad_id=-1,
        unk_surface=r' \342\201\207 ', num_threads=1,
        shuffle_input_sentence=False, minloglevel=2)
    return Path(str(prefix) + '.model')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='UTF-8 text you have rights to use')
    p.add_argument('--prefix', type=Path, required=True)
    p.add_argument('--vocab-size', type=int, default=2048)
    a = p.parse_args()
    if not 260 <= a.vocab_size <= 65536:
        p.error('--vocab-size must be 260..65536 (and fit the training corpus)')
    print(train(a.input.resolve(strict=True), a.prefix, a.vocab_size))


if __name__ == '__main__':
    main()
