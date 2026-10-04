# SPDX-License-Identifier: MIT
"""Small project-authored tokenizer fixtures generated locally, never vendored."""
import atexit
from functools import lru_cache
from pathlib import Path
import random
import shutil
import string
import tempfile

from train_tokenizer import train


@lru_cache(maxsize=1)
def model_directory():
    directory = Path(tempfile.mkdtemp(prefix='chatds-test-tokenizers-'))
    atexit.register(shutil.rmtree, directory, ignore_errors=True)
    corpus = directory / 'synthetic.txt'
    rng = random.Random(156)
    base = ['hello world', 'The cat sat on the mat.',
            'How do I print in Python?', "print('hi')", 'How do I reverse a list?',
            'xs[::-1]', 'How do I add two numbers?', 'a + b',
            'C: George was first.\nQ: who was george washington\nA: a president.',
            'Q: make name all caps\nA: name.upper()']
    lines = base * 150
    for _ in range(3000):
        lines.append(' '.join(''.join(rng.choice(string.ascii_lowercase)
                                     for _ in range(rng.randrange(3, 12))) for _ in range(8)))
    corpus.write_text('\n'.join(lines) + '\n')
    for size in (1024, 2048):
        train(corpus, directory / f'tok{size}-synthetic', size)
    return directory


def model_path(size=2048):
    return model_directory() / f'tok{size}-synthetic.model'
