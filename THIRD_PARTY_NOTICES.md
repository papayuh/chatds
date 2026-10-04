# third-party notices

ChatDS code is MIT-licensed; see root `LICENSE`. These notices preserve
third-party attribution. Data licenses and linked-runtime notices are separate
obligations.

## MIT training code

`train/configurator.py` is adapted from
[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT).
Copyright (c) 2022 Andrej Karpathy.

`train/train_instruct.py` is derived from the MIT
[RileyGreiff/ds-llm](https://github.com/RileyGreiff/ds-llm) training fork of
[karpathy/llama2.c](https://github.com/karpathy/llama2.c).
Their LICENSE text identifies Copyright (c) 2023 Andrej.
Optional model/export/tokenizer modules are fetched, not vendored in this tree.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Wikipedia-derived data

Context excerpts in `ds/engine-golden/fixtures.json`, the benchmark archive and
historical result text are derived from Simple English Wikipedia contributors.
Source: https://simple.wikipedia.org/ (article titles appear in the retained
questions/context). The KB fixture `corpus/kb/fixtures/mini.xml` contains short
edited/synthetic test pages; conservatively apply the same attribution to its
article-like text.

License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Text was extracted, markup removed, transliterated to ASCII and truncated for
context records; test fixture pages add synthetic markup. No endorsement by
Wikipedia contributors is implied. These data excerpts are not licensed by
the project's code license. Retain attribution and share-alike terms when
redistributing adapted text.

The private release KB3 used the Simple English Wikipedia pages/articles dump:
https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2
Last-Modified: 2026-09-01 20:27:10 GMT; SHA1:
`1bdd97642b5f511def336cce8afd34996db52d49`.
A generated index must carry its own dump-specific attribution, not assume this
notice identifies a different source dump.

## tokenizer-training data

The private project's SentencePiece models were trained using TinyStories text
plus project/generated instruction material; those models are not shipped. TinyStories is distributed under
[CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/):
https://huggingface.co/datasets/roneneldan/TinyStories
Section 3.5 imposes no obligations on publication of Results, defined in
section 1.11 as computational outputs containing no more than a de minimis
portion of the input data. Whether these mixed-source artifacts qualify, and
their complete training provenance, remain release review items. Neither this notice nor a source-code license grants blanket
rights to training datasets or model artifacts.

## external tools and linked ROM runtime

- BlocksDS/libnds: zlib license; FatFs and picolibc/libm: BSD-style notices.
  These are installed externally, not vendored. Before distributing a ROM,
  include notices for the exact linked ARM9/ARM7 runtime, not only the SDK's
  top-level license directory.
- melonDS: GPLv3; used as a test tool, not included in kits.
- DSpico firmware/DLDI: LNH team; supplied by the user's flashcart, not copied
  into this code-only snapshot.
- PyTorch, NumPy, SentencePiece, GitHub Actions and the ARM toolchain retain
  their upstream licenses; they are dependencies, not bundled source.
