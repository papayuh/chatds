#!/usr/bin/env python3
"""Price a tokenizer against the frozen eval suite.

  tokens_per_answer.py eval/suite-v1.jsonl tok512.bin tok1024.bin ...
  tokens_per_answer.py eval/suite-v1.jsonl spm4096.model      # needs sentencepiece

`.bin` = llama2.c tokenizer.bin (parsed here, stdlib only): int32 max_token_length,
then vocab_size records of [float32 score][int32 len][bytes]. Vocab size is inferred
from the file length. `.model` = sentencepiece, used only if the module imports.

Prints, per tokenizer: tokens for every prompt and every canonical answer.
Answer tokens are what bytes-per-correct-answer is multiplied by, so a small vocab
that is cheap per token can still lose here.
"""
import json, struct, sys
from collections import defaultdict


class Llama2cTokenizer:
    """Greedy score-ordered BPE merge, same algorithm as llama2.c encode()."""

    def __init__(self, path):
        buf = open(path, "rb").read()
        self.max_token_length = struct.unpack_from("<i", buf, 0)[0]
        off, self.vocab, self.scores = 4, [], []
        while off < len(buf):
            score, ln = struct.unpack_from("<fi", buf, off)
            off += 8
            self.vocab.append(buf[off:off + ln])
            self.scores.append(score)
            off += ln
        self.id = {t: i for i, t in enumerate(self.vocab)}

    def encode(self, text, bos=True):
        toks = [1] if bos else []
        if text:
            toks.append(self.id.get(b" ", 0) or self.id.get(b"\xe2\x96\x81", 0))
        for ch in text:  # per unicode char, byte-fallback if unknown
            b = ch.encode("utf-8")
            i = self.id.get(b)
            toks.append(i) if i is not None else toks.extend(x + 3 for x in b)
        while True:
            best, best_i = -1e10, -1
            for i in range(len(toks) - 1):
                m = self.id.get(self.vocab[toks[i]] + self.vocab[toks[i + 1]])
                if m is not None and self.scores[m] > best:
                    best, best_i = self.scores[m], i
            if best_i < 0:
                return toks
            toks[best_i] = self.id[self.vocab[toks[best_i]] + self.vocab[toks[best_i + 1]]]
            del toks[best_i + 1]


class SpmTokenizer:
    def __init__(self, path):
        import sentencepiece
        self.sp = sentencepiece.SentencePieceProcessor(model_file=path)
        self.vocab = range(self.sp.get_piece_size())

    def encode(self, text, bos=True):
        return self.sp.encode(text, add_bos=bos)


def stats(ns):
    ns = sorted(ns)
    return sum(ns) / len(ns), ns[len(ns) // 2], ns[-1]


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    suite = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    print("%-24s %6s %26s %26s %8s" % ("tokenizer", "vocab", "prompt tok mean/p50/max",
                                       "answer tok mean/p50/max", "chr/tok"))
    for path in sys.argv[2:]:
        tk = SpmTokenizer(path) if path.endswith(".model") else Llama2cTokenizer(path)
        p = [len(tk.encode(i["prompt"])) - 1 for i in suite]      # drop BOS
        a = [len(tk.encode(i["answer"], bos=False)) for i in suite]
        chars = sum(len(i["answer"]) for i in suite)
        pm, pp, px = stats(p)
        am, ap, ax = stats(a)
        print("%-24s %6d %26s %26s %8.2f" % (
            path.split("/")[-1], len(tk.vocab),
            "%.1f / %d / %d" % (pm, pp, px), "%.1f / %d / %d" % (am, ap, ax),
            chars / sum(a)))
        per = defaultdict(list)
        for i, n in zip(suite, a):
            per[i["category"]].append(n)
        print("   answer tokens by category: " + "  ".join(
            "%s %.1f" % (c, sum(v) / len(v)) for c, v in sorted(per.items())))


if __name__ == "__main__":
    main()
