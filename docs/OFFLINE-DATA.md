# offline data

`corpus/offline_python.py` produces deterministic instruction pairs from
project-authored tasks, without model calls, network or provider credentials.
It wraps 246 tasks in eight phrasings, then filters malformed, duplicate and
evaluation-overlapping pairs. Reference snippets are parsed, never executed.
Syntax validity does not prove semantic correctness or generalization.

```sh
python3 corpus/offline_python.py --out build/offline-python.jsonl --seed 42
python3 corpus/postprocess.py --selftest
```

The frozen suite's hash is checked before contamination filtering. Optional
`--extra-eval FILE` supplies an additional holdout. The generator refuses to
overwrite the original training corpus. Generated corpora are ignored, not
shipped. Exporting training shards needs the optional MIT tokenizer module;
see [training](../train/README.md).

`corpus/kb/` separately builds short contexts from Simple English Wikipedia.
Index data and context excerpts retain CC BY-SA 4.0 terms; include attribution,
source URL, license URL and a note that text was extracted/truncated/indexed.
The fixture `mini.xml` contains short factual test pages, not a full dump.
Treat it conservatively as Wikipedia-derived material. See
[third-party notices](../THIRD_PARTY_NOTICES.md).

TinyStories was used to train the private project's SentencePiece tokenizers,
which are not shipped. Its
CDLA-Sharing-1.0 terms and the status of trained artifacts need review before
redistributing tokenizer/model downloads. DeepSeek teacher outputs were used
in private training; a code license does not license those datasets or weights.
[Public-release audit](PUBLIC_RELEASE_AUDIT.md) separates tracked samples from
ignored training/index/model assets.
