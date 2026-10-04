# corpus

Project scripts generate offline Python/bugfix examples, instruction-pair
mixes and optional teacher requests. They filter evaluation contamination and
record tokenizer fingerprints when exporting training shards.

```sh
python3 corpus/offline_python.py --out build/offline-python.jsonl
python3 corpus/curriculum_v2.py --check-suite
python3 corpus/postprocess.py --selftest
python3 corpus/submit_deepseek.py --selftest
```

Teacher submissions need `DEEPSEEK_API_KEY` in the environment or a local `.env`
file; neither belongs in git. `.env.example` contains no key. Offline generation
needs no network or key. Training shard export needs NumPy, SentencePiece and
the optional MIT training dependency described in [train/](../train/README.md).

`kb/` builds a fact index from Simple English Wikipedia. See
[kb/FORMAT.md](kb/FORMAT.md) and [data provenance](../docs/OFFLINE-DATA.md).
Set `CHATDS_USER_AGENT` to your own Wikimedia-compliant contact string when
downloading a dump. Generated training corpora, dumps and indexes are ignored;
the repository contains scripts, evaluation samples and a small KB test fixture,
not the training dataset. Data redistribution rights remain separate from the
code license. See [public-release audit](../docs/PUBLIC_RELEASE_AUDIT.md).
