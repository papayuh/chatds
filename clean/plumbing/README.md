# ChatDS plumbing (clean engine)

Product glue between the clean engine (`clean/include/chatds_engine.h`) and the DS app:
KB3 retrieval, the `C:/Q:/A:` prompt, `calc()`, the generation loop with KV/memory
planning and streaming, and card I/O. A C port of `corpus/chatds_runtime.py` and
`corpus/kb/kb.py`; nothing else was read. API and limits: `plumbing.h`.

| file | what |
|---|---|
| `kb.c` | KB2/KB3 reader: one-sector cache, every read bounds-checked, header validated at open |
| `retrieve.c` | gate, n-gram retrieval with fuzzy fallback, prompt |
| `calc.c` | int64 fixed-point `calc()` (no 128-bit math, the ARM9 has none) |
| `generate.c` | `cds_generate`, `cds_session_plan` (shrinks the KV window to fit a memory budget), `cds_ask`; `cds_gen_begin`/`cds_gen_step`/`cds_ask_begin` run the same loop one forward at a time (the DS UI) |
| `io.c` | `run.txt`, file preflight, atomic `out.txt`/`ids.txt`/`ctx.txt`/`calc.txt`/`heap.txt` (status last) |

## Test

```sh
make -C clean/plumbing test                     # host: parity vs Python + C unit tests (ASan/UBSan)
CHATDS_KB=/path/to/simplewiki.kb make -C clean/plumbing test   # also the full KB3
```

`test_parity.py` diffs KB context + prompt byte for byte against the Python reference over
every `eval/*.jsonl` question, the hwkit smoke questions and the runtime's own cases; calc over
the reference cases plus a seeded fuzz. `test_plumbing.c` drives the generation loop, KV planning,
result files and malformed KBs against a stub engine. With the BlocksDS toolchain installed it
also cross-compiles for the ARM9 and fails if any function frame is dynamic or over 1 KiB.

## Wiring on the DS

```c
static int enc(void *t, const char *s, size_t n, int bos, int *ids, int max) {
    return tok_encode(t, s, n, bos, ids, max);   /* ds/tokenizer */
}
static const char *pc(void *t, int id, int *len) { return tok_piece(t, id, len); }
cds_tokenizer tk = {enc, pc, &tok};

cds_session_plan(model, 0, 192, heap_free - reserve, CHATDS_KERNEL_FAST, &session);
cds_ask(session, &tk, have_kb ? &kb : NULL, cfg.instruct, question, &opts, &answer);
cds_publish("fat:/chatds/", &result);
```

Keep `cds_kb` and `cds_answer` static or on the heap (about 3 KiB and 1 KiB), not on the DTCM
stack. Questions over 127 bytes are refused, as the DS editor never produces them.
