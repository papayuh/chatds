/* SPDX-License-Identifier: MIT
 * ChatDS product plumbing on top of chatds_engine.h: KB3 retrieval, the
 * C:/Q:/A: prompt, calc(), the generation loop with KV/memory planning and
 * streaming, and card I/O (run.txt, preflight, atomic result files).
 *
 * C twin of corpus/chatds_runtime.py + corpus/kb/kb.py, written from those
 * files only. Retrieval, prompt and calc strings must match them byte for
 * byte: clean/plumbing/test_parity.py checks it over the eval/smoke questions.
 * No floats. Engine and tokenizer stay behind their interfaces. */
#ifndef CDS_PLUMBING_H
#define CDS_PLUMBING_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "chatds_engine.h"

#define CDS_BOS 1
#define CDS_EOS 2
#define CDS_QUESTION_MAX 127 /* DS editor limit; longer questions are refused */
#define CDS_KEY_MAX 63       /* KB keys longer than this are never stored */
#define CDS_CTX_CAP 256      /* a record's u8 length + NUL; KB3 records are <= 100 */
/* "C: " ctx "\n" "Q: " question "\nA:" + NUL */
#define CDS_PROMPT_CAP (3 + (CDS_CTX_CAP - 1) + 1 + 3 + CDS_QUESTION_MAX + 3 + 1)
#define CDS_CALC_CAP 320 /* longer outputs are shown raw, never evaluated */

/* ---- KB2/KB3 reader (corpus/kb/FORMAT.md). Seek+read only, one-sector
 * buffer like kb.py; every read is bounds-checked against the file size. */
#define CDS_FZ_WORDS 8 /* distinct query words sent to the fuzzy index */
#define CDS_FZ_CANDS 4

typedef struct {
    FILE *f;
    uint32_t size, nb, bt, ko, to, fnb, fbt, fent, fwo;
    uint32_t version;
    uint32_t sector; /* sector held in buf, UINT32_MAX = none */
    unsigned char buf[512];
    int err;         /* read/format error since the last retrieve */
    uint32_t lookups, sector_reads; /* since the last retrieve */
    /* per-question fuzzy cache (kept here, not on the DS stack) */
    int fz_n;
    char fz_word[CDS_FZ_WORDS][CDS_KEY_MAX + 1];
    char fz_cand[CDS_FZ_WORDS][CDS_FZ_CANDS][CDS_KEY_MAX + 1];
    int fz_ncand[CDS_FZ_WORDS];
} cds_kb;

typedef enum { CDS_KB_OK = 0, CDS_KB_MISSING = -1, CDS_KB_BAD = -2 } cds_kb_status;

cds_kb_status cds_kb_open(cds_kb *kb, const char *path);
void cds_kb_close(cds_kb *kb);
/* Exact lookup of normalize(key). 1 = hit (sentence in out, CDS_CTX_CAP bytes), 0 = miss. */
int cds_kb_lookup(cds_kb *kb, const char *key, char *out);
/* Indexed words within one edit/transposition of q, most frequent first. Returns the count. */
int cds_kb_fuzzy(cds_kb *kb, const char *q, char out[CDS_FZ_CANDS][CDS_KEY_MAX + 1]);

/* ---- retrieval + prompt (chatds_runtime.py) */
/* kb.normalize for ASCII: lowercase, non [a-z0-9] runs -> one space, trimmed.
 * Bytes >= 0x80 are dropped. Returns the length, or -1 if it does not fit. */
int cds_normalize(const char *s, char *out, size_t cap);
int cds_wants_context(const char *question);
/* 1 = context sentence in out (CDS_CTX_CAP bytes); 0 = none (gated, miss,
 * question over CDS_QUESTION_MAX, NULL kb, or kb->err). */
int cds_retrieve(cds_kb *kb, const char *question, char *out);
/* "C: ctx\nQ: q\nA:" (ctx NULL or "") -> "Q: q\nA:". Returns the length, -1 if cap is too small. */
int cds_format_prompt(const char *question, const char *ctx, char *out, size_t cap);
/* `calc(38*47)` -> `calc(38*47) = 1786` in out; returns its length. 0 = not a
 * calc call (or out too small): show the output unchanged. */
int cds_apply_calc(const char *output, char *out, size_t cap);

/* ---- generation over the engine API */
/* Tokenizer adapter; ds/tokenizer's tok_encode/tok_piece have these shapes. */
typedef struct {
    /* BOS + text when bos; writes at most max ids, returns the full count, -1 on error */
    int (*encode)(void *ctx, const char *text, size_t len, int bos, int *ids, int max);
    /* raw bytes of one id (streamed as is); NULL if out of range */
    const char *(*piece)(void *ctx, int id, int *len);
    void *ctx;
} cds_tokenizer;

typedef enum {
    CDS_STOP_NONE = 0, CDS_STOP_EOS, CDS_STOP_BOS, CDS_STOP_BUDGET,
    CDS_STOP_CONTEXT, CDS_STOP_CANCELLED
} cds_stop;
const char *cds_stop_string(cds_stop stop);

typedef struct {
    uint32_t max_new;                /* generated-token budget (run.txt steps) */
    int (*cancel)(void *user);       /* polled before every forward; nonzero cancels */
    void (*on_piece)(void *user, uint32_t id, const char *bytes, int len); /* streaming */
    void *user;
    int *ids;                        /* prompt-id scratch, ids_cap >= session context */
    uint32_t ids_cap;
    uint32_t *out_ids;               /* generated ids (max_new of them), may be NULL */
    char *text;                      /* answer bytes, NUL-terminated, may be NULL */
    size_t text_cap;
} cds_generate_opts;

typedef struct {
    uint32_t prompt_tokens, generated; /* prompt_tokens counts BOS */
    size_t text_len;                   /* bytes kept in text */
    int text_truncated;
    cds_stop stop;
} cds_generate_result;

/* Resets the session, prefills BOS+prompt (classifier only on the last prompt
 * token), then greedy decodes until EOS/BOS, the budget, a full context or
 * cancel. A prompt longer than the context returns CHATDS_CONTEXT_FULL before
 * any forward. Non-OK statuses are engine/tokenizer failures. */
chatds_status cds_generate(chatds_session *s, const cds_tokenizer *tok, const char *prompt,
                           const cds_generate_opts *o, cds_generate_result *r);

/* cds_generate one forward at a time, for callers that cannot block (the DS UI).
 * begin does everything before the first forward and returns its status; each
 * step then makes at most one forward. A step returns 1 while there is more to
 * do, 0 once finished: g->st is the status cds_generate would have returned and
 * the result is final. cds_generate is begin + steps. */
typedef struct {
    chatds_session *s;
    const cds_tokenizer *tok;
    const cds_generate_opts *o;
    cds_generate_result *r;
    char *calc;       /* set by cds_ask_begin when the answer gets calc() */
    chatds_output out;
    uint32_t fed;     /* prompt ids forwarded */
    chatds_status st;
    int done;
} cds_gen;
chatds_status cds_gen_begin(cds_gen *g, chatds_session *s, const cds_tokenizer *tok, const char *prompt,
                            const cds_generate_opts *o, cds_generate_result *r);
int cds_gen_step(cds_gen *g);

/* Opens a session that fits mem_budget bytes (session heap + TCM): tries ctx
 * (0 = model window) and on CHATDS_OUT_OF_MEMORY steps the KV window down by
 * 32 tokens, never below min_ctx. */
chatds_status cds_session_plan(const chatds_model *m, uint32_t ctx, uint32_t min_ctx,
                               size_t mem_budget, chatds_kernel kernel, chatds_session **out);

/* One question end to end: retrieval (if kb), prompt, generation, calc. */
typedef struct {
    char ctx[CDS_CTX_CAP];       /* "" if no context */
    char prompt[CDS_PROMPT_CAP];
    char calc[CDS_CALC_CAP];     /* "" if the answer is not a calc call */
    cds_generate_result gen;
} cds_answer;

/* instruct=0 (story fixture): the question is the whole prompt, no KB, no calc.
 * o->text must be set for calc to run. */
chatds_status cds_ask(chatds_session *s, const cds_tokenizer *tok, cds_kb *kb, int instruct,
                      const char *question, const cds_generate_opts *o, cds_answer *a);

/* cds_ask one forward at a time: retrieval and prompt here, then cds_gen_step
 * until 0; calc runs on the step that finishes. cds_ask is begin + steps. */
chatds_status cds_ask_begin(cds_gen *g, chatds_session *s, const cds_tokenizer *tok, cds_kb *kb,
                            int instruct, const char *question, const cds_generate_opts *o, cds_answer *a);

/* ---- card I/O */
typedef struct {
    uint32_t steps;
    int kbd, instruct;
    char prompt[CDS_QUESTION_MAX + 1];
} cds_config;

/* key=value lines, '#' comments, unknown keys ignored. Defaults: steps 64,
 * kbd 1, instruct 1, prompt "". 0 ok, -1 unreadable, -2 bad value. */
int cds_config_read(const char *path, cds_config *c);

/* File present, size >= min_size, starts with magic (NULL: any). Fills *size. */
chatds_status cds_preflight(const char *path, const char *magic, size_t magic_len,
                            uint32_t min_size, uint32_t *size);

/* Write path via path.tmp + rename, so readers never see a half file. */
FILE *cds_atomic_open(const char *path);
int cds_atomic_commit(FILE *f, const char *path); /* 0 ok; closes f either way */

typedef struct {
    const cds_answer *answer;
    const char *text;           /* streamed answer bytes */
    const uint32_t *ids;        /* generated ids */
    uint32_t ms;                /* caller-measured inference time */
    int ok;                     /* 0 writes status=FAIL */
    int instruct;
    const chatds_session *session;   /* heap.txt, may be NULL */
    size_t model_bytes;
    const char *heap_extra;     /* platform heap lines, may be NULL */
    const cds_kb *kb;           /* ctx.txt stats, may be NULL */
} cds_result;

/* Writes ids.txt, ctx.txt, calc.txt (instruct), heap.txt, then out.txt last
 * with status= as its final line, each atomically. dir ends with '/'. 0 ok. */
int cds_publish(const char *dir, const cds_result *r);

#endif
