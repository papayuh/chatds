/* SPDX-License-Identifier: MIT
 * Host unit tests with a stub engine + byte tokenizer: generation loop, KV planning, cds_ask,
 * card I/O and malformed-KB handling. Run by test_parity.py: test-plumbing MINI_KB TMPDIR */
#define _GNU_SOURCE /* memmem */
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "plumbing.h"

/* ---- stub engine: scripted argmax per classified forward; KV costs 1000 B per token */
struct chatds_model { chatds_model_config cfg; };
struct chatds_session { uint32_t ctx, pos; };
static uint32_t script[64], script_n, script_i, forwards, classified;
static int32_t logit;

chatds_status chatds_model_from_memory(const void *b, size_t n, chatds_model **out) {
    (void)b, (void)n;
    *out = calloc(1, sizeof **out);
    (*out)->cfg.context_length = 256, (*out)->cfg.vocab_size = 3 + 256;
    return CHATDS_OK;
}
const chatds_model_config *chatds_model_get_config(const chatds_model *m) { return &m->cfg; }
void chatds_model_destroy(chatds_model *m) { free(m); }
chatds_status chatds_session_create(const chatds_model *m, const chatds_session_options *o, chatds_session **out) {
    uint32_t ctx = o && o->context_length ? o->context_length : m->cfg.context_length;
    *out = NULL;
    if (o && o->max_memory_bytes && (size_t)ctx * 1000 > o->max_memory_bytes)
        return CHATDS_OUT_OF_MEMORY;
    *out = calloc(1, sizeof **out);
    (*out)->ctx = ctx;
    return CHATDS_OK;
}
void chatds_session_destroy(chatds_session *s) { free(s); }
void chatds_session_reset(chatds_session *s) { s->pos = 0; }
void chatds_session_get_info(const chatds_session *s, chatds_session_info *o) {
    memset(o, 0, sizeof *o);
    o->position = s->pos, o->context_length = s->ctx, o->kv_bytes = (size_t)s->ctx * 1000;
}
chatds_status chatds_session_forward(chatds_session *s, uint32_t token, chatds_output *out) {
    if (token >= 3 + 256)
        return CHATDS_INVALID_ARGUMENT;
    if (s->pos >= s->ctx)
        return CHATDS_CONTEXT_FULL;
    s->pos++, forwards++;
    if (out) {
        classified++;
        out->argmax = script_i < script_n ? script[script_i++] : 3 + 'x';
        out->logits_q16 = &logit, out->logits_count = 1;
    }
    return CHATDS_OK;
}

static void set_script(const char *text, int end) {
    script_n = script_i = forwards = classified = 0;
    for (; *text; text++)
        script[script_n++] = 3 + (unsigned char)*text;
    if (end)
        script[script_n++] = (uint32_t)end;
}

/* ---- byte tokenizer: BOS, then id 3 + byte */
static int enc(void *c, const char *t, size_t n, int bos, int *ids, int max) {
    (void)c;
    int k = 0;
    if (bos && k < max)
        ids[k] = CDS_BOS;
    k += bos;
    for (size_t i = 0; i < n; i++, k++)
        if (k < max)
            ids[k] = 3 + (unsigned char)t[i];
    return k;
}
static char bytes[256];
static const char *piece(void *c, int id, int *len) {
    (void)c;
    if (id < 0 || id >= 3 + 256)
        return NULL;
    *len = id >= 3;
    if (id >= 3)
        bytes[id - 3] = (char)(id - 3);
    return id >= 3 ? &bytes[id - 3] : "";
}
static const cds_tokenizer tok = {enc, piece, NULL};

static int polls, cancel_at;
static int cancel_cb(void *u) { (void)u; return ++polls >= cancel_at; }
static char streamed[256];
static void on_piece(void *u, uint32_t id, const char *b, int n) { (void)u, (void)id; strncat(streamed, b, n); }

static char *slurp(const char *path, long *n) {
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    *n = ftell(f);
    char *b = malloc(*n + 1);
    fseek(f, 0, SEEK_SET);
    assert(fread(b, 1, *n, f) == (size_t)*n);
    b[*n] = 0;
    fclose(f);
    return b;
}
static void spit(const char *path, const void *b, long n) {
    FILE *f = fopen(path, "wb");
    if (n < 0)
        n = (long)strlen(b);
    assert(f && fwrite(b, 1, n, f) == (size_t)n);
    fclose(f);
}
static void put32(char *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (char)(v >> 8 * i); }

static void test_generate(chatds_model *m) {
    chatds_session *s;
    int ids[256];
    uint32_t out_ids[64];
    char text[64];
    cds_generate_opts o = {.max_new = 16, .ids = ids, .ids_cap = 256, .out_ids = out_ids, .text = text,
                           .text_cap = sizeof text, .on_piece = on_piece};
    cds_generate_result r;
    assert(chatds_session_create(m, NULL, &s) == CHATDS_OK);

    set_script("ok", CDS_EOS);
    streamed[0] = 0;
    assert(cds_generate(s, &tok, "hi", &o, &r) == CHATDS_OK);
    assert(r.stop == CDS_STOP_EOS && r.prompt_tokens == 3 && r.generated == 2);
    assert(!strcmp(text, "ok") && !strcmp(streamed, "ok") && r.text_len == 2 && !r.text_truncated);
    assert(out_ids[0] == 3 + 'o' && out_ids[1] == 3 + 'k');
    assert(forwards == 3 + 2 && classified == 3); /* classifier only on the last prompt token */

    set_script("abc", CDS_EOS); /* budget: the last budgeted token is not forwarded */
    o.max_new = 2;
    assert(cds_generate(s, &tok, "hi", &o, &r) == CHATDS_OK);
    assert(r.stop == CDS_STOP_BUDGET && r.generated == 2 && !strcmp(text, "ab") && forwards == 3 + 1);
    o.max_new = 0;
    assert(cds_generate(s, &tok, "hi", &o, &r) == CHATDS_OK && r.stop == CDS_STOP_BUDGET && r.generated == 0);

    set_script("", CDS_BOS);
    o.max_new = 16;
    assert(cds_generate(s, &tok, "hi", &o, &r) == CHATDS_OK && r.stop == CDS_STOP_BOS && r.generated == 0);

    set_script("abcdef", CDS_EOS); /* tiny buffer: truncated, still generated */
    o.text_cap = 3;
    assert(cds_generate(s, &tok, "hi", &o, &r) == CHATDS_OK);
    assert(r.generated == 6 && r.text_truncated && !strcmp(text, "ab"));
    o.text_cap = sizeof text;

    polls = 0, cancel_at = 2; /* cancel during prefill */
    o.cancel = cancel_cb;
    set_script("abc", CDS_EOS);
    assert(cds_generate(s, &tok, "hi", &o, &r) == CHATDS_OK && r.stop == CDS_STOP_CANCELLED && r.generated == 0);
    polls = 0, cancel_at = 4; /* cancel after prefill, before streaming a pending token */
    set_script("abc", CDS_EOS);
    assert(cds_generate(s, &tok, "hi", &o, &r) == CHATDS_OK && r.stop == CDS_STOP_CANCELLED && r.generated == 0);
    polls = 0, cancel_at = 5; /* 3 prefill polls, one before emit, one after emit */
    set_script("abc", CDS_EOS);
    assert(cds_generate(s, &tok, "hi", &o, &r) == CHATDS_OK && r.stop == CDS_STOP_CANCELLED && r.generated == 1);
    o.cancel = NULL;
    chatds_session_destroy(s);

    chatds_session_options opt = {5, CHATDS_KERNEL_FAST, 0}; /* window 5 */
    assert(chatds_session_create(m, &opt, &s) == CHATDS_OK);
    set_script("abcdefgh", CDS_EOS);
    assert(cds_generate(s, &tok, "hi", &o, &r) == CHATDS_OK);
    /* positions 3 after prefill; tokens 1 and 2 fill the window; token 3 cannot be forwarded */
    assert(r.stop == CDS_STOP_CONTEXT && r.generated == 3 && !strcmp(text, "abc"));
    forwards = 0;
    assert(cds_generate(s, &tok, "hello", &o, &r) == CHATDS_CONTEXT_FULL && forwards == 0);
    chatds_session_destroy(s);
}

static void test_plan(chatds_model *m) {
    chatds_session *s;
    assert(cds_session_plan(m, 0, 64, 200000, CHATDS_KERNEL_FAST, &s) == CHATDS_OK);
    assert(s->ctx == 192); /* 256, 224 do not fit 200 kB */
    chatds_session_destroy(s);
    assert(cds_session_plan(m, 0, 230, 200000, CHATDS_KERNEL_FAST, &s) == CHATDS_OUT_OF_MEMORY && !s);
    assert(cds_session_plan(m, 100, 0, 1000000, CHATDS_KERNEL_FAST, &s) == CHATDS_OK && s->ctx == 100);
    chatds_session_destroy(s);
    assert(cds_session_plan(m, 0, 0, 0, CHATDS_KERNEL_FAST, &s) == CHATDS_OUT_OF_MEMORY);
}

static void test_ask_publish(chatds_model *m, const char *kbp, const char *dir) {
    chatds_session *s;
    cds_kb kb;
    cds_answer a;
    int ids[256];
    uint32_t out_ids[64];
    char text[64], path[512], *b;
    long n;
    cds_generate_opts o = {.max_new = 32, .ids = ids, .ids_cap = 256, .out_ids = out_ids, .text = text,
                           .text_cap = sizeof text};
    assert(cds_kb_open(&kb, kbp) == CDS_KB_OK);
    assert(chatds_session_create(m, NULL, &s) == CHATDS_OK);
    set_script(" calc(2+3)", CDS_EOS);
    assert(cds_ask(s, &tok, &kb, 1, "what is france", &o, &a) == CHATDS_OK);
    assert(!strcmp(a.ctx, "France is a country in western Europe."));
    assert(!strcmp(a.prompt, "C: France is a country in western Europe.\nQ: what is france\nA:"));
    assert(!strcmp(a.calc, "calc(2+3) = 5") && a.gen.prompt_tokens == 1 + strlen(a.prompt));

    cds_result res = {.answer = &a, .text = text, .ids = out_ids, .ms = 42, .ok = 1, .instruct = 1,
                      .session = s, .model_bytes = 123, .heap_extra = "heap_start=0x0\n", .kb = &kb};
    assert(cds_publish(dir, &res) == 0);
    snprintf(path, sizeof path, "%sout.txt", dir);
    b = slurp(path, &n);
    char want[256];
    snprintf(want, sizeof want, " calc(2+3)\ntokens=%u\nprompt_tokens=%u\ngenerated=10\nstop=eos\nms=42\nstatus=OK\n",
             a.gen.prompt_tokens - 1 + 10, a.gen.prompt_tokens);
    assert(!strcmp(b, want));
    free(b);
    snprintf(path, sizeof path, "%sctx.txt", dir);
    b = slurp(path, &n);
    const char *ctx_want = "ctx=France is a country in western Europe.\nkb=ok lookups=";
    assert(!strncmp(b, ctx_want, strlen(ctx_want)));
    free(b);
    snprintf(path, sizeof path, "%scalc.txt", dir);
    assert(!strcmp(b = slurp(path, &n), "calc=calc(2+3) = 5\n"));
    free(b);
    snprintf(path, sizeof path, "%sids.txt", dir);
    b = slurp(path, &n);
    assert(!strncmp(b, "35\n102\n", 7)); /* ' ' and 'c' as byte ids */
    free(b);
    snprintf(path, sizeof path, "%sheap.txt", dir);
    b = slurp(path, &n);
    assert(strstr(b, "model_bytes=123\n") && strstr(b, "kv_bytes=256000\n") && strstr(b, "heap_start=0x0\n"));
    free(b);
    snprintf(path, sizeof path, "%sout.txt.tmp", dir);
    assert(!fopen(path, "rb"));

    set_script("hello", CDS_EOS); /* not fact-shaped: no lookup; not calc: no calc line */
    assert(cds_ask(s, &tok, &kb, 1, "hiii", &o, &a) == CHATDS_OK);
    assert(!a.ctx[0] && !a.calc[0] && !strcmp(a.prompt, "Q: hiii\nA:") && !strcmp(text, "hello"));
    set_script("x", CDS_EOS); /* story mode: raw prompt */
    assert(cds_ask(s, &tok, &kb, 0, "Once upon a time", &o, &a) == CHATDS_OK && !strcmp(a.prompt, "Once upon a time"));
    char longq[200];
    memset(longq, 'a', sizeof longq - 1);
    longq[sizeof longq - 1] = 0;
    assert(cds_ask(s, &tok, &kb, 1, longq, &o, &a) == CHATDS_INVALID_ARGUMENT);
    assert(cds_ask(s, &tok, NULL, 1, "what is france", &o, &a) == CHATDS_OK && !a.ctx[0]); /* no kb.bin */

    res.ok = 0, res.instruct = 0, res.session = NULL, res.kb = NULL;
    assert(cds_publish(dir, &res) == 0);
    snprintf(path, sizeof path, "%sout.txt", dir);
    b = slurp(path, &n);
    assert(n > 12 && !strcmp(b + n - 12, "status=FAIL\n"));
    free(b);
    chatds_session_destroy(s);
    cds_kb_close(&kb);
}

static void test_io(const char *dir) {
    char path[512];
    cds_config c;
    uint32_t size;
    snprintf(path, sizeof path, "%srun.txt", dir);
    spit(path, "# comment\r\nsteps=48\r\nkbd=0\nprompt=who was george washington\nother=1\n", -1);
    assert(cds_config_read(path, &c) == 0);
    assert(c.steps == 48 && c.kbd == 0 && c.instruct == 1 && !strcmp(c.prompt, "who was george washington"));
    spit(path, "steps=lots\ninstruct=0\n", -1);
    assert(cds_config_read(path, &c) == -2 && c.steps == 64 && c.instruct == 0);
    snprintf(path, sizeof path, "%snope.txt", dir);
    assert(cds_config_read(path, &c) == -1);
    assert(cds_preflight(path, NULL, 0, 0, &size) == CHATDS_IO_ERROR);
    snprintf(path, sizeof path, "%smodel.bin", dir);
    spit(path, "TIE1\1\0\0\0", 8);
    assert(cds_preflight(path, "TIE1", 4, 8, &size) == CHATDS_OK && size == 8);
    assert(cds_preflight(path, "TIE1", 4, 512, &size) == CHATDS_BAD_FORMAT);
    assert(cds_preflight(path, "KB3\0", 4, 0, &size) == CHATDS_BAD_FORMAT);
}

static void test_kb(const char *kbp, const char *dir) {
    cds_kb kb;
    char out[CDS_CTX_CAP], c[CDS_FZ_CANDS][CDS_KEY_MAX + 1], path[512];
    long n;
    char *orig = slurp(kbp, &n), *b = malloc(n);
    assert(cds_kb_open(&kb, kbp) == CDS_KB_OK);
    assert(cds_kb_lookup(&kb, "France", out) && !strcmp(out, "France is a country in western Europe."));
    assert(cds_kb_lookup(&kb, "  PARIS!", out) && !strcmp(out, "Paris is the capital city of France."));
    assert(cds_kb_lookup(&kb, "lutetia", out) && !strcmp(out, "Paris is the capital city of France."));
    const char *miss[] = {"mercury", "list of rivers", "tiny", "talk paris", "nothing here", "",
                          "a very long key that is well over sixty three characters long, so never stored"};
    for (unsigned i = 0; i < sizeof miss / sizeof *miss; i++)
        assert(!cds_kb_lookup(&kb, miss[i], out));
    kb.sector = UINT32_MAX, kb.sector_reads = 0;
    assert(cds_kb_lookup(&kb, "france", out) && kb.sector_reads <= 3);
    assert(cds_kb_fuzzy(&kb, "franse", c) == 1 && !strcmp(c[0], "france"));
    assert(cds_kb_fuzzy(&kb, "frnace", c) == 1 && !strcmp(c[0], "france"));
    assert(cds_kb_fuzzy(&kb, "lutetiaa", c) == 1 && !strcmp(c[0], "lutetia"));
    assert(cds_kb_fuzzy(&kb, "cafe", c) == 0 && cds_kb_fuzzy(&kb, "zzzzzz", c) == 0);
    assert(cds_kb_fuzzy(&kb, "france", c) == 0 && !kb.err);
    cds_kb_close(&kb);

    snprintf(path, sizeof path, "%sbad.kb", dir);
    assert(cds_kb_open(&kb, path) == CDS_KB_MISSING);
    struct { uint32_t at, val; } hdr[] = {{32, (uint32_t)n + 1}, {8, 3}, {36, 0}, {20, (uint32_t)n}, {48, 0xFFFFFFF0u}};
    for (unsigned i = 0; i < sizeof hdr / sizeof *hdr; i++) { /* size, pow2, fz pow2, offsets inside the file */
        memcpy(b, orig, n);
        put32(b + hdr[i].at, hdr[i].val);
        spit(path, b, n);
        assert(cds_kb_open(&kb, path) == CDS_KB_BAD);
    }
    spit(path, orig, n - 1); /* truncated */
    assert(cds_kb_open(&kb, path) == CDS_KB_BAD);
    memcpy(b, orig, n);
    b[1] = 'B' + 1; /* "KC3" */
    spit(path, b, n);
    assert(cds_kb_open(&kb, path) == CDS_KB_BAD);

    /* corrupt record length: 255 bytes would overrun a 101-byte buffer; ours holds 256 and rd
     * refuses reads past the file end */
    memcpy(b, orig, n);
    char *rec = memmem(b, n, "France is a country", 19);
    assert(rec);
    rec[-1] = (char)255;
    spit(path, b, n);
    assert(cds_kb_open(&kb, path) == CDS_KB_OK);
    int hit = cds_kb_lookup(&kb, "france", out);
    assert(hit ? strlen(out) <= 255 : kb.err);
    cds_kb_close(&kb);

    /* garbage bucket table: a miss with err set, not a hang or an out-of-bounds read */
    memcpy(b, orig, n);
    uint32_t bt = (unsigned char)b[20] | (unsigned char)b[21] << 8, nb = (unsigned char)b[8] | (unsigned char)b[9] << 8;
    for (uint32_t i = 0; i <= nb; i++)
        put32(b + bt + 4 * i, i * 0x00100000u);
    spit(path, b, n);
    assert(cds_kb_open(&kb, path) == CDS_KB_OK);
    assert(!cds_kb_lookup(&kb, "france", out) && kb.err);
    assert(!cds_retrieve(&kb, "what is france", out)); /* err is reset per question, still a miss */
    cds_kb_close(&kb);
    remove(path);
    free(orig), free(b);
}

int main(int argc, char **argv) {
    chatds_model *m;
    assert(argc == 3);
    assert(chatds_model_from_memory("", 0, &m) == CHATDS_OK);
    test_generate(m);
    test_plan(m);
    test_kb(argv[1], argv[2]);
    test_ask_publish(m, argv[1], argv[2]);
    test_io(argv[2]);
    chatds_model_destroy(m);
    puts("test-plumbing: all passed");
    return 0;
}
