/* SPDX-License-Identifier: MIT
 * Generation loop, session/KV planning and the per-question pipeline over chatds_engine.h. */
#include <string.h>
#include "plumbing.h"

#define CTX_STEP 32

const char *cds_stop_string(cds_stop stop) {
    static const char *const s[] = {"none", "eos", "bos", "budget", "context", "cancelled"};
    return (unsigned)stop < sizeof s / sizeof *s ? s[stop] : "none";
}

chatds_status cds_gen_begin(cds_gen *g, chatds_session *s, const cds_tokenizer *tok, const char *prompt,
                            const cds_generate_opts *o, cds_generate_result *r) {
    chatds_session_info info;
    memset(g, 0, sizeof *g);
    g->done = 1;
    memset(r, 0, sizeof *r);
    if (o->text && o->text_cap)
        o->text[0] = 0;
    if (!s || !tok || !prompt || !o->ids || !o->ids_cap)
        return CHATDS_INVALID_ARGUMENT;
    chatds_session_reset(s);
    chatds_session_get_info(s, &info);
    int n = tok->encode(tok->ctx, prompt, strlen(prompt), 1, o->ids, (int)o->ids_cap);
    if (n < 0)
        return CHATDS_OUT_OF_MEMORY;
    if (n == 0)
        return CHATDS_INVALID_ARGUMENT;
    if ((uint32_t)n > o->ids_cap || (uint32_t)n > info.context_length)
        return CHATDS_CONTEXT_FULL;
    r->prompt_tokens = (uint32_t)n;
    g->s = s;
    g->tok = tok;
    g->o = o;
    g->r = r;
    g->done = 0;
    return CHATDS_OK;
}

static int gen_end(cds_gen *g, chatds_status st, cds_stop stop) {
    g->st = st;
    g->r->stop = stop;
    g->done = 1;
    if (st == CHATDS_OK && g->calc && g->o->text && !g->r->text_truncated)
        cds_apply_calc(g->o->text, g->calc, CDS_CALC_CAP);
    return 0;
}

int cds_gen_step(cds_gen *g) {
    const cds_generate_opts *o = g->o;
    cds_generate_result *r = g->r;
    chatds_status st;
    if (g->done)
        return 0;
    if (o->cancel && o->cancel(o->user))
        return gen_end(g, CHATDS_OK, CDS_STOP_CANCELLED);
    if (g->fed < r->prompt_tokens) { /* prefill: the classifier only runs on the last prompt token */
        int last = g->fed + 1 == r->prompt_tokens;
        if ((st = chatds_session_forward(g->s, (uint32_t)o->ids[g->fed], last ? &g->out : NULL)) != CHATDS_OK)
            return gen_end(g, st, CDS_STOP_NONE);
        g->fed++;
        return 1;
    }
    if (r->generated >= o->max_new)
        return gen_end(g, CHATDS_OK, CDS_STOP_BUDGET);
    uint32_t t = g->out.argmax;
    if (t == CDS_EOS || t == CDS_BOS)
        return gen_end(g, CHATDS_OK, t == CDS_EOS ? CDS_STOP_EOS : CDS_STOP_BOS);
    int len = 0;
    const char *p = g->tok->piece(g->tok->ctx, (int)t, &len);
    if (!p)
        return gen_end(g, CHATDS_BAD_FORMAT, CDS_STOP_NONE); /* model vocab larger than the tokenizer's */
    if (o->out_ids)
        o->out_ids[r->generated] = t;
    r->generated++;
    if (o->text && o->text_cap) {
        size_t k = (size_t)len < o->text_cap - 1 - r->text_len ? (size_t)len : o->text_cap - 1 - r->text_len;
        memcpy(o->text + r->text_len, p, k);
        r->text_len += k;
        o->text[r->text_len] = 0;
        r->text_truncated |= k < (size_t)len;
    }
    if (o->on_piece)
        o->on_piece(o->user, t, p, len);
    if (r->generated >= o->max_new) /* the last budgeted token needs no forward */
        return gen_end(g, CHATDS_OK, CDS_STOP_BUDGET);
    if (o->cancel && o->cancel(o->user))
        return gen_end(g, CHATDS_OK, CDS_STOP_CANCELLED);
    st = chatds_session_forward(g->s, t, &g->out);
    if (st == CHATDS_CONTEXT_FULL)
        return gen_end(g, CHATDS_OK, CDS_STOP_CONTEXT);
    if (st != CHATDS_OK)
        return gen_end(g, st, CDS_STOP_NONE);
    return 1;
}

chatds_status cds_generate(chatds_session *s, const cds_tokenizer *tok, const char *prompt,
                           const cds_generate_opts *o, cds_generate_result *r) {
    cds_gen g;
    chatds_status st = cds_gen_begin(&g, s, tok, prompt, o, r);
    if (st != CHATDS_OK)
        return st;
    while (cds_gen_step(&g)) {
    }
    return g.st;
}

chatds_status cds_session_plan(const chatds_model *m, uint32_t ctx, uint32_t min_ctx,
                               size_t mem_budget, chatds_kernel kernel, chatds_session **out) {
    uint32_t window = chatds_model_get_config(m)->context_length;
    chatds_status st = CHATDS_INVALID_ARGUMENT;
    *out = NULL;
    if (!ctx || ctx > window)
        ctx = window;
    if (!mem_budget)
        return CHATDS_OUT_OF_MEMORY; /* 0 would mean the engine's ceiling, not "nothing free" */
    for (;;) {
        chatds_session_options opt = {ctx, kernel, mem_budget};
        st = chatds_session_create(m, &opt, out);
        if (st != CHATDS_OUT_OF_MEMORY || ctx <= min_ctx || ctx <= CTX_STEP)
            return st;
        ctx = ctx - CTX_STEP < min_ctx ? min_ctx : ctx - CTX_STEP;
    }
}

chatds_status cds_ask_begin(cds_gen *g, chatds_session *s, const cds_tokenizer *tok, cds_kb *kb,
                            int instruct, const char *question, const cds_generate_opts *o, cds_answer *a) {
    a->ctx[0] = a->calc[0] = a->prompt[0] = 0;
    memset(&a->gen, 0, sizeof a->gen);
    memset(g, 0, sizeof *g);
    g->done = 1;
    if (strlen(question) > CDS_QUESTION_MAX)
        return CHATDS_INVALID_ARGUMENT;
    if (!instruct) {
        strcpy(a->prompt, question);
    } else {
        if (!cds_retrieve(kb, question, a->ctx))
            a->ctx[0] = 0;
        if (cds_format_prompt(question, a->ctx, a->prompt, sizeof a->prompt) < 0)
            return CHATDS_INVALID_ARGUMENT; /* cannot happen: CDS_PROMPT_CAP covers both caps */
    }
    chatds_status st = cds_gen_begin(g, s, tok, a->prompt, o, &a->gen);
    if (instruct)
        g->calc = a->calc;
    return st;
}

chatds_status cds_ask(chatds_session *s, const cds_tokenizer *tok, cds_kb *kb, int instruct,
                      const char *question, const cds_generate_opts *o, cds_answer *a) {
    cds_gen g;
    chatds_status st = cds_ask_begin(&g, s, tok, kb, instruct, question, o, a);
    if (st != CHATDS_OK)
        return st;
    while (cds_gen_step(&g)) {
    }
    return g.st;
}
