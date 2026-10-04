/* SPDX-License-Identifier: MIT */
#include <string.h>

#include "chatds_ui_engine.h"

static int eg_cancelled(void *user) { return ((chatds_ui_engine_gen *)user)->cancelled; }

static void eg_piece(void *user, uint32_t id, const char *bytes, int len)
{
    chatds_ui_engine_gen *g = user;
    size_t n = (size_t)len < g->cap - g->len ? (size_t)len : g->cap - g->len;
    (void)id;
    memcpy(g->out + g->len, bytes, n);
    g->len += n;
}

static int eg_begin(void *ctx, const char *question)
{
    chatds_ui_engine_gen *g = ctx;
    g->cancelled = 0;
    g->calc_shown = 0;
    return cds_ask_begin(&g->run, g->session, g->tok, g->kb, g->instruct, question, &g->opts,
                         &g->answer) != CHATDS_OK;
}

static chatds_gen_event eg_step(void *ctx, char *out, size_t cap, size_t *len)
{
    chatds_ui_engine_gen *g = ctx;
    g->out = out;
    g->cap = cap;
    g->len = 0;
    int more = cds_gen_step(&g->run);
    *len = g->len;
    if (g->len) return CHATDS_GEN_TEXT;
    if (more) return CHATDS_GEN_PROGRESS;
    if (g->run.st != CHATDS_OK) {
        const char *m = chatds_status_string(g->run.st);
        *len = strlen(m) < cap ? strlen(m) : cap;
        memcpy(out, m, *len);
        return CHATDS_GEN_ERROR;
    }
    if (g->answer.calc[0] && !g->calc_shown) { /* "calc(6*7)" streamed: add " = 42" */
        const char *r = strchr(g->answer.calc, '=') - 1;
        *len = strlen(r) < cap ? strlen(r) : cap;
        memcpy(out, r, *len);
        g->calc_shown = 1;
        return CHATDS_GEN_TEXT;
    }
    return CHATDS_GEN_DONE;
}

static void eg_cancel(void *ctx) { ((chatds_ui_engine_gen *)ctx)->cancelled = 1; }

void chatds_ui_engine_gen_init(chatds_ui_engine_gen *g, chatds_session *session,
                               const cds_tokenizer *tok, cds_kb *kb, int instruct, int *ids,
                               uint32_t ids_cap, uint32_t max_new)
{
    memset(g, 0, sizeof *g);
    g->gen.ctx = g;
    g->gen.begin = eg_begin;
    g->gen.step = eg_step;
    g->gen.cancel = eg_cancel;
    g->session = session;
    g->tok = tok;
    g->kb = kb;
    g->instruct = instruct;
    g->opts.max_new = max_new;
    g->opts.cancel = eg_cancelled;
    g->opts.on_piece = eg_piece;
    g->opts.user = g;
    g->opts.ids = ids;
    g->opts.ids_cap = ids_cap;
    g->opts.text = g->text;
    g->opts.text_cap = sizeof g->text;
}
