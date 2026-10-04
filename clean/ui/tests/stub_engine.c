/* SPDX-License-Identifier: MIT */
#include <stdlib.h>
#include <string.h>

#include "stub_engine.h"

stub_engine g_stub;

struct chatds_session { int unused; };
static struct chatds_session the_session;

chatds_status chatds_session_create(const chatds_model *m, const chatds_session_options *o,
                                    chatds_session **out)
{
    (void)m; (void)o;
    *out = &the_session;
    return CHATDS_OK;
}

void chatds_session_destroy(chatds_session *s) { (void)s; }

const chatds_model_config *chatds_model_get_config(const chatds_model *m)
{
    static chatds_model_config c;
    (void)m;
    c.context_length = g_stub.context_length;
    return &c;
}

void chatds_session_get_info(const chatds_session *s, chatds_session_info *out)
{
    (void)s;
    memset(out, 0, sizeof *out);
    out->position = g_stub.pos;
    out->context_length = g_stub.context_length;
}

void chatds_session_reset(chatds_session *s)
{
    (void)s;
    g_stub.resets++;
    g_stub.pos = g_stub.forwards = g_stub.prefill_forwards = g_stub.classified = 0;
}

chatds_status chatds_session_forward(chatds_session *s, uint32_t token, chatds_output *out)
{
    (void)s;
    if (g_stub.pos >= g_stub.context_length) return CHATDS_CONTEXT_FULL;
    g_stub.pos++;
    g_stub.forwards++;
    g_stub.last_token = token;
    if (!out) {
        g_stub.prefill_forwards++;
        return CHATDS_OK;
    }
    out->logits_q16 = NULL;
    out->logits_count = 0;
    out->argmax = g_stub.classified < g_stub.answer_len ? g_stub.answer[g_stub.classified]
                                                         : g_stub.eos;
    g_stub.classified++;
    return CHATDS_OK;
}

const char *chatds_status_string(chatds_status st)
{
    static const char *const s[] = {"ok", "invalid argument", "io error", "bad format",
                                    "unsupported", "out of memory", "context full"};
    return (unsigned)st < 7 ? s[st] : "unknown";
}
