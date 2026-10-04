/* SPDX-License-Identifier: MIT
 * Generator that answers through the product plumbing (clean/plumbing), one
 * engine forward per UI step.
 *
 * Retrieval, the C:/Q:/A: prompt, the generation loop with its EOS/BOS/budget/
 * context stops, the tokenizer interface and calc() are plumbing's own
 * (cds_ask_begin + cds_gen_step), so the UI answers exactly like cds_ask. The
 * adapter only streams each piece to the UI as it is produced and appends the
 * calc() result when there is one. Cancel goes through the plumbing's cancel
 * poll, and the UI stops stepping, so it takes effect between forwards.
 */
#ifndef CHATDS_UI_ENGINE_H
#define CHATDS_UI_ENGINE_H

#include "chatds_ui.h"
#include "plumbing.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    chatds_ui_gen gen;
    chatds_session *session;
    const cds_tokenizer *tok;
    cds_kb *kb;            /* NULL: no retrieval */
    int instruct;          /* as cds_ask */
    cds_generate_opts opts;
    /* state */
    cds_gen run;
    cds_answer answer;
    char text[CDS_CALC_CAP]; /* answer bytes; enough for every calc() */
    char *out;               /* current step's output */
    size_t cap, len;
    uint8_t cancelled, calc_shown;
} chatds_ui_engine_gen;

/* ids: prompt-id scratch, ids_cap >= the session context (cds_generate_opts). */
void chatds_ui_engine_gen_init(chatds_ui_engine_gen *g, chatds_session *session,
                               const cds_tokenizer *tok, cds_kb *kb, int instruct, int *ids,
                               uint32_t ids_cap, uint32_t max_new);

#ifdef __cplusplus
}
#endif
#endif
