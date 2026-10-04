/* SPDX-License-Identifier: MIT
 * Streaming answer source for the touch UI.
 *
 * The UI never sees tokens, sessions or models. It asks a generator to start
 * on a question and then pulls text once per frame, so text reaches the screen
 * as it is produced and a cancel is honoured between steps (the engine API has
 * no cancellation of its own).
 */
#ifndef CHATDS_UI_GEN_H
#define CHATDS_UI_GEN_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CHATDS_GEN_PROGRESS = 0, /* worked, nothing to show yet (prefill) */
    CHATDS_GEN_TEXT,         /* *len bytes appended to out */
    CHATDS_GEN_DONE,         /* finished; out untouched */
    CHATDS_GEN_ERROR         /* failed; out holds a short message, *len bytes */
} chatds_gen_event;

typedef struct chatds_ui_gen {
    void *ctx;
    /* Start answering. `question` is only valid during this call: copy it.
     * Returns 0 on success; nonzero = refused, nothing started. */
    int (*begin)(void *ctx, const char *question);
    /* One bounded unit of work (one engine forward on the real engine).
     * out has room for cap bytes; set *len for TEXT and ERROR. */
    chatds_gen_event (*step)(void *ctx, char *out, size_t cap, size_t *len);
    /* Abandon the current answer. The generator is reusable via begin(). */
    void (*cancel)(void *ctx);
} chatds_ui_gen;

#ifdef __cplusplus
}
#endif
#endif
