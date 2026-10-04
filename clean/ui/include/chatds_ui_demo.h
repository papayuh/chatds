/* SPDX-License-Identifier: MIT
 * Canned-answer generator and scripted-input driver. These stand in for the
 * engine so the UI can be exercised on the host and in melonDS without a model.
 */
#ifndef CHATDS_UI_DEMO_H
#define CHATDS_UI_DEMO_H

#include "chatds_ui.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- demo generator ----------------------------------------------------------
 * Questions starting with "long" get a multi-paragraph answer, "fail" errors
 * after two steps, "empty" finishes with no text; anything else is echoed.
 * 3 prefill PROGRESS steps, then 3 characters per step. */
typedef struct {
    chatds_ui_gen gen;
    char answer[1280];
    uint16_t len, pos;
    uint8_t prefill, mode;
} chatds_demo_gen;

void chatds_demo_gen_init(chatds_demo_gen *d);

/* ---- scripted input ------------------------------------------------------------
 * One command per line, '#' starts a comment:
 *   wait N            idle N frames
 *   tap X Y           press and release at a touchscreen point
 *   hold X Y N        keep the pen down N frames
 *   type TEXT         tap the keys that type TEXT (switches layers/shift)
 *   button NAME       tap send|stop|up|down|about
 *   key NAME          press a hardware key: up|down|b|start|select
 *   until idle MAX    run frames until the answer finishes (at most MAX)
 *   dump NAME         hand the screens and a text state summary to the host
 * Every action goes through chatds_ui_frame() as ordinary pen/key input. */
typedef struct {
    void *ctx;
    void (*tick)(void *ctx);                    /* before every frame (may be NULL) */
    void (*dump)(void *ctx, const char *name, chatds_ui *ui);
    uint16_t *top, *bottom;                     /* framebuffers rendered each frame */
} chatds_script_host;

/* Returns 0, or the 1-based number of the first line it could not run. */
int chatds_script_run(chatds_ui *ui, const char *script, const chatds_script_host *host);

/* Text summary of the UI state (busy/about/scroll/input and every log line),
 * delivered line by line to sink. */
void chatds_script_state_text(const chatds_ui *ui, void (*sink)(void *ctx, const char *line),
                              void *ctx);

#ifdef __cplusplus
}
#endif
#endif
