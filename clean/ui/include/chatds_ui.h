/* SPDX-License-Identifier: MIT
 * ChatDS touch UI core: two 256x192 screens, no heap, no platform calls.
 *
 *   top    chat log (word-wrapped, scrollable, streams as text arrives) with a
 *          busy spinner in the header; also shows the credits screen
 *   bottom typed-question box, button bar (Send/Stop/Up/Down/About) and the
 *          on-screen keyboard
 *
 * The caller owns the clock and the hardware: feed one chatds_ui_input per
 * frame (60 Hz on the DS), then call chatds_ui_render() into the two
 * framebuffers. Pixels are 16-bit ARGB1555 with bit 15 set, row stride 256,
 * written with 16-bit stores only, so the buffers may be DS VRAM directly.
 */
#ifndef CHATDS_UI_CORE_H
#define CHATDS_UI_CORE_H

#include <stddef.h>
#include <stdint.h>

#include "chatds_ui_gen.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CHATDS_UI_W 256
#define CHATDS_UI_H 192

/* Capacities. Override at compile time; the log evicts oldest messages first. */
#ifndef CHATDS_UI_INPUT_MAX
#define CHATDS_UI_INPUT_MAX 127   /* bytes of typed question; CDS_QUESTION_MAX */
#endif
#ifndef CHATDS_UI_LOG_BYTES
#define CHATDS_UI_LOG_BYTES 6144  /* chat history text; keep under 65536 */
#endif
#ifndef CHATDS_UI_LOG_LINES
#define CHATDS_UI_LOG_LINES 320   /* wrapped lines kept */
#endif
#define CHATDS_UI_STEP_TEXT 64    /* largest text chunk taken from a generator step */

/* Hardware-key bits for chatds_ui_input.keys (edge-triggered by the UI). */
#define CHATDS_KEY_UP     0x01
#define CHATDS_KEY_DOWN   0x02
#define CHATDS_KEY_B      0x04  /* cancel */
#define CHATDS_KEY_START  0x08  /* send */
#define CHATDS_KEY_SELECT 0x10  /* about / back */

typedef struct {
    uint8_t touch; /* pen/finger currently down */
    uint8_t x, y;  /* touchscreen pixel, 0..255 x 0..191; ignored when !touch */
    uint8_t keys;  /* CHATDS_KEY_* currently held */
} chatds_ui_input;

typedef enum {
    CHATDS_ROLE_USER = 1,   /* values double as separator bytes in the log */
    CHATDS_ROLE_ANSWER = 2,
    CHATDS_ROLE_NOTE = 3
} chatds_ui_role;

typedef struct { uint16_t off; uint8_t len; uint8_t flags; } chatds_ui_line;

/* Everything below is private: callers allocate the struct, never poke it. */
typedef struct {
    char text[CHATDS_UI_LOG_BYTES];
    chatds_ui_line lines[CHATDS_UI_LOG_LINES];
    uint16_t used, nlines;
    uint16_t open_off;  /* offset of the newest message's role byte */
    uint16_t open_line; /* first line index of the newest message */
    uint8_t has_open;
} chatds_ui_log;

typedef struct {
    chatds_ui_log log;
    const chatds_ui_gen *gen;
    char input[CHATDS_UI_INPUT_MAX + 1];
    uint16_t input_len;
    uint16_t scroll, about_scroll; /* lines hidden below the view / above it */
    uint32_t frame, busy_since, busy_ms;
    uint32_t (*clock_ms)(void *ctx);
    void *clock_ctx;
    uint16_t got_text;
    uint16_t hold;                 /* frames the current press has lasted */
    uint8_t layer, shift;          /* keyboard state */
    uint8_t busy, about, dirty;    /* dirty: bit0 top, bit1 bottom */
    uint8_t press_kind, press_a, press_b; /* control under the pen, if any */
    uint8_t prev_keys;
    uint8_t touching, lx, ly;     /* previous frame's pen state */
} chatds_ui;

/* Default monotonic clock. DS implementation reserves hardware timers 0/1. */
uint32_t chatds_ui_millis(void *ctx);
void chatds_ui_init(chatds_ui *ui, const chatds_ui_gen *gen);
/* Install a monotonic millisecond source before sending; NULL restores default.
 * Unsigned subtraction supports clock wrap (busy durations below 2^32 ms). */
void chatds_ui_set_clock(chatds_ui *ui, uint32_t (*clock_ms)(void *), void *ctx);
/* Process one frame of input and, while busy, advance the generator. */
void chatds_ui_frame(chatds_ui *ui, const chatds_ui_input *in);
/* Redraw whichever screens changed since the last call. Returns bitmask of
 * the screens written (1 top, 2 bottom). Pass NULL to skip a screen. */
int chatds_ui_render(chatds_ui *ui, uint16_t *top, uint16_t *bottom);
/* Redraw both screens regardless of change tracking. */
void chatds_ui_render_all(chatds_ui *ui, uint16_t *top, uint16_t *bottom);

int chatds_ui_is_busy(const chatds_ui *ui);
int chatds_ui_is_about(const chatds_ui *ui);
const char *chatds_ui_input_text(const chatds_ui *ui);
size_t chatds_ui_line_count(const chatds_ui *ui);
/* Copy wrapped log line i (0 = oldest) into buf, NUL-terminated, no role
 * marker. Returns its role, or 0 if i is out of range. */
int chatds_ui_line_text(const chatds_ui *ui, size_t i, char *buf, size_t cap);
size_t chatds_ui_view_scroll(const chatds_ui *ui);

/* Keyboard geometry, for scripted input. Touch point of the key that types
 * `ch` on `layer`; returns 0 and fills *x,*y, or -1 if ch is not typeable. */
#define CHATDS_LAYER_LETTERS 0
#define CHATDS_LAYER_SYMBOLS 1
int chatds_ui_key_for_char(int ch, int *layer, int *shift);
int chatds_ui_key_point(int layer, int code, int *x, int *y);
/* Special key codes accepted by chatds_ui_key_point. */
#define CHATDS_K_SHIFT 0x81
#define CHATDS_K_BKSP  0x82
#define CHATDS_K_LAYER 0x83
#define CHATDS_K_ENTER 0x84
/* Button-bar touch points. */
enum { CHATDS_BTN_SEND, CHATDS_BTN_STOP, CHATDS_BTN_UP, CHATDS_BTN_DOWN, CHATDS_BTN_ABOUT };
void chatds_ui_button_point(int button, int *x, int *y);

/* Credits text shown on the About screen (ASCII, '\n' separated). */
extern const char chatds_ui_credits[];

#ifdef __cplusplus
}
#endif
#endif
