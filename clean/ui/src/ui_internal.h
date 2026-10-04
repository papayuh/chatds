/* SPDX-License-Identifier: MIT
 * Private to clean/ui/src. */
#ifndef CHATDS_UI_INTERNAL_H
#define CHATDS_UI_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "chatds_ui.h"

/* ---- layout (DS pixels) -------------------------------------------------- */
#define GLYPH_W 6          /* 5 px glyph + 1 px gap */
#define ROW_H 9            /* chat/credits line pitch */
#define HEADER_H 12        /* top-screen header incl. rule */
#define TOP_X 14           /* chat text origin (after role marker column) */
#define TOP_Y 14
#define TOP_ROWS 19
#define TOP_COLS 39        /* wrap width of the log and credits */

#define IN_X 5
#define IN_Y 2
#define IN_COLS 41         /* input box wrap width */
#define IN_ROWS 4
#define BAR_Y 37
#define BAR_H 24
#define BAR_N 5
#define KBD_Y 66
#define KBD_KEY_H 25
#define KBD_ROWS 5

/* Keyboard key codes: printable ASCII is itself, 0 is padding. */
#define K_SHIFT CHATDS_K_SHIFT
#define K_BKSP  CHATDS_K_BKSP
#define K_LAYER CHATDS_K_LAYER
#define K_ENTER CHATDS_K_ENTER

typedef struct { uint8_t code, w; } ui_kdef; /* w == 0 ends a row */

/* ---- font / gfx ------------------------------------------------------------ */
extern const uint8_t chatds_ui_font[95][8];

#define RGB15(r, g, b) ((uint16_t)(0x8000 | ((r) | ((g) << 5) | ((b) << 10))))
enum {
    COL_BG = RGB15(1, 2, 5),
    COL_PANEL = RGB15(3, 5, 10),
    COL_TEXT = RGB15(28, 28, 29),
    COL_DIM = RGB15(13, 14, 17),
    COL_USER = RGB15(14, 24, 31),
    COL_ANSWER = RGB15(29, 29, 27),
    COL_NOTE = RGB15(31, 25, 8),
    COL_KEY = RGB15(7, 9, 15),
    COL_KEY_SPECIAL = RGB15(5, 6, 11),
    COL_KEY_DOWN = RGB15(12, 22, 31),
    COL_KEY_ON = RGB15(10, 18, 12),
    COL_EDGE = RGB15(2, 3, 6),
    COL_OK = RGB15(8, 24, 12),
    COL_STOP = RGB15(26, 6, 6),
    COL_BUSY = RGB15(31, 22, 6),
    COL_OFF = RGB15(4, 5, 8)
};

void ui_fill(uint16_t *fb, int x, int y, int w, int h, uint16_t c);
void ui_text(uint16_t *fb, int x, int y, const char *s, size_t n, uint16_t c);
void ui_text_centered(uint16_t *fb, int x, int y, int w, const char *s, uint16_t c);

/* ---- word wrap -------------------------------------------------------------- */
#define LINE_ROLE_MASK 0x03
#define LINE_FIRST 0x80
/* Wrap s[0..n) at `cols`; line offsets are base + index. Returns lines written
 * (stops at cap). Newlines split paragraphs; words longer than cols break hard. */
size_t ui_wrap(const char *s, size_t n, size_t base, size_t cols, uint8_t flags,
               chatds_ui_line *out, size_t cap);

/* ---- chat log ---------------------------------------------------------------- */
void ui_log_clear(chatds_ui_log *l);
void ui_log_begin(chatds_ui_log *l, chatds_ui_role role);
/* Appends sanitised text to the newest message; returns the change in that
 * message's line count (may be negative when a reflow shrinks it). Lines evicted
 * from the top are not counted. */
int ui_log_append(chatds_ui_log *l, const char *s, size_t n);
size_t ui_log_msg_len(const chatds_ui_log *l); /* bytes in the newest message */

/* ---- keyboard ---------------------------------------------------------------- */
const ui_kdef *ui_kbd_row(int layer, int row);
/* Hit test; fills layer-relative row/index and returns the key code, or 0. */
int ui_kbd_hit(int layer, int x, int y, int *row, int *idx);
void ui_kbd_rect(int layer, int row, int idx, int *x0, int *x1);
const char *ui_kbd_label(int code, int shift, int layer, char *tmp);

/* ---- render ------------------------------------------------------------------ */
void ui_render_top(const chatds_ui *ui, uint16_t *fb);
void ui_render_bottom(const chatds_ui *ui, uint16_t *fb);
size_t chatds_ui_credits_lines(void);
int ui_btn_enabled(const chatds_ui *ui, int button);
const char *ui_btn_label(const chatds_ui *ui, int button);

/* press_kind values */
enum { PRESS_NONE = 0, PRESS_KEY, PRESS_BTN, PRESS_DEAD };

#endif
