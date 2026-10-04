/* SPDX-License-Identifier: MIT
 * Input handling and the busy/idle/about state machine.
 */
#include <string.h>

#include "ui_internal.h"

#define SCROLL_STEP 3
#define HOLD_DELAY 20  /* frames before a held scroll button repeats */
#define HOLD_RATE 4

enum { DIRTY_TOP = 1, DIRTY_BOT = 2 };

void chatds_ui_init(chatds_ui *ui, const chatds_ui_gen *gen)
{
    memset(ui, 0, sizeof *ui);
    ui->gen = gen;
    chatds_ui_set_clock(ui, NULL, NULL);
    ui->dirty = DIRTY_TOP | DIRTY_BOT;
    ui_log_clear(&ui->log);
    static const char hello[] = "ChatDS is ready. Type a question and press Send.";
    ui_log_begin(&ui->log, CHATDS_ROLE_NOTE);
    ui_log_append(&ui->log, hello, sizeof hello - 1);
}

void chatds_ui_set_clock(chatds_ui *ui, uint32_t (*clock_ms)(void *), void *ctx)
{
    ui->clock_ms = clock_ms ? clock_ms : chatds_ui_millis;
    ui->clock_ctx = ctx;
    (void)ui->clock_ms(ui->clock_ctx); /* Start hardware time before generation. */
}

int chatds_ui_is_busy(const chatds_ui *ui) { return ui->busy; }
int chatds_ui_is_about(const chatds_ui *ui) { return ui->about; }
const char *chatds_ui_input_text(const chatds_ui *ui) { return ui->input; }
size_t chatds_ui_line_count(const chatds_ui *ui) { return ui->log.nlines; }
size_t chatds_ui_view_scroll(const chatds_ui *ui) { return ui->scroll; }

int chatds_ui_line_text(const chatds_ui *ui, size_t i, char *buf, size_t cap)
{
    if (i >= ui->log.nlines || !cap) return 0;
    const chatds_ui_line *ln = &ui->log.lines[i];
    size_t n = ln->len < cap - 1 ? ln->len : cap - 1;
    memcpy(buf, ui->log.text + ln->off, n);
    buf[n] = 0;
    return ln->flags & LINE_ROLE_MASK;
}

void chatds_ui_button_point(int b, int *x, int *y)
{
    *x = (b * CHATDS_UI_W / BAR_N + (b + 1) * CHATDS_UI_W / BAR_N) / 2;
    *y = BAR_Y + BAR_H / 2;
}

/* ---- scrolling ---------------------------------------------------------------- */
static size_t log_max_scroll(const chatds_ui *ui)
{
    return ui->log.nlines > TOP_ROWS ? ui->log.nlines - TOP_ROWS : 0;
}

static void scroll_view(chatds_ui *ui, int dir) /* dir > 0: toward older / earlier text */
{
    if (ui->about) {
        size_t total = chatds_ui_credits_lines();
        size_t max = total > TOP_ROWS ? total - TOP_ROWS : 0;
        long v = (long)ui->about_scroll - dir * SCROLL_STEP;
        ui->about_scroll = (uint16_t)(v < 0 ? 0 : (size_t)v > max ? max : (size_t)v);
    } else {
        long v = (long)ui->scroll + dir * SCROLL_STEP;
        size_t max = log_max_scroll(ui);
        ui->scroll = (uint16_t)(v < 0 ? 0 : (size_t)v > max ? max : (size_t)v);
    }
    ui->dirty |= DIRTY_TOP;
}

/* ---- generation ----------------------------------------------------------------- */
static void finish(chatds_ui *ui)
{
    ui->busy = 0;
    ui->dirty |= DIRTY_TOP | DIRTY_BOT;
}

/* d lines were added below the view: a scrolled-up view moves with them. */
static void hold_view(chatds_ui *ui, int d)
{
    if (ui->scroll && d > 0) {
        size_t v = ui->scroll + (size_t)d, max = log_max_scroll(ui);
        ui->scroll = (uint16_t)(v > max ? max : v);
    }
    if (ui->scroll > log_max_scroll(ui)) ui->scroll = (uint16_t)log_max_scroll(ui);
    ui->dirty |= DIRTY_TOP;
}

static void log_text(chatds_ui *ui, const char *s, size_t n)
{
    hold_view(ui, ui_log_append(&ui->log, s, n));
}

static void note(chatds_ui *ui, const char *s)
{
    ui_log_begin(&ui->log, CHATDS_ROLE_NOTE);
    hold_view(ui, 1);
    log_text(ui, s, strlen(s));
}

static void send(chatds_ui *ui)
{
    if (ui->busy || ui->about || !ui->input_len || !ui->gen) return;
    uint32_t started = ui->clock_ms(ui->clock_ctx);
    if (ui->gen->begin(ui->gen->ctx, ui->input)) {
        note(ui, "[could not start]");
        return;
    }
    ui_log_begin(&ui->log, CHATDS_ROLE_USER);
    ui_log_append(&ui->log, ui->input, ui->input_len);
    ui_log_begin(&ui->log, CHATDS_ROLE_ANSWER);
    ui->input_len = 0;
    ui->input[0] = 0;
    ui->scroll = 0;
    ui->busy = 1;
    ui->busy_since = started;
    ui->busy_ms = 0;
    ui->got_text = 0;
    ui->dirty |= DIRTY_TOP | DIRTY_BOT;
}

static void cancel(chatds_ui *ui)
{
    if (!ui->busy) return;
    ui->gen->cancel(ui->gen->ctx);
    note(ui, "[cancelled]");
    finish(ui);
}

static void step(chatds_ui *ui)
{
    char buf[CHATDS_UI_STEP_TEXT];
    size_t n = 0;
    switch (ui->gen->step(ui->gen->ctx, buf, sizeof buf, &n)) {
    case CHATDS_GEN_TEXT:
        if (n > sizeof buf) n = sizeof buf;
        if (n) { ui->got_text = 1; log_text(ui, buf, n); }
        break;
    case CHATDS_GEN_DONE:
        if (!ui->got_text) log_text(ui, "(no answer)", 11);
        finish(ui);
        break;
    case CHATDS_GEN_ERROR: {
        char msg[CHATDS_UI_STEP_TEXT + 12] = "[error: ";
        if (n > sizeof buf) n = sizeof buf;
        memcpy(msg + 8, buf, n);
        msg[8 + n] = ']';
        msg[9 + n] = 0;
        note(ui, msg);
        finish(ui);
        break;
    }
    default:
        break;
    }
}

/* ---- touch ---------------------------------------------------------------------- */
static void type_key(chatds_ui *ui, int code)
{
    switch (code) {
    case K_SHIFT:
        if (ui->layer == CHATDS_LAYER_LETTERS) ui->shift ^= 1;
        break;
    case K_LAYER:
        ui->layer ^= 1;
        ui->shift = 0;
        break;
    case K_BKSP:
        if (ui->input_len) ui->input[--ui->input_len] = 0;
        break;
    case K_ENTER:
        send(ui);
        break;
    default:
        if (ui->input_len < CHATDS_UI_INPUT_MAX) {
            if (ui->shift && ui->layer == CHATDS_LAYER_LETTERS && code >= 'a' && code <= 'z')
                code -= 32;
            ui->input[ui->input_len++] = (char)code;
            ui->input[ui->input_len] = 0;
        }
        ui->shift = 0;
    }
    ui->dirty |= DIRTY_BOT;
}

static void press_button(chatds_ui *ui, int b)
{
    switch (b) {
    case CHATDS_BTN_SEND: send(ui); break;
    case CHATDS_BTN_STOP: cancel(ui); break;
    case CHATDS_BTN_UP: scroll_view(ui, 1); break;
    case CHATDS_BTN_DOWN: scroll_view(ui, -1); break;
    case CHATDS_BTN_ABOUT:
        ui->about ^= 1;
        ui->about_scroll = 0;
        break;
    }
    ui->dirty |= DIRTY_TOP | DIRTY_BOT;
}

/* What is under (x, y): sets kind/a/b; PRESS_DEAD for a disabled button. */
static void hit(const chatds_ui *ui, int x, int y, int *kind, int *a, int *b)
{
    *kind = PRESS_NONE;
    *a = *b = 0;
    if (y >= BAR_Y && y < BAR_Y + BAR_H) {
        int i = x * BAR_N / CHATDS_UI_W;
        *kind = ui_btn_enabled(ui, i) ? PRESS_BTN : PRESS_DEAD;
        *a = i;
    } else if (!ui->about) {
        int row, idx;
        if (ui_kbd_hit(ui->layer, x, y, &row, &idx)) {
            *kind = PRESS_KEY;
            *a = row;
            *b = idx;
        }
    }
}

static int is_scroll_button(int kind, int a)
{
    return kind == PRESS_BTN && (a == CHATDS_BTN_UP || a == CHATDS_BTN_DOWN);
}

static void touch(chatds_ui *ui, const chatds_ui_input *in)
{
    if (in->touch) {
        if (!ui->touching) {
            int k, a, b;
            hit(ui, in->x, in->y, &k, &a, &b);
            ui->press_kind = (uint8_t)k;
            ui->press_a = (uint8_t)a;
            ui->press_b = (uint8_t)b;
            ui->hold = 0;
            ui->dirty |= DIRTY_BOT;
            if (is_scroll_button(k, a)) press_button(ui, a); /* acts on press */
        } else {
            ui->hold++;
            if (is_scroll_button(ui->press_kind, ui->press_a) && ui->hold >= HOLD_DELAY &&
                (ui->hold - HOLD_DELAY) % HOLD_RATE == 0)
                press_button(ui, ui->press_a);
        }
        ui->lx = in->x;
        ui->ly = in->y;
    } else if (ui->touching) {
        int k, a, b;
        hit(ui, ui->lx, ui->ly, &k, &a, &b);
        /* Fire only if the pen is released on the control it went down on. */
        if (k == ui->press_kind && a == ui->press_a && b == ui->press_b) {
            if (k == PRESS_KEY) type_key(ui, ui_kbd_row(ui->layer, a)[b].code);
            else if (k == PRESS_BTN && !is_scroll_button(k, a)) press_button(ui, a);
        }
        ui->press_kind = PRESS_NONE;
        ui->dirty |= DIRTY_BOT;
    }
    ui->touching = in->touch;
}

void chatds_ui_frame(chatds_ui *ui, const chatds_ui_input *in)
{
    ui->frame++;

    uint8_t edge = (uint8_t)(in->keys & ~ui->prev_keys);
    ui->prev_keys = in->keys;
    if (edge & CHATDS_KEY_UP) scroll_view(ui, 1);
    if (edge & CHATDS_KEY_DOWN) scroll_view(ui, -1);
    if (edge & CHATDS_KEY_B) cancel(ui);
    if (edge & CHATDS_KEY_START) send(ui);
    if (edge & CHATDS_KEY_SELECT) press_button(ui, CHATDS_BTN_ABOUT);

    touch(ui, in);

    if (ui->busy) {
        step(ui);
    }
}

/* ---- rendering -------------------------------------------------------------------- */
int chatds_ui_render(chatds_ui *ui, uint16_t *top, uint16_t *bottom)
{
    int done = 0;
    if (ui->busy) {
        uint32_t elapsed = ui->clock_ms(ui->clock_ctx) - ui->busy_since;
        if (elapsed / 100 != ui->busy_ms / 100) ui->dirty |= DIRTY_TOP;
        ui->busy_ms = elapsed;
    }
    if ((ui->dirty & DIRTY_TOP) && top) { ui_render_top(ui, top); done |= 1; }
    if ((ui->dirty & DIRTY_BOT) && bottom) { ui_render_bottom(ui, bottom); done |= 2; }
    ui->dirty = (uint8_t)(ui->dirty & ~done);
    return done;
}

void chatds_ui_render_all(chatds_ui *ui, uint16_t *top, uint16_t *bottom)
{
    ui->dirty = DIRTY_TOP | DIRTY_BOT;
    chatds_ui_render(ui, top, bottom);
}
