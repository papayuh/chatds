/* SPDX-License-Identifier: MIT
 * Software renderer for both screens. Pure functions of the UI state.
 */
#include "ui_internal.h"

static const char SPIN[4] = {'|', '/', '-', '\\'};

static size_t slen(const char *s)
{
    size_t n = 0;
    while (s[n]) n++;
    return n;
}

int ui_btn_enabled(const chatds_ui *ui, int b)
{
    switch (b) {
    case CHATDS_BTN_SEND: return !ui->about && !ui->busy && ui->input_len > 0;
    case CHATDS_BTN_STOP: return ui->busy != 0;
    default: return 1;
    }
}

const char *ui_btn_label(const chatds_ui *ui, int b)
{
    static const char *const names[BAR_N] = {"Send", "Stop", "Up", "Down", "About"};
    return (b == CHATDS_BTN_ABOUT && ui->about) ? "Back" : names[b];
}

/* ---- credits (wrapped once, text never changes) ----------------------------- */
#define CREDIT_LINES 96
static chatds_ui_line credit_lines[CREDIT_LINES];
static size_t credit_n;

static size_t credits_wrap(void)
{
    if (!credit_n)
        credit_n = ui_wrap(chatds_ui_credits, slen(chatds_ui_credits), 0, TOP_COLS, 0,
                           credit_lines, CREDIT_LINES);
    return credit_n;
}

size_t chatds_ui_credits_lines(void) { return credits_wrap(); }

static void scrollbar(uint16_t *fb, size_t total, size_t first)
{
    int h = TOP_ROWS * ROW_H;
    ui_fill(fb, 252, TOP_Y, 3, h, COL_PANEL);
    if (total <= TOP_ROWS) return;
    int th = (int)(h * TOP_ROWS / total);
    if (th < 4) th = 4;
    int ty = (int)((h - th) * first / (total - TOP_ROWS));
    ui_fill(fb, 252, TOP_Y + ty, 3, th, COL_DIM);
}

static uint16_t role_color(int role)
{
    return role == CHATDS_ROLE_USER ? COL_USER : role == CHATDS_ROLE_NOTE ? COL_NOTE : COL_ANSWER;
}

static void header(const chatds_ui *ui, uint16_t *fb)
{
    ui_fill(fb, 0, 0, CHATDS_UI_W, HEADER_H, COL_PANEL);
    ui_fill(fb, 0, HEADER_H - 1, CHATDS_UI_W, 1, COL_EDGE);
    ui_text(fb, 4, 2, ui->about ? "ChatDS - about" : "ChatDS", ui->about ? 14 : 6, COL_TEXT);
    char s[24];
    size_t n = 0;
    uint16_t c = COL_DIM;
    if (ui->busy) {
        uint32_t t = ui->busy_ms;
        uint32_t sec = t / 1000;
        const char *w = ui->got_text ? "answering " : "thinking ";
        s[n++] = SPIN[(t / 100) & 3];
        s[n++] = ' ';
        while (*w) s[n++] = *w++;
        char d[6];
        int k = 0;
        do { d[k++] = (char)('0' + sec % 10); sec /= 10; } while (sec && k < 5);
        while (k) s[n++] = d[--k];
        s[n++] = 's';
        c = COL_BUSY;
    } else {
        const char *w = "ready";
        while (*w) s[n++] = *w++;
    }
    ui_text(fb, CHATDS_UI_W - 4 - (int)n * GLYPH_W, 2, s, n, c);
}

void ui_render_top(const chatds_ui *ui, uint16_t *fb)
{
    ui_fill(fb, 0, 0, CHATDS_UI_W, CHATDS_UI_H, COL_BG);
    header(ui, fb);
    if (ui->about) {
        size_t total = credits_wrap();
        size_t first = ui->about_scroll;
        if (first + TOP_ROWS > total) first = total > TOP_ROWS ? total - TOP_ROWS : 0;
        for (size_t r = 0; r < TOP_ROWS && first + r < total; r++) {
            const chatds_ui_line *ln = &credit_lines[first + r];
            ui_text(fb, 4, TOP_Y + (int)r * ROW_H, chatds_ui_credits + ln->off, ln->len, COL_TEXT);
        }
        scrollbar(fb, total, first);
        return;
    }
    const chatds_ui_log *l = &ui->log;
    size_t total = l->nlines;
    size_t hidden = ui->scroll;
    size_t first = total > TOP_ROWS + hidden ? total - TOP_ROWS - hidden : 0;
    for (size_t r = 0; r < TOP_ROWS && first + r < total; r++) {
        const chatds_ui_line *ln = &l->lines[first + r];
        int role = ln->flags & LINE_ROLE_MASK;
        int y = TOP_Y + (int)r * ROW_H;
        if (ln->flags & LINE_FIRST) {
            const char m = role == CHATDS_ROLE_USER ? 'Q' : role == CHATDS_ROLE_NOTE ? '!' : 'A';
            ui_text(fb, 3, y, &m, 1, role_color(role));
        }
        ui_text(fb, TOP_X, y, l->text + ln->off, ln->len, role_color(role));
    }
    scrollbar(fb, total, first);
}

static void button(uint16_t *fb, int x0, int x1, int y0, int h, uint16_t c, const char *label,
                   uint16_t tc)
{
    ui_fill(fb, x0, y0, x1 - x0, h, c);
    ui_text_centered(fb, x0, y0 + (h - 7) / 2, x1 - x0, label, tc);
}

void ui_render_bottom(const chatds_ui *ui, uint16_t *fb)
{
    ui_fill(fb, 0, 0, CHATDS_UI_W, CHATDS_UI_H, COL_BG);

    /* typed question: last IN_ROWS wrapped lines, caret after the last one */
    ui_fill(fb, 2, 0, CHATDS_UI_W - 4, IN_ROWS * 8 + 3, COL_PANEL);
    if (ui->input_len) {
        chatds_ui_line ln[CHATDS_UI_INPUT_MAX / 8 + 4];
        size_t n = ui_wrap(ui->input, ui->input_len, 0, IN_COLS, 0, ln, sizeof ln / sizeof ln[0]);
        size_t first = n > IN_ROWS ? n - IN_ROWS : 0;
        for (size_t r = first; r < n; r++)
            ui_text(fb, IN_X, IN_Y + (int)(r - first) * 8, ui->input + ln[r].off, ln[r].len, COL_TEXT);
        const chatds_ui_line *last = &ln[n - 1];
        if (last->len < IN_COLS)
            ui_text(fb, IN_X + last->len * GLYPH_W, IN_Y + (int)(n - 1 - first) * 8, "_", 1, COL_BUSY);
    } else {
        ui_text(fb, IN_X, IN_Y, "_ type a question", 17, COL_DIM);
    }

    for (int b = 0; b < BAR_N; b++) {
        int x0 = b * CHATDS_UI_W / BAR_N + 1, x1 = (b + 1) * CHATDS_UI_W / BAR_N - 1;
        int en = ui_btn_enabled(ui, b);
        uint16_t c = !en ? COL_OFF : b == CHATDS_BTN_SEND ? COL_OK : b == CHATDS_BTN_STOP ? COL_STOP : COL_KEY;
        if (en && ui->press_kind == PRESS_BTN && ui->press_a == b) c = COL_KEY_DOWN;
        button(fb, x0, x1, BAR_Y, BAR_H - 2, c, ui_btn_label(ui, b), en ? COL_TEXT : COL_DIM);
    }

    if (ui->about) {
        static const char *const hint[] = {"Credits and licenses.", "Up/Down scroll the list.",
                                           "Back returns to the chat."};
        for (int i = 0; i < 3; i++)
            ui_text_centered(fb, 0, KBD_Y + 24 + i * 14, CHATDS_UI_W, hint[i], COL_DIM);
        return;
    }

    for (int r = 0; r < KBD_ROWS; r++) {
        const ui_kdef *row = ui_kbd_row(ui->layer, r);
        for (int i = 0; row[i].w; i++) {
            int code = row[i].code;
            if (!code) continue;
            int x0, x1;
            ui_kbd_rect(ui->layer, r, i, &x0, &x1);
            int special = code >= 0x80;
            uint16_t c = special ? COL_KEY_SPECIAL : COL_KEY;
            if (code == K_SHIFT && ui->shift) c = COL_KEY_ON;
            if (code == K_LAYER && ui->layer) c = COL_KEY_ON;
            if (ui->press_kind == PRESS_KEY && ui->press_a == r && ui->press_b == i) c = COL_KEY_DOWN;
            char tmp[2];
            const char *lab = ui_kbd_label(code, ui->shift && ui->layer == CHATDS_LAYER_LETTERS, ui->layer, tmp);
            button(fb, x0 + 1, x1 - 1, KBD_Y + r * KBD_KEY_H + 1, KBD_KEY_H - 2, c, lab, COL_TEXT);
        }
    }
}
