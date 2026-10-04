/* SPDX-License-Identifier: MIT
 * On-screen keyboard geometry. Key widths are weights; each row is stretched to
 * the full 256 px, so rows with different key counts still tile edge to edge.
 */
#include "ui_internal.h"

#define PAD(w) {0, w}
#define KEY(c, w) {(uint8_t)(c), w}
#define END {0, 0}

static const ui_kdef LETTERS[KBD_ROWS][12] = {
    {KEY('1',10),KEY('2',10),KEY('3',10),KEY('4',10),KEY('5',10),KEY('6',10),KEY('7',10),KEY('8',10),KEY('9',10),KEY('0',10),END},
    {KEY('q',10),KEY('w',10),KEY('e',10),KEY('r',10),KEY('t',10),KEY('y',10),KEY('u',10),KEY('i',10),KEY('o',10),KEY('p',10),END},
    {PAD(5),KEY('a',10),KEY('s',10),KEY('d',10),KEY('f',10),KEY('g',10),KEY('h',10),KEY('j',10),KEY('k',10),KEY('l',10),PAD(5),END},
    {KEY(K_SHIFT,15),KEY('z',10),KEY('x',10),KEY('c',10),KEY('v',10),KEY('b',10),KEY('n',10),KEY('m',10),KEY(K_BKSP,15),END},
    {KEY(K_LAYER,15),KEY(',',10),KEY(' ',35),KEY('.',10),KEY('?',10),KEY(K_ENTER,20),END},
};

static const ui_kdef SYMBOLS[KBD_ROWS][12] = {
    {KEY('1',10),KEY('2',10),KEY('3',10),KEY('4',10),KEY('5',10),KEY('6',10),KEY('7',10),KEY('8',10),KEY('9',10),KEY('0',10),END},
    {KEY('!',10),KEY('@',10),KEY('#',10),KEY('$',10),KEY('%',10),KEY('^',10),KEY('&',10),KEY('*',10),KEY('(',10),KEY(')',10),END},
    {KEY('-',10),KEY('_',10),KEY('=',10),KEY('+',10),KEY('[',10),KEY(']',10),KEY('{',10),KEY('}',10),KEY('\\',10),KEY('|',10),END},
    {KEY('~',9),KEY('`',9),KEY('<',9),KEY('>',9),KEY(':',9),KEY(';',9),KEY('"',9),KEY('\'',9),KEY('/',9),KEY(K_BKSP,19),END},
    {KEY(K_LAYER,15),KEY(',',10),KEY(' ',35),KEY('.',10),KEY('?',10),KEY(K_ENTER,20),END},
};

const ui_kdef *ui_kbd_row(int layer, int row)
{
    return (layer == CHATDS_LAYER_SYMBOLS ? SYMBOLS : LETTERS)[row];
}

static int row_total(const ui_kdef *r)
{
    int t = 0;
    for (; r->w; r++) t += r->w;
    return t;
}

void ui_kbd_rect(int layer, int row, int idx, int *x0, int *x1)
{
    const ui_kdef *r = ui_kbd_row(layer, row);
    int total = row_total(r), cum = 0;
    for (int i = 0; i < idx; i++) cum += r[i].w;
    *x0 = CHATDS_UI_W * cum / total;
    *x1 = CHATDS_UI_W * (cum + r[idx].w) / total;
}

int ui_kbd_hit(int layer, int x, int y, int *row, int *idx)
{
    if (y < KBD_Y || y >= KBD_Y + KBD_ROWS * KBD_KEY_H) return 0;
    int rr = (y - KBD_Y) / KBD_KEY_H;
    const ui_kdef *r = ui_kbd_row(layer, rr);
    for (int i = 0; r[i].w; i++) {
        int x0, x1;
        ui_kbd_rect(layer, rr, i, &x0, &x1);
        if (x >= x0 && x < x1) {
            if (!r[i].code) return 0;
            *row = rr;
            *idx = i;
            return r[i].code;
        }
    }
    return 0;
}

const char *ui_kbd_label(int code, int shift, int layer, char *tmp)
{
    switch (code) {
    case K_SHIFT: return shift ? "SHIFT" : "Shift";
    case K_BKSP: return "Del";
    case K_LAYER: return layer == CHATDS_LAYER_LETTERS ? "?123" : "abc";
    case K_ENTER: return "Send";
    case ' ': return "space";
    }
    tmp[0] = (char)((shift && code >= 'a' && code <= 'z') ? code - 32 : code);
    tmp[1] = 0;
    return tmp;
}

int chatds_ui_key_for_char(int ch, int *layer, int *shift)
{
    if (ch < 32 || ch > 126) return -1;
    for (int l = 0; l < 2; l++)
        for (int r = 0; r < KBD_ROWS; r++)
            for (const ui_kdef *k = ui_kbd_row(l, r); k->w; k++) {
                int c = k->code;
                if (c == ch || (l == CHATDS_LAYER_LETTERS && c >= 'a' && c <= 'z' && c - 32 == ch)) {
                    *layer = l;
                    *shift = (c != ch);
                    return 0;
                }
            }
    return -1;
}

int chatds_ui_key_point(int layer, int code, int *x, int *y)
{
    for (int r = 0; r < KBD_ROWS; r++) {
        const ui_kdef *row = ui_kbd_row(layer, r);
        for (int i = 0; row[i].w; i++)
            if (row[i].code == code) {
                int x0, x1;
                ui_kbd_rect(layer, r, i, &x0, &x1);
                *x = (x0 + x1) / 2;
                *y = KBD_Y + r * KBD_KEY_H + KBD_KEY_H / 2;
                return 0;
            }
    }
    return -1;
}
