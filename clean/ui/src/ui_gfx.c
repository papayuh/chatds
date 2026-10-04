/* SPDX-License-Identifier: MIT */
#include "ui_internal.h"

void ui_fill(uint16_t *fb, int x, int y, int w, int h, uint16_t c)
{
    if (x < 0) { w += x; x = 0; }
    if (y < 0) { h += y; y = 0; }
    if (x + w > CHATDS_UI_W) w = CHATDS_UI_W - x;
    if (y + h > CHATDS_UI_H) h = CHATDS_UI_H - y;
    for (int j = 0; j < h; j++) {
        uint16_t *p = fb + (y + j) * CHATDS_UI_W + x;
        for (int i = 0; i < w; i++) p[i] = c;
    }
}

static void glyph(uint16_t *fb, int x, int y, unsigned char ch, uint16_t c)
{
    if (ch < 32 || ch > 126) ch = '?';
    const uint8_t *g = chatds_ui_font[ch - 32];
    for (int r = 0; r < 8; r++) {
        int py = y + r;
        if (py < 0 || py >= CHATDS_UI_H) continue;
        for (int b = 0; b < 5; b++) {
            int px = x + b;
            if ((g[r] >> (4 - b)) & 1 && px >= 0 && px < CHATDS_UI_W)
                fb[py * CHATDS_UI_W + px] = c;
        }
    }
}

void ui_text(uint16_t *fb, int x, int y, const char *s, size_t n, uint16_t c)
{
    for (size_t i = 0; i < n; i++) glyph(fb, x + (int)i * GLYPH_W, y, (unsigned char)s[i], c);
}

void ui_text_centered(uint16_t *fb, int x, int y, int w, const char *s, uint16_t c)
{
    size_t n = 0;
    while (s[n]) n++;
    int tw = (int)n * GLYPH_W - 1;
    ui_text(fb, x + (w - tw) / 2, y, s, n, c);
}
