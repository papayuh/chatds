/* SPDX-License-Identifier: MIT
 * Run a UI script on the host with the demo generator and write what the DS
 * would show: ui_sim SCRIPT OUTDIR. Per dump NAME it writes NAME.top / NAME.bot
 * (raw 256x192 ARGB1555, little endian), NAME.txt (state summary) and
 * NAME.ppm (both screens stacked, for looking at). */
#include <stdio.h>
#include <stdlib.h>

#include "chatds_ui_demo.h"

static uint16_t top[CHATDS_UI_W * CHATDS_UI_H], bot[CHATDS_UI_W * CHATDS_UI_H];
static const char *outdir;
static FILE *txt;

static void line(void *ctx, const char *s)
{
    (void)ctx;
    fprintf(txt, "%s\n", s);
}

static void dump(void *ctx, const char *name, chatds_ui *ui)
{
    (void)ctx;
    char path[512];
    snprintf(path, sizeof path, "%s/%s.top", outdir, name);
    FILE *f = fopen(path, "wb");
    if (f) { fwrite(top, 2, sizeof top / 2, f); fclose(f); }
    snprintf(path, sizeof path, "%s/%s.bot", outdir, name);
    f = fopen(path, "wb");
    if (f) { fwrite(bot, 2, sizeof bot / 2, f); fclose(f); }
    snprintf(path, sizeof path, "%s/%s.ppm", outdir, name);
    f = fopen(path, "wb");
    if (f) {
        fprintf(f, "P6\n256 384\n255\n");
        const uint16_t *fbs[2] = {top, bot};
        for (int s = 0; s < 2; s++)
            for (int i = 0; i < CHATDS_UI_W * CHATDS_UI_H; i++) {
                uint16_t p = fbs[s][i];
                unsigned char rgb[3] = {(unsigned char)((p & 31) * 255 / 31),
                                        (unsigned char)(((p >> 5) & 31) * 255 / 31),
                                        (unsigned char)(((p >> 10) & 31) * 255 / 31)};
                fwrite(rgb, 1, 3, f);
            }
        fclose(f);
    }
    snprintf(path, sizeof path, "%s/%s.txt", outdir, name);
    txt = fopen(path, "w");
    if (txt) { chatds_script_state_text(ui, line, NULL); fclose(txt); }
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: ui_sim SCRIPT OUTDIR\n");
        return 2;
    }
    outdir = argv[2];
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 2; }
    static char script[16384];
    size_t n = fread(script, 1, sizeof script - 1, f);
    script[n] = 0;
    fclose(f);

    static chatds_demo_gen gen;
    static chatds_ui ui;
    chatds_demo_gen_init(&gen);
    chatds_ui_init(&ui, &gen.gen);
    chatds_script_host host = {NULL, NULL, dump, top, bot};
    int bad = chatds_script_run(&ui, script, &host);
    if (bad) fprintf(stderr, "ui_sim: script line %d failed\n", bad);
    return bad ? 1 : 0;
}
