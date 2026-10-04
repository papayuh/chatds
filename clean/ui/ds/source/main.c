/* SPDX-License-Identifier: MIT
 * DS shell for the ChatDS touch UI (BlocksDS / libnds).
 *
 * Both screens are 16-bit bitmap backgrounds and the UI renders straight into
 * VRAM, so the UI costs no framebuffer RAM. Normal run: pen and keys drive the
 * UI at 60 Hz. Test run: if fat:/chatds-ui/script.txt exists, the script is
 * played through the same UI code instead of the hardware, the screens are
 * dumped to fat:/chatds-ui/ and fat:/chatds-ui/result.txt ends in status=OK.
 *
 * The answer source here is the canned demo generator. The engine is wired in
 * by building a chatds_ui_engine_gen instead (see clean/ui/README.md).
 */
#include <fat.h>
#include <nds.h>
#include <stdio.h>
#include <stdlib.h>

#include "chatds_ui_demo.h"

#define DIR "fat:/chatds-ui/"

static chatds_ui ui;
static chatds_demo_gen demo;
static uint16_t *top, *bottom;

static void vblank(void *ctx)
{
    (void)ctx;
    swiWaitForVBlank();
}

/* ---- scripted test run --------------------------------------------------------- */
static FILE *txt;
static int dumps;

static void state_line(void *ctx, const char *s)
{
    (void)ctx;
    fprintf(txt, "%s\n", s);
}

static void write_screen(const char *name, const char *ext, const uint16_t *fb)
{
    char path[64];
    snprintf(path, sizeof path, DIR "%s.%s", name, ext);
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fwrite(fb, 2, CHATDS_UI_W * CHATDS_UI_H, f);
    fclose(f);
}

static void dump(void *ctx, const char *name, chatds_ui *u)
{
    (void)ctx;
    char path[64];
    write_screen(name, "top", top);
    write_screen(name, "bot", bottom);
    snprintf(path, sizeof path, DIR "%s.txt", name);
    txt = fopen(path, "w");
    if (txt) {
        chatds_script_state_text(u, state_line, NULL);
        fclose(txt);
    }
    dumps++;
}

static int run_script(void)
{
    FILE *f = fopen(DIR "script.txt", "rb");
    if (!f) return 0;
    static char script[16384];
    size_t n = fread(script, 1, sizeof script - 1, f);
    script[n] = 0;
    fclose(f);

    chatds_script_host host = {NULL, vblank, dump, top, bottom};
    int bad = chatds_script_run(&ui, script, &host);

    f = fopen(DIR "result.txt", "w");
    if (f) {
        fprintf(f, "frames=%lu\ndumps=%d\nbad_line=%d\n%s\n", (unsigned long)ui.frame, dumps, bad,
                bad ? "status=FAIL" : "status=OK");
        fclose(f);
    }
    return 1;
}

/* ---- normal run ------------------------------------------------------------------ */
static void interactive(void)
{
    for (;;) {
        swiWaitForVBlank();
        scanKeys();
        uint32_t held = keysHeld();
        chatds_ui_input in = {0, 0, 0, 0};
        if (held & KEY_UP) in.keys |= CHATDS_KEY_UP;
        if (held & KEY_DOWN) in.keys |= CHATDS_KEY_DOWN;
        if (held & KEY_B) in.keys |= CHATDS_KEY_B;
        if (held & KEY_START) in.keys |= CHATDS_KEY_START;
        if (held & KEY_SELECT) in.keys |= CHATDS_KEY_SELECT;
        if (held & KEY_TOUCH) {
            touchPosition t;
            touchRead(&t);
            in.touch = 1;
            in.x = (uint8_t)t.px;
            in.y = (uint8_t)t.py;
        }
        chatds_ui_frame(&ui, &in);
        /* ponytail: draws into the visible VRAM; a big redraw can tear for a frame.
         * Add a second page and flip if it shows on hardware. */
        chatds_ui_render(&ui, top, bottom);
    }
}

int main(void)
{
    videoSetMode(MODE_5_2D);
    videoSetModeSub(MODE_5_2D);
    vramSetBankA(VRAM_A_MAIN_BG);
    vramSetBankC(VRAM_C_SUB_BG);
    int bg_main = bgInit(3, BgType_Bmp16, BgSize_B16_256x256, 0, 0);
    int bg_sub = bgInitSub(3, BgType_Bmp16, BgSize_B16_256x256, 0, 0);
    top = bgGetGfxPtr(bg_main);
    bottom = bgGetGfxPtr(bg_sub);
    lcdMainOnTop();

    chatds_demo_gen_init(&demo);
    chatds_ui_init(&ui, &demo.gen);
    chatds_ui_render_all(&ui, top, bottom);

    if (fatInitDefault() && run_script())
        for (;;) swiWaitForVBlank();
    interactive();
    return 0;
}
