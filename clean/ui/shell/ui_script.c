/* SPDX-License-Identifier: MIT */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chatds_ui_demo.h"

typedef struct {
    chatds_ui *ui;
    const chatds_script_host *host;
    chatds_ui_input in;
} runner;

static void frame(runner *r)
{
    if (r->host->tick) r->host->tick(r->host->ctx);
    chatds_ui_frame(r->ui, &r->in);
    chatds_ui_render(r->ui, r->host->top, r->host->bottom);
}

static void idle(runner *r, int n)
{
    r->in.touch = 0;
    while (n-- > 0) frame(r);
}

static void hold(runner *r, int x, int y, int n)
{
    r->in.touch = 1;
    r->in.x = (uint8_t)x;
    r->in.y = (uint8_t)y;
    while (n-- > 0) frame(r);
    idle(r, 2); /* release, then a quiet frame */
}

static void tap(runner *r, int x, int y) { hold(r, x, y, 3); }

static void tap_key(runner *r, int layer, int code)
{
    int x, y;
    if (chatds_ui_key_point(layer, code, &x, &y) == 0) tap(r, x, y);
}

static int type_text(runner *r, const char *s, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        int layer, shift;
        if (chatds_ui_key_for_char((unsigned char)s[i], &layer, &shift)) return -1;
        if (r->ui->layer != layer) tap_key(r, r->ui->layer, CHATDS_K_LAYER);
        if (layer == CHATDS_LAYER_LETTERS && r->ui->shift != shift)
            tap_key(r, layer, CHATDS_K_SHIFT);
        tap_key(r, layer, shift ? s[i] + 32 : s[i]);
    }
    return 0;
}

static int word(const char **p, char *out, size_t cap)
{
    while (**p == ' ') (*p)++;
    size_t n = 0;
    while (**p && **p != ' ' && **p != '\n' && **p != '\r') {
        if (n + 1 < cap) out[n++] = **p;
        (*p)++;
    }
    out[n] = 0;
    return n != 0;
}

/* Script playback uses a deterministic 60 Hz clock on both host and DS,
 * so screen comparisons do not depend on emulator speed or FAT I/O latency. */
static uint32_t script_millis(void *ctx)
{
    const chatds_ui *ui = ctx;
    return (uint32_t)((uint64_t)ui->frame * 1000 / 60);
}

int chatds_script_run(chatds_ui *ui, const char *script, const chatds_script_host *host)
{
    chatds_ui_set_clock(ui, script_millis, ui);
    runner r = {ui, host, {0, 0, 0, 0}};
    int lineno = 0;
    chatds_ui_render_all(ui, host->top, host->bottom); /* screens exist before frame 1 */
    for (const char *p = script; *p;) {
        const char *eol = strchr(p, '\n');
        size_t linelen = eol ? (size_t)(eol - p) : strlen(p);
        const char *next = eol ? eol + 1 : p + linelen;
        lineno++;
        const char *q = p;
        const char *end = p + linelen;
        while (q < end && *q == ' ') q++;
        if (q == end || *q == '#' || *q == '\r') { p = next; continue; }

        char cmd[16], a[16], b[16];
        const char *args = q;
        if (!word(&args, cmd, sizeof cmd)) return lineno;
        int ok = 1;
        if (!strcmp(cmd, "type")) {
            while (*args == ' ') args++;
            size_t n = (size_t)(end - args);
            while (n && args[n - 1] == '\r') n--;
            ok = type_text(&r, args, n) == 0;
        } else if (!strcmp(cmd, "wait")) {
            word(&args, a, sizeof a);
            idle(&r, atoi(a));
        } else if (!strcmp(cmd, "tap")) {
            word(&args, a, sizeof a);
            word(&args, b, sizeof b);
            tap(&r, atoi(a), atoi(b));
        } else if (!strcmp(cmd, "hold")) {
            char c[16];
            word(&args, a, sizeof a);
            word(&args, b, sizeof b);
            word(&args, c, sizeof c);
            hold(&r, atoi(a), atoi(b), atoi(c));
        } else if (!strcmp(cmd, "button")) {
            static const char *const names[] = {"send", "stop", "up", "down", "about"};
            int i = 0, x, y;
            word(&args, a, sizeof a);
            while (i < 5 && strcmp(a, names[i])) i++;
            ok = i < 5;
            if (ok) {
                chatds_ui_button_point(i, &x, &y);
                tap(&r, x, y);
            }
        } else if (!strcmp(cmd, "key")) {
            static const struct { const char *n; uint8_t bit; } keys[] = {
                {"up", CHATDS_KEY_UP}, {"down", CHATDS_KEY_DOWN}, {"b", CHATDS_KEY_B},
                {"start", CHATDS_KEY_START}, {"select", CHATDS_KEY_SELECT}};
            int i = 0;
            word(&args, a, sizeof a);
            while (i < 5 && strcmp(a, keys[i].n)) i++;
            ok = i < 5;
            if (ok) {
                r.in.keys = keys[i].bit;
                frame(&r);
                r.in.keys = 0;
                frame(&r);
            }
        } else if (!strcmp(cmd, "until")) {
            word(&args, a, sizeof a);
            word(&args, b, sizeof b);
            ok = !strcmp(a, "idle");
            for (int n = atoi(b); ok && n > 0 && chatds_ui_is_busy(ui); n--) idle(&r, 1);
        } else if (!strcmp(cmd, "dump")) {
            word(&args, a, sizeof a);
            ok = a[0] != 0;
            if (ok) host->dump(host->ctx, a, ui);
        } else {
            ok = 0;
        }
        if (!ok) return lineno;
        p = next;
    }
    return 0;
}

void chatds_script_state_text(const chatds_ui *ui, void (*sink)(void *ctx, const char *line),
                              void *ctx)
{
    char buf[CHATDS_UI_INPUT_MAX + 32];
    static const char role[] = " QA!";
    sprintf(buf, "busy=%d about=%d scroll=%u layer=%d shift=%d", chatds_ui_is_busy(ui),
            chatds_ui_is_about(ui), (unsigned)chatds_ui_view_scroll(ui), ui->layer, ui->shift);
    sink(ctx, buf);
    sprintf(buf, "input=%s", chatds_ui_input_text(ui));
    sink(ctx, buf);
    sprintf(buf, "lines=%u", (unsigned)chatds_ui_line_count(ui));
    sink(ctx, buf);
    for (size_t i = 0; i < chatds_ui_line_count(ui); i++) {
        char t[64];
        int rl = chatds_ui_line_text(ui, i, t, sizeof t);
        char out[80];
        sprintf(out, "%c|%s", role[rl], t);
        sink(ctx, out);
    }
}
