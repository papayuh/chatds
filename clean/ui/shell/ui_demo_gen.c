/* SPDX-License-Identifier: MIT */
#include <string.h>

#include "chatds_ui_demo.h"

enum { MODE_TEXT, MODE_FAIL, MODE_EMPTY };

static const char LONG_ANSWER[] =
    "This is a long canned answer used to test word wrapping and scrolling on the "
    "chat screen. Short words, then a-very-long-hyphenated-word-that-cannot-possibly-fit-on-one-line, "
    "then more text.\n"
    "\n"
    "Second paragraph after a blank line. The log keeps streaming while you scroll "
    "up, and the view stays where you left it until you scroll back down.\n"
    "Third paragraph, one more line break above. The quick brown fox jumps over the "
    "lazy dog, again and again, so the answer fills several screens of text. "
    "Pack my box with five dozen liquor jugs.\n"
    "\n"
    "A list:\n"
    "1. first item\n"
    "2. second item, a bit longer so that it wraps around the edge of the screen\n"
    "3. third item\n"
    "\n"
    "Last paragraph: how vexingly quick daft zebras jump, and the sixth sheik's sixth "
    "sheep is sick. This is the end of the long canned answer.";

static int demo_begin(void *ctx, const char *q)
{
    chatds_demo_gen *d = ctx;
    d->pos = 0;
    d->prefill = 3;
    d->mode = MODE_TEXT;
    if (!strncmp(q, "long", 4)) {
        d->len = sizeof LONG_ANSWER - 1;
        memcpy(d->answer, LONG_ANSWER, d->len);
    } else if (!strncmp(q, "fail", 4)) {
        d->mode = MODE_FAIL;
        d->len = 0;
    } else if (!strncmp(q, "empty", 5)) {
        d->mode = MODE_EMPTY;
        d->len = 0;
    } else {
        static const char pre[] = "You asked: ";
        size_t n = strlen(q);
        if (n > 200) n = 200;
        memcpy(d->answer, pre, sizeof pre - 1);
        memcpy(d->answer + sizeof pre - 1, q, n);
        memcpy(d->answer + sizeof pre - 1 + n, ". Canned demo answer.", 21);
        d->len = (uint16_t)(sizeof pre - 1 + n + 21);
    }
    return 0;
}

static chatds_gen_event demo_step(void *ctx, char *out, size_t cap, size_t *len)
{
    chatds_demo_gen *d = ctx;
    if (d->mode == MODE_FAIL && d->prefill <= 1) {
        memcpy(out, "demo failure", 12);
        *len = 12;
        return CHATDS_GEN_ERROR;
    }
    if (d->prefill) {
        d->prefill--;
        return CHATDS_GEN_PROGRESS;
    }
    if (d->pos >= d->len) return CHATDS_GEN_DONE;
    size_t n = d->len - d->pos < 3 ? d->len - d->pos : 3;
    if (n > cap) n = cap;
    memcpy(out, d->answer + d->pos, n);
    d->pos = (uint16_t)(d->pos + n);
    *len = n;
    return CHATDS_GEN_TEXT;
}

static void demo_cancel(void *ctx)
{
    chatds_demo_gen *d = ctx;
    d->pos = d->len;
}

void chatds_demo_gen_init(chatds_demo_gen *d)
{
    memset(d, 0, sizeof *d);
    d->gen.ctx = d;
    d->gen.begin = demo_begin;
    d->gen.step = demo_step;
    d->gen.cancel = demo_cancel;
}
