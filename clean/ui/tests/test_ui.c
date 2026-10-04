/* SPDX-License-Identifier: MIT
 * Host tests for the UI core, the scripted-input driver and the engine adapter
 * (against a stub engine). Plain asserts; exits nonzero on any failure. */
#include <stdio.h>
#include <string.h>

#include "chatds_ui_demo.h"
#include "chatds_ui_engine.h"
#include "stub_engine.h"
#include "ui_internal.h"

static int checks, fails;
#define CHECK(c) do { checks++; if (!(c)) { fails++; printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); } } while (0)

static chatds_demo_gen dg;
static chatds_ui ui;
static uint16_t top[CHATDS_UI_W * CHATDS_UI_H], bot[CHATDS_UI_W * CHATDS_UI_H];

static void nodump(void *c, const char *n, chatds_ui *u) { (void)c; (void)n; (void)u; }
static chatds_script_host host = {NULL, NULL, nodump, top, bot};

static void setup(void)
{
    chatds_demo_gen_init(&dg);
    chatds_ui_init(&ui, &dg.gen);
}

static void run(const char *script)
{
    int bad = chatds_script_run(&ui, script, &host);
    CHECK(bad == 0);
    if (bad) printf("  script line %d: %s\n", bad, script);
}

static void frames(int n)
{
    chatds_ui_input in = {0, 0, 0, 0};
    while (n-- > 0) {
        chatds_ui_frame(&ui, &in);
        chatds_ui_render(&ui, top, bot);
    }
}

static void pen(int down, int x, int y)
{
    chatds_ui_input in = {(uint8_t)down, (uint8_t)x, (uint8_t)y, 0};
    chatds_ui_frame(&ui, &in);
    chatds_ui_render(&ui, top, bot);
}

static int has_line(const char *needle, int role)
{
    char t[64];
    for (size_t i = 0; i < chatds_ui_line_count(&ui); i++) {
        int r = chatds_ui_line_text(&ui, i, t, sizeof t);
        if ((!role || r == role) && strstr(t, needle)) return 1;
    }
    return 0;
}

static size_t answer_len(void) /* characters in the newest answer message */
{
    size_t n = 0;
    char t[64];
    for (size_t i = 0; i < chatds_ui_line_count(&ui); i++)
        if (chatds_ui_line_text(&ui, i, t, sizeof t) == CHATDS_ROLE_ANSWER) n += strlen(t);
    return n;
}

/* ---- wrap ------------------------------------------------------------------------ */
static int wrap_check(const char *s, size_t cols, const char *const *want, size_t nwant)
{
    chatds_ui_line ln[16];
    size_t n = ui_wrap(s, strlen(s), 0, cols, 0, ln, 16);
    if (n != nwant) return 0;
    for (size_t i = 0; i < n; i++)
        if (ln[i].len != strlen(want[i]) || memcmp(s + ln[i].off, want[i], ln[i].len)) return 0;
    return 1;
}

static void test_wrap(void)
{
    CHECK(wrap_check("hello world", 5, (const char *[]){"hello", "world"}, 2));
    CHECK(wrap_check("abcd efgh", 9, (const char *[]){"abcd efgh"}, 1));
    CHECK(wrap_check("aaaaaaaaaaaa", 5, (const char *[]){"aaaaa", "aaaaa", "aa"}, 3));
    CHECK(wrap_check("ab\n\ncd", 5, (const char *[]){"ab", "", "cd"}, 3));
    CHECK(wrap_check("", 5, (const char *[]){""}, 1));
    CHECK(wrap_check("hello ", 5, (const char *[]){"hello"}, 1));
    CHECK(wrap_check("a  b", 1, (const char *[]){"a", "b"}, 2));
    CHECK(wrap_check("  indented", 20, (const char *[]){"  indented"}, 1));
    CHECK(wrap_check("one two three", 7, (const char *[]){"one two", "three"}, 2));
    chatds_ui_line ln[2]; /* capacity is respected */
    CHECK(ui_wrap("a b c d e f", 11, 0, 1, 0, ln, 2) == 2);
}

/* ---- log ------------------------------------------------------------------------- */
static chatds_ui_log lg;

static void test_log(void)
{
    ui_log_clear(&lg);
    ui_log_begin(&lg, CHATDS_ROLE_ANSWER);
    const char *s = "streamed text arrives in tiny pieces and still wraps correctly at word ends";
    for (size_t i = 0; s[i]; i += 3) ui_log_append(&lg, s + i, strlen(s + i) < 3 ? strlen(s + i) : 3);
    chatds_ui_log lg2 = lg;
    ui_log_clear(&lg2);
    ui_log_begin(&lg2, CHATDS_ROLE_ANSWER);
    ui_log_append(&lg2, s, strlen(s));
    CHECK(lg.nlines == lg2.nlines && !memcmp(lg.lines, lg2.lines, lg.nlines * sizeof lg.lines[0]));
    CHECK(lg.lines[0].flags & LINE_FIRST);

    /* sanitising: CR dropped, tab -> space, control and non-ASCII -> '?' */
    ui_log_clear(&lg);
    ui_log_begin(&lg, CHATDS_ROLE_NOTE);
    ui_log_append(&lg, "a\r\tb\x01\xff", 6);
    CHECK(lg.used == 1 + 5 && !memcmp(lg.text + 1, "a b??", 5));

    /* eviction: oldest go first, newest message stays whole, tables stay valid */
    ui_log_clear(&lg);
    char msg[60];
    for (int i = 0; i < 400; i++) {
        ui_log_begin(&lg, i & 1 ? CHATDS_ROLE_ANSWER : CHATDS_ROLE_USER);
        int n = snprintf(msg, sizeof msg, "message number %d with some filler words", i);
        ui_log_append(&lg, msg, (size_t)n);
    }
    CHECK(lg.used <= CHATDS_UI_LOG_BYTES && lg.nlines <= CHATDS_UI_LOG_LINES);
    CHECK(lg.has_open && lg.open_line < lg.nlines);
    CHECK((lg.lines[lg.open_line].flags & LINE_FIRST) != 0);
    CHECK(!memcmp(lg.text + lg.lines[lg.nlines - 1].off, "filler words", 12) ||
          lg.lines[lg.nlines - 1].len > 0);
    int ok = 1;
    for (size_t i = 0; i < lg.nlines; i++)
        ok &= lg.lines[i].off + lg.lines[i].len <= lg.used;
    CHECK(ok);
    CHECK(strstr(lg.text + lg.open_off, "number 399") != NULL);

    /* appending to a full log reports only lines added at the bottom, not evictions */
    ui_log_begin(&lg, CHATDS_ROLE_ANSWER);
    int open0 = lg.nlines - lg.open_line;
    char wide[200];
    memset(wide, 'y', sizeof wide);
    int d = ui_log_append(&lg, wide, sizeof wide);
    CHECK(lg.open_line > 0 && d == (int)(lg.nlines - lg.open_line) - open0 && d > 0);

    /* one message bigger than the whole log: truncated, no overflow */
    ui_log_clear(&lg);
    ui_log_begin(&lg, CHATDS_ROLE_ANSWER);
    char big[200];
    memset(big, 'x', sizeof big);
    for (int i = 0; i < 60; i++) ui_log_append(&lg, big, sizeof big);
    CHECK(lg.used == CHATDS_UI_LOG_BYTES && lg.has_open && lg.open_off == 0);
}

/* ---- keyboard ----------------------------------------------------------------------- */
static void test_keyboard(void)
{
    /* every printable ASCII char can be typed, and its touch point hits that key */
    for (int ch = 32; ch < 127; ch++) {
        int layer, shift, x, y, row, idx;
        CHECK(chatds_ui_key_for_char(ch, &layer, &shift) == 0);
        int code = shift ? ch + 32 : ch;
        CHECK(chatds_ui_key_point(layer, code, &x, &y) == 0);
        CHECK(ui_kbd_hit(layer, x, y, &row, &idx) == code);
    }
    int l, s;
    CHECK(chatds_ui_key_for_char(7, &l, &s) < 0 && chatds_ui_key_for_char(200, &l, &s) < 0);

    /* each row tiles exactly 256 px with no gaps or overlaps; keys are finger sized */
    for (int layer = 0; layer < 2; layer++)
        for (int r = 0; r < KBD_ROWS; r++) {
            int edge = 0;
            for (int i = 0; ui_kbd_row(layer, r)[i].w; i++) {
                int x0, x1;
                ui_kbd_rect(layer, r, i, &x0, &x1);
                CHECK(x0 == edge);
                if (ui_kbd_row(layer, r)[i].code) CHECK(x1 - x0 >= 15);
                edge = x1;
            }
            CHECK(edge == CHATDS_UI_W);
        }
    /* everything below the bar and above the keys is not a key */
    int row, idx;
    CHECK(ui_kbd_hit(0, 100, KBD_Y - 1, &row, &idx) == 0);
    CHECK(ui_kbd_hit(0, 100, KBD_Y + KBD_ROWS * KBD_KEY_H, &row, &idx) == 0);
    CHECK(KBD_Y + KBD_ROWS * KBD_KEY_H <= CHATDS_UI_H);
    CHECK(BAR_Y + BAR_H <= KBD_Y);
}

/* ---- typing and sending ------------------------------------------------------------- */
static void test_typing(void)
{
    setup();
    run("type Hi there?");
    CHECK(!strcmp(chatds_ui_input_text(&ui), "Hi there?"));
    run("type 1+2=(3)\ntype ~`|\\{}");
    CHECK(!strcmp(chatds_ui_input_text(&ui), "Hi there?1+2=(3)~`|\\{}"));
    CHECK(ui.layer == CHATDS_LAYER_SYMBOLS);
    run("type A");
    CHECK(ui.layer == CHATDS_LAYER_LETTERS && ui.shift == 0);
    CHECK(ui.input[ui.input_len - 1] == 'A');

    /* backspace key and the input cap */
    int x, y;
    chatds_ui_key_point(CHATDS_LAYER_LETTERS, CHATDS_K_BKSP, &x, &y);
    size_t before = ui.input_len;
    char cmd[40];
    snprintf(cmd, sizeof cmd, "tap %d %d\n", x, y);
    run(cmd);
    CHECK(ui.input_len == before - 1);

    setup();
    char many[200];
    strcpy(many, "type ");
    memset(many + 5, 'a', 170);
    many[175] = 0;
    run(many);
    CHECK(ui.input_len == CHATDS_UI_INPUT_MAX);
}

static void test_release_off_key(void)
{
    setup();
    int x, y;
    chatds_ui_key_point(0, 'q', &x, &y);
    pen(1, x, y);
    pen(1, x, y);
    pen(1, 200, 100); /* slide to another key... */
    pen(0, 0, 0);     /* ...and let go there */
    frames(2);
    CHECK(ui.input_len == 0);
    pen(1, x, y);
    pen(0, 0, 0);
    CHECK(!strcmp(chatds_ui_input_text(&ui), "q"));
    /* a press that starts outside any key never types */
    pen(1, 3, KBD_Y - 2);
    pen(1, x, y);
    pen(0, 0, 0);
    CHECK(ui.input_len == 1);
}

static void test_send_streams_and_finishes(void)
{
    setup();
    run("button send");
    CHECK(!chatds_ui_is_busy(&ui)); /* empty input: nothing to send */
    run("type hello");
    run("button send");
    CHECK(chatds_ui_is_busy(&ui));
    CHECK(ui.input_len == 0);
    CHECK(has_line("hello", CHATDS_ROLE_USER));
    size_t last = 0;
    int grew = 0;
    for (int i = 0; i < 12; i++) {
        frames(1);
        size_t n = answer_len();
        CHECK(n >= last);
        if (n > last) grew++;
        last = n;
    }
    CHECK(chatds_ui_is_busy(&ui) && grew >= 4 && last > 0); /* text shows up before the end */
    run("until idle 400");
    CHECK(!chatds_ui_is_busy(&ui));
    CHECK(has_line("You asked: hello", CHATDS_ROLE_ANSWER));
}

static void test_enter_and_start_key(void)
{
    setup();
    run("type a");
    int x, y;
    chatds_ui_key_point(0, CHATDS_K_ENTER, &x, &y);
    char cmd[40];
    snprintf(cmd, sizeof cmd, "tap %d %d", x, y);
    run(cmd);
    CHECK(chatds_ui_is_busy(&ui));
    run("until idle 400");
    run("type b\nkey start");
    CHECK(chatds_ui_is_busy(&ui));
}

static void test_cancel(void)
{
    setup();
    run("button stop"); /* idle: stop is disabled */
    CHECK(!has_line("cancelled", 0));
    run("type long\nbutton send\nwait 30");
    CHECK(chatds_ui_is_busy(&ui));
    CHECK(answer_len() > 0);
    run("button stop");
    CHECK(!chatds_ui_is_busy(&ui));
    size_t at = answer_len();
    CHECK(has_line("[cancelled]", CHATDS_ROLE_NOTE));
    CHECK(dg.pos == dg.len); /* generator told to stop, never stepped again */
    frames(30);
    CHECK(answer_len() == at);

    run("type again\nbutton send\nwait 10\nkey b");
    CHECK(!chatds_ui_is_busy(&ui)); /* B cancels too */
    run("type once more\nbutton send\nuntil idle 400"); /* usable after a cancel */
    CHECK(has_line("You asked: once more", 0));
}

static void test_errors_and_empty(void)
{
    setup();
    run("type fail now\nbutton send\nuntil idle 100");
    CHECK(!chatds_ui_is_busy(&ui) && has_line("[error: demo failure]", CHATDS_ROLE_NOTE));
    run("type empty\nbutton send\nuntil idle 100");
    CHECK(has_line("(no answer)", CHATDS_ROLE_ANSWER));
}

static int refuse(void *c, const char *q) { (void)c; (void)q; return 1; }

static void test_refused_start(void)
{
    chatds_ui_gen g = {NULL, refuse, NULL, NULL};
    chatds_ui_init(&ui, &g);
    run("type x\nbutton send");
    CHECK(!chatds_ui_is_busy(&ui) && has_line("[could not start]", CHATDS_ROLE_NOTE));
    CHECK(!strcmp(chatds_ui_input_text(&ui), "x")); /* question kept for a retry */
}

/* ---- scrolling ------------------------------------------------------------------------ */
static void test_scroll(void)
{
    setup();
    run("type long\nbutton send\nuntil idle 600");
    CHECK(chatds_ui_line_count(&ui) > TOP_ROWS);
    CHECK(chatds_ui_view_scroll(&ui) == 0);
    run("button up");
    CHECK(chatds_ui_view_scroll(&ui) == 3);
    run("button down");
    CHECK(chatds_ui_view_scroll(&ui) == 0);
    run("button down");
    CHECK(chatds_ui_view_scroll(&ui) == 0); /* clamps at the newest line */
    run("key up\nkey up");
    CHECK(chatds_ui_view_scroll(&ui) == 6);
    run("key down");
    CHECK(chatds_ui_view_scroll(&ui) == 3);
    for (int i = 0; i < 40; i++) run("button up");
    CHECK(chatds_ui_view_scroll(&ui) == chatds_ui_line_count(&ui) - TOP_ROWS); /* clamps at oldest */

    /* holding Up repeats */
    setup();
    run("type long\nbutton send\nuntil idle 600");
    int x, y;
    chatds_ui_button_point(CHATDS_BTN_UP, &x, &y);
    char cmd[40];
    snprintf(cmd, sizeof cmd, "hold %d %d 50", x, y);
    run(cmd);
    CHECK(chatds_ui_view_scroll(&ui) > 3 * 2); /* more than the one step of a tap */

    /* a view scrolled up stays on the same text while more arrives below it */
    setup();
    run("type long\nbutton send\nwait 250\nbutton up");
    CHECK(chatds_ui_line_count(&ui) > TOP_ROWS && chatds_ui_is_busy(&ui));
    long first_before = (long)chatds_ui_line_count(&ui) - TOP_ROWS - (long)chatds_ui_view_scroll(&ui);
    run("wait 40");
    long first_after = (long)chatds_ui_line_count(&ui) - TOP_ROWS - (long)chatds_ui_view_scroll(&ui);
    CHECK(first_before == first_after);
    run("until idle 600");
    CHECK(chatds_ui_view_scroll(&ui) > 0);

    /* ...and while a note is added below it */
    setup();
    run("type long\nbutton send\nwait 250\nbutton up");
    first_before = (long)chatds_ui_line_count(&ui) - TOP_ROWS - (long)chatds_ui_view_scroll(&ui);
    run("button stop");
    CHECK(has_line("[cancelled]", CHATDS_ROLE_NOTE));
    first_after = (long)chatds_ui_line_count(&ui) - TOP_ROWS - (long)chatds_ui_view_scroll(&ui);
    CHECK(first_before == first_after);

    /* sending a new question jumps back to the bottom */
    run("type next\nbutton send");
    CHECK(chatds_ui_view_scroll(&ui) == 0);
}

/* ---- about -------------------------------------------------------------------------- */
static void test_about(void)
{
    setup();
    run("button about");
    CHECK(chatds_ui_is_about(&ui));
    static const char *const must[] = {"zlib", "FatFs", "picolibc", "CC BY-SA 4.0", "MIT",
                                       "GPLv3", "CDLA-Sharing-1.0", "libnds", "TinyStories"};
    for (size_t i = 0; i < sizeof must / sizeof must[0]; i++)
        CHECK(strstr(chatds_ui_credits, must[i]) != NULL);
    int printable = 1;
    for (const char *p = chatds_ui_credits; *p; p++) printable &= *p == '\n' || (*p >= 32 && *p < 127);
    CHECK(printable); /* the font can draw every credits character */
    CHECK(chatds_ui_credits_lines() > TOP_ROWS);

    uint16_t before[CHATDS_UI_W * CHATDS_UI_H];
    chatds_ui_render_all(&ui, top, bot);
    memcpy(before, top, sizeof before);
    run("button down");
    CHECK(ui.about_scroll == 3);
    chatds_ui_render_all(&ui, top, bot);
    CHECK(memcmp(before, top, sizeof before) != 0);
    run("button up");
    CHECK(ui.about_scroll == 0);

    /* the keyboard is hidden: taps on it do nothing */
    run("tap 40 100\ntap 128 170");
    CHECK(ui.input_len == 0);
    run("button about");
    CHECK(!chatds_ui_is_about(&ui));
    CHECK(chatds_ui_line_count(&ui) >= 1);

    /* generation continues while the credits are open */
    run("type hi\nbutton send\nbutton about\nwait 100\nbutton about\nuntil idle 400");
    CHECK(has_line("You asked: hi", CHATDS_ROLE_ANSWER));

    /* START does not send while the credits are open, like the disabled Send button */
    run("type again\nbutton about\nkey start");
    CHECK(!chatds_ui_is_busy(&ui) && ui.input_len == 5);
    run("button about\nkey start");
    CHECK(chatds_ui_is_busy(&ui));
}

/* ---- rendering ------------------------------------------------------------------------ */
static void test_render(void)
{
    setup();
    static uint16_t guarded[(CHATDS_UI_H + 2) * CHATDS_UI_W];
    for (size_t i = 0; i < sizeof guarded / sizeof guarded[0]; i++) guarded[i] = 0xBEEF;
    uint16_t *fb = guarded + CHATDS_UI_W;
    chatds_ui_render_all(&ui, fb, NULL);
    int guards = 1, alpha = 1;
    for (int i = 0; i < CHATDS_UI_W; i++)
        guards &= guarded[i] == 0xBEEF && guarded[(CHATDS_UI_H + 1) * CHATDS_UI_W + i] == 0xBEEF;
    for (int i = 0; i < CHATDS_UI_W * CHATDS_UI_H; i++) alpha &= (fb[i] & 0x8000) != 0;
    CHECK(guards && alpha);
    chatds_ui_render_all(&ui, NULL, fb);
    guards = alpha = 1;
    for (int i = 0; i < CHATDS_UI_W; i++)
        guards &= guarded[i] == 0xBEEF && guarded[(CHATDS_UI_H + 1) * CHATDS_UI_W + i] == 0xBEEF;
    for (int i = 0; i < CHATDS_UI_W * CHATDS_UI_H; i++) alpha &= (fb[i] & 0x8000) != 0;
    CHECK(guards && alpha);

    /* change tracking: nothing to redraw until something changes */
    chatds_ui_render_all(&ui, top, bot);
    CHECK(chatds_ui_render(&ui, top, bot) == 0);
    frames(5);
    CHECK(chatds_ui_render(&ui, top, bot) == 0);
    run("type a");
    chatds_ui_render(&ui, top, bot);
    CHECK(chatds_ui_render(&ui, top, bot) == 0);

    /* rendering is a pure function of state */
    static uint16_t a[CHATDS_UI_W * CHATDS_UI_H], b[CHATDS_UI_W * CHATDS_UI_H];
    chatds_ui_render_all(&ui, a, b);
    chatds_ui_render_all(&ui, top, bot);
    CHECK(!memcmp(a, top, sizeof a) && !memcmp(b, bot, sizeof b));

    /* the busy header animates, and the idle one does not */
    run("button send");
    CHECK(chatds_ui_is_busy(&ui));
    uint16_t h0[CHATDS_UI_W * HEADER_H], h1[CHATDS_UI_W * HEADER_H];
    memcpy(h0, top, sizeof h0);
    frames(6);
    memcpy(h1, top, sizeof h1);
    CHECK(memcmp(h0, h1, sizeof h0) != 0);
    int has_busy = 0;
    for (int i = 0; i < CHATDS_UI_W * HEADER_H; i++) has_busy |= h1[i] == COL_BUSY;
    CHECK(has_busy);

    /* a pressed key lights up, and releasing restores it */
    setup();
    int x, y;
    chatds_ui_key_point(0, 'g', &x, &y);
    chatds_ui_render_all(&ui, top, bot);
    uint16_t idle_px = bot[y * CHATDS_UI_W + x + 5];
    pen(1, x, y);
    CHECK(bot[y * CHATDS_UI_W + x + 5] == COL_KEY_DOWN && idle_px != COL_KEY_DOWN);
    pen(0, 0, 0);
    CHECK(bot[y * CHATDS_UI_W + x + 5] == idle_px);

    /* Send is only green with something to send and nothing running */
    setup();
    chatds_ui_render_all(&ui, top, bot);
    int sx, sy;
    chatds_ui_button_point(CHATDS_BTN_SEND, &sx, &sy);
    CHECK(bot[(sy - 8) * CHATDS_UI_W + sx - 20] == COL_OFF);
    run("type z");
    CHECK(bot[(sy - 8) * CHATDS_UI_W + sx - 20] == COL_OK);
}

static uint32_t fake_millis(void *ctx) { return *(uint32_t *)ctx; }

static void check_busy_header(const char *status)
{
    uint16_t expected[CHATDS_UI_W * CHATDS_UI_H];
    ui_fill(expected, 0, 0, CHATDS_UI_W, HEADER_H, COL_PANEL);
    ui_fill(expected, 0, HEADER_H - 1, CHATDS_UI_W, 1, COL_EDGE);
    ui_text(expected, 4, 2, "ChatDS", 6, COL_TEXT);
    size_t n = strlen(status);
    ui_text(expected, CHATDS_UI_W - 4 - (int)n * GLYPH_W, 2, status, n, COL_BUSY);
    CHECK(!memcmp(expected, top, CHATDS_UI_W * HEADER_H * sizeof *top));
}

static void test_busy_clock(void)
{
    setup();
    run("type a");
    uint32_t now = UINT32_MAX - 50;
    chatds_ui_set_clock(&ui, fake_millis, &now);
    chatds_ui_input in = {0, 0, 0, CHATDS_KEY_START};
    chatds_ui_frame(&ui, &in);
    CHECK(chatds_ui_is_busy(&ui));
    chatds_ui_render(&ui, top, bot);
    uint32_t frame = ui.frame;
    const char *word = ui.got_text ? "answering" : "thinking";
    char status[24];
    snprintf(status, sizeof status, "| %s 0s", word);
    check_busy_header(status);
    now += 99; /* includes uint32 clock wrap, with no input/generator frames */
    CHECK(chatds_ui_render(&ui, top, bot) == 0);
    now++;
    CHECK(chatds_ui_render(&ui, top, bot) == 1);
    snprintf(status, sizeof status, "/ %s 0s", word);
    check_busy_header(status);
    now += 1050; /* skip many animation phases during a blocking engine step */
    CHECK(chatds_ui_render(&ui, NULL, bot) == 0);
    CHECK(chatds_ui_render(&ui, top, bot) == 1);
    snprintf(status, sizeof status, "\\ %s 1s", word);
    check_busy_header(status);
    now += 850;
    CHECK(chatds_ui_render(&ui, top, bot) == 1);
    snprintf(status, sizeof status, "| %s 2s", word);
    check_busy_header(status);
    CHECK(ui.frame == frame);
    CHECK(chatds_ui_render(&ui, top, bot) == 0);
}

/* ---- engine adapter against the stub engine ------------------------------------------- */
/* Byte tokenizer: id = byte + 3, so 1 and 2 stay BOS and EOS. */
static int tk_encode(void *c, const char *t, size_t n, int bos, int *ids, int max)
{
    int k = 0;
    (void)c;
    if (bos && k < max) ids[k] = CDS_BOS;
    k += bos != 0;
    for (size_t i = 0; i < n; i++, k++)
        if (k < max) ids[k] = (unsigned char)t[i] + 3;
    return k;
}

static const char *tk_piece(void *c, int id, int *len)
{
    static char b;
    (void)c;
    if (id < 3 || id > 258) return NULL;
    b = (char)(id - 3);
    *len = 1;
    return &b;
}

static const cds_tokenizer tk = {tk_encode, tk_piece, NULL};
static uint32_t ans[64];
static int ids[96];

static void engine_setup(chatds_ui_engine_gen *g, const char *answer, uint32_t ctxlen, uint32_t max_new)
{
    size_t n = strlen(answer);
    for (size_t i = 0; i < n; i++) ans[i] = (unsigned char)answer[i] + 3;
    memset(&g_stub, 0, sizeof g_stub);
    g_stub.answer = ans;
    g_stub.answer_len = (uint32_t)n;
    g_stub.eos = CDS_EOS;
    g_stub.context_length = ctxlen;
    chatds_session *s;
    chatds_session_create(NULL, NULL, &s);
    chatds_ui_engine_gen_init(g, s, &tk, NULL, 1, ids, 96, max_new);
    chatds_ui_init(&ui, &g->gen);
}

static void test_engine_adapter(void)
{
    static chatds_ui_engine_gen g;
    engine_setup(&g, "OK!", 256, 50);
    run("type ab\nbutton send\nuntil idle 200");
    CHECK(has_line("OK!", CHATDS_ROLE_ANSWER) && answer_len() == 3);
    /* prompt = BOS + "Q: ab\nA:" = 9 tokens: 8 without logits, 1 with; then 3 decode steps */
    CHECK(g_stub.prefill_forwards == 8 && g_stub.classified == 4 && g_stub.forwards == 12);
    CHECK(g_stub.resets == 1);
    CHECK(!strcmp(g.answer.prompt, "Q: ab\nA:") && g.answer.gen.stop == CDS_STOP_EOS);

    /* cancel stops calling the engine; the next question starts clean */
    engine_setup(&g, "abcdefghijklmnopqrstuvwxyz", 256, 50);
    run("type ab\nbutton send\nwait 14");
    uint32_t seen = g_stub.forwards;
    CHECK(chatds_ui_is_busy(&ui) && seen > 9);
    run("button stop");
    seen = g_stub.forwards;
    frames(20);
    CHECK(g_stub.forwards == seen);
    run("type cd\nbutton send\nuntil idle 200");
    CHECK(g_stub.resets == 2 && g.answer.gen.generated == 26 && g.answer.gen.stop == CDS_STOP_EOS);

    /* answer budget in tokens */
    engine_setup(&g, "abcdefghijklmnop", 256, 5);
    run("type x\nbutton send\nuntil idle 200");
    CHECK(answer_len() == 5 && g.answer.gen.stop == CDS_STOP_BUDGET);

    /* BOS from the model ends the answer, like EOS */
    engine_setup(&g, "hix", 256, 50);
    ans[1] = CDS_BOS;
    run("type x\nbutton send\nuntil idle 200");
    CHECK(answer_len() == 1 && g.answer.gen.stop == CDS_STOP_BOS);

    /* calc() answers get their result, as in cds_ask */
    engine_setup(&g, "calc(6*7)", 256, 50);
    run("type x\nbutton send\nuntil idle 200");
    CHECK(has_line("calc(6*7) = 42", CHATDS_ROLE_ANSWER));

    /* full window mid-answer: keep what was produced, then stop quietly */
    engine_setup(&g, "abcdefghijklmnop", 12, 50);
    run("type ab\nbutton send\nuntil idle 200");
    CHECK(answer_len() == 4 && !has_line("error", 0) && g.answer.gen.stop == CDS_STOP_CONTEXT);

    /* a tokenizer that cannot show a model id is an error, not a hang */
    engine_setup(&g, "ab", 256, 50);
    ans[1] = 400;
    run("type x\nbutton send\nuntil idle 200");
    CHECK(!chatds_ui_is_busy(&ui) && has_line("[error: bad format]", CHATDS_ROLE_NOTE));

    /* a prompt longer than the window or the id buffer refuses the question */
    engine_setup(&g, "abc", 5, 50);
    run("type ab\nbutton send\nuntil idle 200");
    CHECK(!chatds_ui_is_busy(&ui) && has_line("[could not start]", 0) && g_stub.forwards == 0);
    engine_setup(&g, "abc", 256, 50);
    g.opts.ids_cap = 2;
    run("type hello there\nbutton send");
    CHECK(!chatds_ui_is_busy(&ui) && has_line("[could not start]", 0));
}

static void test_script_errors(void)
{
    setup();
    CHECK(chatds_script_run(&ui, "wait 1\nbogus\n", &host) == 2);
    CHECK(chatds_script_run(&ui, "button nope", &host) == 1);
    CHECK(chatds_script_run(&ui, "type \x01", &host) == 1);
    CHECK(chatds_script_run(&ui, "# only a comment\n\n", &host) == 0);
}

int main(void)
{
    test_wrap();
    test_log();
    test_keyboard();
    test_typing();
    test_release_off_key();
    test_send_streams_and_finishes();
    test_enter_and_start_key();
    test_cancel();
    test_errors_and_empty();
    test_refused_start();
    test_scroll();
    test_about();
    test_render();
    test_busy_clock();
    test_engine_adapter();
    test_script_errors();
    printf("%d checks, %d failed\n", checks, fails);
    return fails != 0;
}
