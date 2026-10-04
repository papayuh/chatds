/* SPDX-License-Identifier: MIT
 * Product shell: every application translation unit is clean-room ChatDS code.
 * User assets and results live under fat:/chatds/.
 */
#ifndef ARM9
#define _POSIX_C_SOURCE 200809L
#include <time.h>
#else
#include <nds.h>
#include <fat.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tok.h"
#include "chatds_ui_engine.h"
#include "chatds_ui_demo.h" /* Scripted input only; the demo generator is not linked. */

static chatds_model *model;
static chatds_session *session;
static tok_t tokenizer;
static cds_kb kb;
static cds_config config;
static cds_tokenizer tok;
static chatds_ui_engine_gen generator;
static chatds_ui_gen wrapped;
static const char *directory;
static int *prompt_ids;
static uint32_t output_ids[256];
static char answer_text[4096];
static uint32_t previous;
static uint64_t elapsed, first, first_piece;
static int published, publication_failed;
static char current_question[CDS_QUESTION_MAX + 1];
static void (*stream_piece)(void *, uint32_t, const char *, int);

/* Sample at each forward/frame, extending the 32-bit DS bus clock. No run is
 * timed using a single wrapping delta (full-context runs can exceed 128 s). */
static uint32_t ticks(void) {
#ifdef ARM9
    return cpuGetTiming();
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint32_t)((uint64_t)t.tv_sec * 33513982 + (uint64_t)t.tv_nsec * 33513982 / 1000000000);
#endif
}
static void sample(void) {
    uint32_t now = ticks();
    elapsed += (uint32_t)(now - previous);
    previous = now;
}
/* The product owns timers 0/1 already; do not start the demo UI timer. */
uint32_t chatds_ui_millis(void *ctx) {
    static uint32_t last;
    static uint64_t total;
    (void)ctx;
    uint32_t now = ticks();
    total += (uint32_t)(now - last);
    last = now;
    return (uint32_t)(total * 1000 / 33513982);
}
static int encode(void *ctx, const char *text, size_t len, int bos, int *ids, int max) {
    return tok_encode(ctx, text, len, bos, ids, max);
}
static const char *piece(void *ctx, int id, int *len) { return tok_piece(ctx, id, len); }

static void observed_piece(void *ctx, uint32_t id, const char *bytes, int len) {
    sample();
    if (!first_piece) first_piece = elapsed;
    stream_piece(ctx, id, bytes, len);
}
static void publish(int ok) {
    char extra[512];
    sample();
    snprintf(extra, sizeof extra, "first_logit_ticks=%llu\nfirst_piece_ticks=%llu\ntotal_ticks=%llu\nkv_bits=%d\n",
             (unsigned long long)first, (unsigned long long)first_piece,
             (unsigned long long)elapsed, 16);
#ifdef ARM9
    size_t used = strlen(extra);
    snprintf(extra + used, sizeof extra - used,
             "heap_start=%p\nheap_end=%p\nheap_limit=%p\nram_break_bytes=%lu\n",
             getHeapStart(), getHeapEnd(), getHeapLimit(),
             (unsigned long)((uintptr_t)getHeapEnd() - 0x02000000u));
#endif
    static char story[sizeof answer_text + CDS_QUESTION_MAX + 1];
    const char *text = answer_text;
    if (!config.instruct) {
        snprintf(story, sizeof story, "%s%s", current_question, answer_text);
        text = story;
    }
    cds_result result = {&generator.answer, text, output_ids,
        (uint32_t)(elapsed * 1000 / 33513982), ok, config.instruct, session,
        chatds_model_memory_bytes(model), extra, kb.f ? &kb : NULL};
    publication_failed = cds_publish(directory, &result) != 0;
    if (publication_failed) fprintf(stderr, "result publication failed\n");
    published = 1;
}
static int begin(void *ctx, const char *question) {
    (void)ctx;
    elapsed = first = first_piece = 0;
    previous = ticks();
    published = publication_failed = 0;
    if (strlen(question) > CDS_QUESTION_MAX) return 1;
    strcpy(current_question, question);
    answer_text[0] = 0;
    int bad = generator.gen.begin(generator.gen.ctx, question);
    sample();
    if (bad) publish(0);
    return bad;
}
static chatds_gen_event step(void *ctx, char *out, size_t cap, size_t *len) {
    (void)ctx;
    chatds_gen_event event = generator.gen.step(generator.gen.ctx, out, cap, len);
    sample();
    if (!first && generator.run.fed == generator.answer.gen.prompt_tokens)
        first = elapsed;
    if (!published && (event == CHATDS_GEN_DONE || event == CHATDS_GEN_ERROR))
        publish(event == CHATDS_GEN_DONE);
    if (publication_failed) {
        const char message[] = "card write failed";
        *len = sizeof message - 1 < cap ? sizeof message - 1 : cap;
        memcpy(out, message, *len);
        return CHATDS_GEN_ERROR;
    }
    return event;
}
static void cancel(void *ctx) {
    (void)ctx;
    generator.gen.cancel(generator.gen.ctx);
    /* The UI stops stepping after cancel; finish the cancellation in plumbing. */
    cds_gen_step(&generator.run);
    if (!published) publish(generator.run.st == CHATDS_OK);
}
static int batch(void) {
    if (begin(NULL, config.prompt)) return 1;
    char text[CHATDS_UI_STEP_TEXT];
    size_t len;
    chatds_gen_event event;
    do { event = step(NULL, text, sizeof text, &len); }
    while (event != CHATDS_GEN_DONE && event != CHATDS_GEN_ERROR);
    return event == CHATDS_GEN_ERROR;
}

#ifdef ARM9
static chatds_ui ui;
static uint16_t *screen_top, *screen_bottom;
static FILE *frame_index;
static unsigned frame_count, script_frames;
static int capture_error;

static void snapshot(const char *name) {
    char path[96];
    snprintf(path, sizeof path, "fat:/chatds/%s.raw", name);
    FILE *f = fopen(path, "wb");
    if (!f) { capture_error = 1; return; }
    size_t pixels = CHATDS_UI_W * CHATDS_UI_H;
    int bad = fwrite(screen_top, 2, pixels, f) != pixels;
    bad |= fwrite(screen_bottom, 2, pixels, f) != pixels;
    bad |= fclose(f) != 0;
    capture_error |= bad;
}
static void script_tick(void *ctx) {
    (void)ctx;
    /* The UI demo driver normally installs a frame clock; the product keeps
     * real bus-clock milliseconds even while its input is scripted. */
    if (!script_frames) chatds_ui_set_clock(&ui, chatds_ui_millis, NULL);
    if (script_frames++ % 12 == 0 && frame_count < 64) {
        char name[16];
        snprintf(name, sizeof name, "f%03u", frame_count++);
        snapshot(name);
        fprintf(frame_index, "%s.raw %lu\n", name, (unsigned long)chatds_ui_millis(NULL));
    }
    swiWaitForVBlank();
}
static void state_sink(void *ctx, const char *line) { fprintf(ctx, "%s\n", line); }
static void script_dump(void *ctx, const char *name, chatds_ui *state) {
    (void)ctx;
    /* Script names are constrained by ui_script.c's parser. */
    if (strchr(name, '/') || strstr(name, "..")) { capture_error = 1; return; }
    snapshot(name);
    fprintf(frame_index, "%s.raw %lu\n", name, (unsigned long)chatds_ui_millis(NULL));
    char path[96];
    snprintf(path, sizeof path, "fat:/chatds/%s.txt", name);
    FILE *f = fopen(path, "w");
    if (!f) { capture_error = 1; return; }
    chatds_script_state_text(state, state_sink, f);
    capture_error |= ferror(f);
    capture_error |= fclose(f) != 0;
}
static int run_script(void) {
    FILE *f = fopen("fat:/chatds/ui-script.txt", "rb");
    if (!f) return 0;
    static char script[8192];
    size_t n = fread(script, 1, sizeof script - 1, f);
    script[n] = 0;
    int bad = ferror(f) || fgetc(f) != EOF;
    fclose(f);
    frame_index = fopen("fat:/chatds/frames.txt", "w");
    if (!frame_index) bad = 1;
    if (!bad) {
        chatds_script_host host = {NULL, script_tick, script_dump, screen_top, screen_bottom};
        bad = chatds_script_run(&ui, script, &host);
    }
    if (frame_index) {
        bad |= ferror(frame_index);
        bad |= fclose(frame_index) != 0;
    }
    f = fopen("fat:/chatds/ui-result.txt", "w");
    if (f) {
        fprintf(f, "frames=%u\nstatus=%s\n", frame_count,
                bad || capture_error || publication_failed ? "FAIL" : "OK");
        fclose(f);
    }
    return 1;
}
static void interactive(void) {
    videoSetMode(MODE_5_2D);
    videoSetModeSub(MODE_5_2D);
    vramSetBankA(VRAM_A_MAIN_BG);
    vramSetBankC(VRAM_C_SUB_BG);
    uint16_t *top = bgGetGfxPtr(bgInit(3, BgType_Bmp16, BgSize_B16_256x256, 0, 0));
    uint16_t *bottom = bgGetGfxPtr(bgInitSub(3, BgType_Bmp16, BgSize_B16_256x256, 0, 0));
    screen_top = top; screen_bottom = bottom;
    lcdMainOnTop();
    chatds_ui_init(&ui, &wrapped);
    /* Prefill is shell configuration, bounded by cds_config_read. */
    strcpy(ui.input, config.prompt);
    ui.input_len = (uint16_t)strlen(config.prompt);
    chatds_ui_render_all(&ui, top, bottom);
    if (run_script()) for (;;) swiWaitForVBlank();
    for (;;) {
        swiWaitForVBlank();
        scanKeys();
        uint32_t held = keysHeld();
        chatds_ui_input in = {0};
        if (held & KEY_UP) in.keys |= CHATDS_KEY_UP;
        if (held & KEY_DOWN) in.keys |= CHATDS_KEY_DOWN;
        if (held & KEY_B) in.keys |= CHATDS_KEY_B;
        if (held & KEY_START) in.keys |= CHATDS_KEY_START;
        if (held & KEY_SELECT) in.keys |= CHATDS_KEY_SELECT;
        if (held & KEY_TOUCH) {
            touchPosition t; touchRead(&t);
            in.touch = 1; in.x = (uint8_t)t.px; in.y = (uint8_t)t.py;
        }
        chatds_ui_frame(&ui, &in);
        chatds_ui_render(&ui, top, bottom);
    }
}
#endif

int main(int argc, char **argv) {
    const char *model_path, *tok_path, *kb_path, *run_path;
    int rc = 1;
#ifdef ARM9
    (void)argc; (void)argv;
    consoleDemoInit();
    cpuStartTiming(0);
    if (!fatInitDefault()) { puts("FAIL: fatInitDefault()"); goto idle; }
    model_path = "fat:/chatds/model.bin"; tok_path = "fat:/chatds/tok.bin";
    kb_path = "fat:/chatds/kb.bin"; run_path = "fat:/chatds/run.txt";
    directory = "fat:/chatds/";
#else
    if (argc != 6) { fprintf(stderr, "usage: chatds-host MODEL TOK KB-or-- RUN OUTDIR/\n"); return 2; }
    model_path = argv[1]; tok_path = argv[2]; kb_path = argv[3]; run_path = argv[4]; directory = argv[5];
    size_t dir_len = strlen(directory);
    if (!dir_len || directory[dir_len - 1] != '/') {
        fprintf(stderr, "OUTDIR must end with '/'\n"); return 2;
    }
#endif
    previous = ticks();
    if (cds_config_read(run_path, &config) || config.steps > 256 || !config.steps) {
        fprintf(stderr, "invalid run.txt\n"); goto fail;
    }
    chatds_status status = chatds_model_load(model_path, &model);
    if (status != CHATDS_OK) { fprintf(stderr, "%s\n", chatds_status_string(status)); goto fail; }
    if (tok_load(&tokenizer, tok_path)) { fprintf(stderr, "invalid tokenizer\n"); goto fail; }
    const chatds_model_config *shape = chatds_model_get_config(model);
    if ((uint32_t)tokenizer.n != shape->vocab_size) goto fail;
    prompt_ids = malloc(shape->context_length * sizeof(*prompt_ids));
    if (!prompt_ids) goto fail;
    status = chatds_session_create(model, NULL, &session);
    if (status != CHATDS_OK) goto fail;
    cds_kb_status ks = cds_kb_open(&kb, kb_path);
    if (ks == CDS_KB_BAD) goto fail; /* optional missing KB is allowed */
    tok = (cds_tokenizer){encode, piece, &tokenizer};
    chatds_ui_engine_gen_init(&generator, session, &tok, kb.f ? &kb : NULL,
                             config.instruct, prompt_ids, shape->context_length, config.steps);
    stream_piece = generator.opts.on_piece;
    generator.opts.on_piece = observed_piece;
    generator.opts.out_ids = output_ids;
    generator.opts.text = answer_text;
    generator.opts.text_cap = sizeof answer_text;
    wrapped = (chatds_ui_gen){NULL, begin, step, cancel};
#ifdef ARM9
    if (config.kbd) interactive();
#endif
    rc = batch();
    goto finish;
fail:
    publish(0);
finish:
    cds_kb_close(&kb);
    free(prompt_ids);
    chatds_session_destroy(session);
    chatds_model_destroy(model);
    tok_free(&tokenizer);
#ifdef ARM9
idle:
    for (;;) swiWaitForVBlank();
#endif
    return rc;
}
