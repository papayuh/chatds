/* SPDX-License-Identifier: MIT
 * Product shell: every application translation unit is clean-room ChatDS code.
 * User assets and results live at /chatds/ on the launch-selected card.
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
#include "dispatch.h"
#include "learned.h"
#include "chatds_storage.h"
#ifdef ARM9
#include "internal.h" /* Private boot kernel smoke check. */
#endif

static chatds_model *model;
static chatds_session *session;
static tok_t tokenizer;
/* KB3 compatibility is never live with KB4 retrieval/resolution. Borrow
 * that typed union member for the entire legacy-reader lifetime. */
static cds_kb *kb;
static cds_config config;
static cds_tokenizer tok;
static chatds_ui_engine_gen generator;
static chatds_ui_gen wrapped;
static c14_runtime runtime;
static const char *directory;
static int *prompt_ids;
static uint32_t output_ids[256];
static char answer_text[4096];
static uint32_t previous;
static uint64_t elapsed, first, first_piece;
static int published, publication_failed;
static char current_question[CDS_QUESTION_MAX + 1];
static void (*stream_piece)(void *, uint32_t, const char *, int);
#ifdef ARM9
static chatds_ui ui;
static uint16_t *screen_top, *screen_bottom;
extern unsigned char __sbss_end[], __sp_usr[];
#define STACK_PAINT 0xc14c14aau
static uintptr_t stack_base, stack_top, initial_sp;
/* Current BlocksDS CRT reserves 48 KiB above getHeapLimit() for main's
 * coroutine stack. Older CRTs run main in DTCM. Verify actual SP before
 * painting either region; never write active frames or the heap. */
static __attribute__((noinline)) void stack_paint(void) {
    uintptr_t sp;
    __asm__ volatile("mov %0, sp" : "=r"(sp));
    initial_sp = sp;
    if (sp > (uintptr_t)__sbss_end && sp <= (uintptr_t)__sp_usr) {
        stack_base = (uintptr_t)__sbss_end; stack_top = (uintptr_t)__sp_usr;
    } else {
        uintptr_t base = (uintptr_t)getHeapLimit();
        if (sp <= base || sp > base + 48 * 1024) return;
        stack_base = base; stack_top = base + 48 * 1024;
    }
    volatile uint32_t *p = (volatile uint32_t *)stack_base;
    while ((uintptr_t)p + sizeof(*p) <= sp - 64) *p++ = STACK_PAINT;
}
static size_t stack_free(void) {
    volatile uint32_t *p = (volatile uint32_t *)stack_base;
    if (!p) return 0;
    while ((uintptr_t)(p + 1) <= stack_top && *p == STACK_PAINT) p++;
    return (uintptr_t)p - stack_base;
}
#endif

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
    if (!config.instruct) stream_piece(ctx, id, bytes, len);
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
             "heap_start=%p\nheap_end=%p\nheap_limit=%p\nram_break_bytes=%lu\nuser_stack_base=%p\nuser_stack_initial_sp=%p\nuser_stack_available_bytes=%lu\nuser_stack_free_bytes=%lu\n",
             getHeapStart(), getHeapEnd(), getHeapLimit(),
             (unsigned long)((uintptr_t)getHeapEnd() - 0x02000000u),
             (void *)stack_base, (void *)initial_sp,
             (unsigned long)(stack_top - stack_base), (unsigned long)stack_free());
#endif
    cds_result result = {&generator.answer, answer_text, output_ids,
        (uint32_t)(elapsed * 1000 / 33513982), ok, config.instruct, session,
        chatds_model_memory_bytes(model), extra, kb && kb->f ? kb : NULL,
        config.instruct ? NULL : current_question};
    publication_failed = config.instruct && runtime.g && c14_log(&runtime, directory) != 0;
    if (publication_failed) result.ok = 0;
    publication_failed |= cds_publish(directory, &result) != 0;
    if (publication_failed) {
        cds_dlg_reset(&runtime.dlg);
        fprintf(stderr, "result publication failed in %s; check card, directory and write access\n", directory);
    }
    published = 1;
}
static int begin(void *ctx, const char *question) {
    (void)ctx;
    elapsed = first = first_piece = 0;
    previous = ticks();
    published = publication_failed = 0;
    if (strlen(question) > CDS_QUESTION_MAX) { cds_dlg_reset(&runtime.dlg); return 1; }
    strcpy(current_question, question);
    answer_text[0] = 0;
    int bad = config.instruct ? c14_begin(&runtime, question) : generator.gen.begin(generator.gen.ctx, question);
#ifdef ARM9
    if (!bad && runtime.mathnb_active && screen_top)
        mnb_render(&runtime.mathnb, &runtime.mathnb_values, screen_top);
#endif
    sample();
    if (bad) publish(0);
    return bad;
}
static chatds_gen_event step(void *ctx, char *out, size_t cap, size_t *len) {
    (void)ctx;
    chatds_gen_event event = config.instruct ? c14_step(&runtime, out, cap, len) : generator.gen.step(generator.gen.ctx, out, cap, len);
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
    if (config.instruct) c14_cancel(&runtime);
    else {
        generator.gen.cancel(generator.gen.ctx);
        /* UI stops stepping after cancel; finish cancellation in plumbing. */
        cds_gen_step(&generator.run);
    }
    if (!published) publish(generator.run.st == CHATDS_OK);
}
static int batch(unsigned cancel_after) {
    if (begin(NULL, config.prompt)) return 1;
    char text[CHATDS_UI_STEP_TEXT];
    size_t len;
    unsigned count = 0;
    chatds_gen_event event;
    do {
        event = step(NULL, text, sizeof text, &len);
        if (event != CHATDS_GEN_DONE && event != CHATDS_GEN_ERROR && ++count == cancel_after) cancel(NULL);
    } while (event != CHATDS_GEN_DONE && event != CHATDS_GEN_ERROR);
    return event == CHATDS_GEN_ERROR;
}

/* Shared host/ROM session test protocol. R resets; S injects permitted bounded
 * state; Q asks; C cancels after N incremental steps. This file is absent from normal kits. Never reads labels. */
static void escaped(FILE *f, const char *s) {
    for (; *s; s++) {
        if (*s == '\\') fputs("\\\\", f);
        else if (*s == '\n') fputs("\\n", f);
        else if (*s == '\t') fputs("\\t", f);
        else fputc(*s, f);
    }
}
static void unescape(char *s) {
    char *out = s;
    for (; *s; s++) {
        if (*s == '\\' && s[1]) { s++; *out++ = *s == 'n' ? '\n' : *s == 't' ? '\t' : *s; }
        else *out++ = *s;
    }
    *out = 0;
}
static int session_input(FILE *in, FILE *out) {
    /* Maximum escaped S row is < 900 bytes; Q is < 260. */
    char line[1024];
    int bad = 0;
    while (fgets(line, sizeof line, in)) {
        line[strcspn(line, "\r\n")] = 0;
        char *fields[12]; int nf = 0;
        fields[nf++] = line;
        for (char *p = line; *p; p++) if (*p == '\t') { *p = 0; if (nf == 12) { bad = 1; break; } fields[nf++] = p + 1; }
        for (int i = 0; i < nf; i++) unescape(fields[i]);
        if (!strcmp(fields[0], "R")) { cds_dlg_reset(&runtime.dlg); continue; }
        if (!strcmp(fields[0], "S") && nf == 9) {
            cds_dlg *d = &runtime.dlg; cds_dlg_reset(d);
            d->task = (uint8_t)atoi(fields[1]);
            snprintf(d->entity, sizeof d->entity, "%.*s", CDS_DLG_ENT, fields[2]);
            snprintf(d->answer, sizeof d->answer, "%.*s", CDS_DLG_ANS, fields[3]);
            snprintf(d->lastq, sizeof d->lastq, "%.*s", CDS_DLG_Q, fields[4]);
            d->pend = atoi(fields[5]) != 0;
            snprintf(d->term, sizeof d->term, "%.*s", CDS_DLG_TERM, fields[6]);
            d->cand[0] = (uint32_t)strtoul(fields[7], NULL, 10); d->cand[1] = (uint32_t)strtoul(fields[8], NULL, 10);
            d->ncand = d->cand[1] ? 2 : d->cand[0] ? 1 : 0; continue;
        }
        unsigned cancel_after = 0;
        const char *question = nf == 2 ? fields[1] : "";
        if (!strcmp(fields[0], "C") && nf == 3) {
            char *end; unsigned long n = strtoul(fields[1], &end, 10);
            if (!*fields[1] || *end || !n || n > 1024) { bad = 1; break; }
            cancel_after = (unsigned)n; question = fields[2];
        } else if (strcmp(fields[0], "Q") || nf != 2) { bad = 1; break; }
        if (strlen(question) > CDS_QUESTION_MAX) { bad = 1; break; }
        strcpy(config.prompt, question);
        int failed = batch(cancel_after); bad |= failed;
        fprintf(out, "%s\t", failed ? "FAIL" : "OK"); escaped(out, runtime.path); fputc('\t', out);
        escaped(out, answer_text); fputc('\t', out); escaped(out, generator.answer.ctx); fputc('\t', out);
        escaped(out, generator.answer.calc);
        fprintf(out, "\t%s\t%lu\t%lu\t", cds_stop_string(generator.answer.gen.stop),
            (unsigned long)generator.answer.gen.prompt_tokens, (unsigned long)generator.answer.gen.generated);
        for (uint32_t i = 0; i < generator.answer.gen.generated; i++) fprintf(out, "%s%lu", i ? "," : "", (unsigned long)output_ids[i]);
        fprintf(out, "\t%llu\n", (unsigned long long)(elapsed * 1000 / 33513982)); fflush(out);
        if (failed) break;
    }
    bad |= ferror(in) || ferror(out);
    return bad;
}
#ifdef ARM9
/* All optional instrumentation uses the same immutable root as model/config. */
static FILE *card_open(const char *name, const char *mode) {
    char path[256];
    if (cds_path_join(path, sizeof path, directory, name)) {
        cds_storage_error("card path failed in directory", directory); return NULL;
    }
    FILE *f = fopen(path, mode);
    if (!f && (mode[0] != 'r' || errno != ENOENT)) cds_storage_error("card open failed", path);
    return f;
}
static int card_close(FILE *f, const char *name) {
    int error = ferror(f) ? (errno ? errno : EIO) : 0;
    if (fclose(f) && !error) error = errno ? errno : EIO;
    if (!error) return 0;
    char path[256];
    if (cds_path_join(path, sizeof path, directory, name))
        return cds_storage_error("card close path failed", directory);
    errno = error;
    return cds_storage_error("card I/O failed", path);
}
static int card_session(void) {
    FILE *in = card_open("session.txt", "rb"); if (!in) return 0;
    FILE *out = card_open("session-results.tsv", "w");
    int bad = !out;
    if (out) {
        bad = session_input(in, out);
        fprintf(out, "status=%s\n", bad ? "FAIL" : "OK");
        card_close(out, "session-results.tsv");
    }
    card_close(in, "session.txt"); return 1;
}
static FILE *frame_index;
static unsigned frame_count, script_frames;
static int capture_error;

static void snapshot(const char *name) {
    char file[96];
    if (cds_path_join(file, sizeof file, name, ".raw")) { capture_error = 1; return; }
    FILE *f = card_open(file, "wb");
    if (!f) { capture_error = 1; return; }
    size_t pixels = CHATDS_UI_W * CHATDS_UI_H;
    int bad = fwrite(screen_top, 2, pixels, f) != pixels;
    bad |= fwrite(screen_bottom, 2, pixels, f) != pixels;
    bad |= card_close(f, file) != 0;
    capture_error |= bad;
}
static void script_tick(void *ctx) {
    (void)ctx;
    /* The UI demo driver normally installs a frame clock; the product keeps
     * real bus-clock milliseconds even while its input is scripted. */
    if (!script_frames) chatds_ui_set_clock(&ui, chatds_ui_millis, NULL);
    if (runtime.mathnb_active && !chatds_ui_is_about(&ui))
        mnb_render(&runtime.mathnb, &runtime.mathnb_values, screen_top);
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
    char file[96];
    if (cds_path_join(file, sizeof file, name, ".txt")) { capture_error = 1; return; }
    FILE *f = card_open(file, "w");
    if (!f) { capture_error = 1; return; }
    chatds_script_state_text(state, state_sink, f);
    capture_error |= card_close(f, file) != 0;
}
static int run_script(void) {
    FILE *f = card_open("ui-script.txt", "rb");
    if (!f) return 0;
    /* Optional instrumentation: bound at 8191 bytes, but allocate only
     * the actual script. Small captures must not burn an artificial 8 KiB. */
    long script_size;
    if (fseek(f, 0, SEEK_END) || (script_size = ftell(f)) < 0 ||
        script_size > 8191 || fseek(f, 0, SEEK_SET)) {
        fclose(f); capture_error = 1; return 1;
    }
    char *script = malloc((size_t)script_size + 1);
    if (!script) { fclose(f); capture_error = 1; return 1; }
    size_t n = fread(script, 1, (size_t)script_size, f);
    script[n] = 0;
    int bad = n != (size_t)script_size || ferror(f) || fgetc(f) != EOF;
    fclose(f);
    frame_index = card_open("frames.txt", "w");
    if (!frame_index) bad = 1;
    if (!bad) {
        chatds_script_host host = {NULL, script_tick, script_dump, screen_top, screen_bottom};
        bad = chatds_script_run(&ui, script, &host);
    }
    free(script);
    if (frame_index) {
        bad |= card_close(frame_index, "frames.txt") != 0;
    }
    f = card_open("ui-result.txt", "w");
    if (f) {
        fprintf(f, "frames=%u\nstatus=%s\n", frame_count,
                bad || capture_error || publication_failed ? "FAIL" : "OK");
        card_close(f, "ui-result.txt");
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
        int redrawn = chatds_ui_render(&ui, top, bottom);
        if (redrawn && runtime.mathnb_active && !chatds_ui_is_about(&ui))
            mnb_render(&runtime.mathnb, &runtime.mathnb_values, top);
    }
}
#endif

int main(int argc, char **argv) {
    const char *model_path, *tok_path, *kb_path, *run_path;
    /* KB4 open consumes its path before the learned-head directory is saved.
     * Share storage; the directory then remains live until main exits. */
    char asset_path[256];
    int rc = 1, serve = 0;
#ifdef ARM9
    (void)argc; (void)argv;
    consoleDemoInit();
    cpuStartTiming(0);
    stack_paint();
    if (!fatInitDefault()) {
        cds_storage_error("FAIL: fatInitDefault (check launch mode/card)", fatGetDefaultDrive()); goto idle;
    }
    directory = cds_storage_root(fatGetDefaultDrive());
    if (!directory) {
        cds_storage_error("FAIL: unsupported storage drive (only fat:/ and sd:/)", fatGetDefaultDrive()); goto idle;
    }
    printf("storage: %s\n", directory);
    char model_file[32], tok_file[32], kb_file[32], run_file[32];
    if (cds_path_join(model_file, sizeof model_file, directory, "model.bin") ||
        cds_path_join(tok_file, sizeof tok_file, directory, "tok.bin") ||
        cds_path_join(kb_file, sizeof kb_file, directory, "kb.bin") ||
        cds_path_join(run_file, sizeof run_file, directory, "run.txt")) {
        cds_storage_error("FAIL: storage path", directory); goto idle;
    }
    model_path = model_file; tok_path = tok_file; kb_path = kb_file; run_path = run_file;
    const char *kernel_check = cds_matvec_self_check() ?
        "gs32 boot self-check PASS" : "gs32 boot self-check FAIL; asm disabled; using C";
    puts(kernel_check);
    FILE *kernel_log = card_open("kernel.txt", "w");
    if (kernel_log) { fprintf(kernel_log, "%s\n", kernel_check); card_close(kernel_log, "kernel.txt"); }

#else
    if (argc != 6 && argc != 7) { fprintf(stderr, "usage: chatds-host MODEL TOK KB-or-- RUN OUTDIR/ [--serve]\n"); return 2; }
    serve = argc == 7 && !strcmp(argv[6], "--serve");
    if (argc == 7 && !serve) return 2;
    model_path = argv[1]; tok_path = argv[2]; kb_path = argv[3]; run_path = argv[4]; directory = argv[5];
    size_t dir_len = strlen(directory);
    if (!dir_len || directory[dir_len - 1] != '/') {
        fprintf(stderr, "OUTDIR must end with '/'\n"); return 2;
    }
#endif
    previous = ticks();
    int config_status = cds_config_read(run_path, &config);
    if (config_status == -1) {
        cds_storage_error("cannot read run.txt", run_path); goto fail;
    }
    if (config_status || config.steps > 256 || !config.steps) {
        fprintf(stderr, "malformed run.txt: %s (check values; steps must be 1..256)\n", run_path); goto fail;
    }
    chatds_status status = chatds_model_load(model_path, &model);
    if (status != CHATDS_OK) { fprintf(stderr, "%s\n", chatds_status_string(status)); goto fail; }
    if (tok_load(&tokenizer, tok_path)) { fprintf(stderr, "invalid tokenizer\n"); goto fail; }
    const chatds_model_config *shape = chatds_model_get_config(model);
    if ((uint32_t)tokenizer.n != shape->vocab_size) goto fail;
    prompt_ids = malloc(shape->context_length * sizeof(*prompt_ids));
    if (!prompt_ids) goto fail;
#ifdef ARM9
    /* UI owns A/C. D (128 KiB) + E (64 KiB) form one contiguous LCDC
     * arena at 0x06860000. Reserve keys there; values remain main-RAM resident.
     * Halfword-only stores; no model paging, no shorter context, no KV quantization change. */
    size_t key_bytes = (size_t)shape->layers * shape->context_length * shape->kv_heads *
        (shape->dim / shape->heads) * sizeof(int16_t);
    if (key_bytes > 192 * 1024) goto fail;
    vramSetBankD(VRAM_D_LCD); vramSetBankE(VRAM_E_LCD);
    status = chatds_session_create_with_keys(model, NULL, (int16_t *)0x06860000, 192 * 1024, &session);
#else
    status = chatds_session_create(model, NULL, &session);
#endif
    if (status != CHATDS_OK) goto fail;
    /* Card KB4 supersedes KB3. A malformed/unreadable KB4 is fatal, never
     * disguised as an absent optional asset. Host accepts explicit KB4 path. */
#ifdef ARM9
    if (cds_path_join(asset_path, sizeof asset_path, directory, "kb4.bin")) goto path_fail;
#else
    FILE *probe = fopen(kb_path, "rb"); char magic[4] = {0};
    if (probe) { if (fread(magic, 1, 4, probe) != 4) magic[0] = 0; fclose(probe); }
    if (!memcmp(magic, "KB4", 3)) {
        if (cds_path_join(asset_path, sizeof asset_path, kb_path, "")) goto path_fail;
    } else {
        const char *slash = strrchr(kb_path, '/');
        if (slash) {
            size_t n = (size_t)(slash - kb_path) + 1;
            if (n >= sizeof asset_path) { errno = ENAMETOOLONG; goto path_fail; }
            memcpy(asset_path, kb_path, n); asset_path[n] = 0;
            if (n + strlen("kb4.bin") >= sizeof asset_path) { errno = ENAMETOOLONG; goto path_fail; }
            strcpy(asset_path + n, "kb4.bin");
        } else if (cds_path_join(asset_path, sizeof asset_path, directory, "kb4.bin")) goto path_fail;
    }
#endif
    if (c14_open(&runtime, &generator, asset_path)) goto fail;
    if (!runtime.kb4.f) {
        kb = &runtime.scratch.legacy;
        cds_kb_status ks = cds_kb_open(kb, kb_path);
        if (ks == CDS_KB_BAD) goto fail;
    }
    tok = (cds_tokenizer){encode, piece, &tokenizer};
    chatds_ui_engine_gen_init(&generator, session, &tok, kb && kb->f ? kb : NULL,
                             config.instruct, prompt_ids, shape->context_length, config.steps);
#ifdef ARM9
    if (cds_path_join(asset_path, sizeof asset_path, directory, "")) goto path_fail;
#else
    const char *learned_env = getenv("CHATDS_ROUND3_DIR");
    const char *model_slash = strrchr(model_path, '/');
    if (learned_env) {
        if (cds_path_join(asset_path, sizeof asset_path, learned_env, "")) goto path_fail;
    } else if (model_slash) {
        size_t n = (size_t)(model_slash - model_path) + 1;
        if (n >= sizeof asset_path) { errno = ENAMETOOLONG; goto path_fail; }
        memcpy(asset_path, model_path, n); asset_path[n] = 0;
    } else strcpy(asset_path, "./");
#endif
    if (config.instruct && c24_models_init(session, asset_path)) {
        fprintf(stderr, "invalid round3 models or insufficient fallback workspace\n"); goto fail;
    }
#ifdef ARM9
    /* Refuse this prototype ROM if the established 3.5 MiB RAM ceiling
     * or the allocator leaves less than 64 KiB after resident initialization. */
    if ((uintptr_t)getHeapEnd() + 65536u > 0x02380000u ||
        (uintptr_t)getHeapEnd() + 65536u > (uintptr_t)getHeapLimit()) {
        fprintf(stderr, "mathnb: insufficient RAM margin\n"); goto fail;
    }
#endif
    stream_piece = generator.opts.on_piece;
    generator.opts.on_piece = observed_piece;
    generator.opts.out_ids = output_ids;
    generator.opts.text = answer_text;
    generator.opts.text_cap = sizeof answer_text;
    wrapped = (chatds_ui_gen){NULL, begin, step, cancel};
#ifdef ARM9
    (void)serve;
    if (card_session()) goto finish;
    if (config.kbd) interactive();
#else
    if (serve) { rc = session_input(stdin, stdout); goto finish; }
#endif
    rc = batch(0);
    goto finish;
path_fail:
    cds_storage_error("asset path failed (no truncation)", directory);
fail:
    publish(0);
finish:
    c14_close(&runtime);
    if (kb) { cds_kb_close(kb); kb = NULL; }
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
