/* SPDX-License-Identifier: MIT
 * Host probe for test_parity.py. Reads one escaped string per stdin line, writes one result line.
 *   cds-prompt prompt [KB]   question -> ctx \t prompt
 *   cds-prompt calc          model output -> apply_calc result (or the output unchanged)
 *   cds-prompt gate          question -> 0/1
 *   cds-prompt fuzzy KB      word -> candidates, space separated
 *   cds-prompt lists         the word lists, one "name: words" per line
 * Escaping: '\\' and bytes outside 0x20..0x7e are written \xHH. */
#include <stdlib.h>
#include <string.h>
#include "plumbing.h"

extern const char *const cds_stop_words, *const cds_cue_words, *const cds_code_words,
    *const cds_code_chars, *const cds_num_words, *const cds_math_words;

static void put(const char *s) {
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c < 0x20 || c > 0x7e || c == '\\')
            printf("\\x%02x", c);
        else
            putchar(c);
    }
}

static int get(char *out, size_t cap) {
    static char line[4096];
    if (!fgets(line, sizeof line, stdin))
        return 0;
    size_t n = 0;
    for (char *p = line; *p && *p != '\n' && n + 1 < cap; p++) {
        if (p[0] == '\\' && p[1] == 'x') {
            char h[3] = {p[2], p[3], 0};
            out[n++] = (char)strtol(h, NULL, 16);
            p += 3;
        } else {
            out[n++] = *p;
        }
    }
    out[n] = 0;
    return 1;
}

int main(int argc, char **argv) {
    static char in[4096], out[4096 + 64];
    cds_kb kb;
    const char *mode = argc > 1 ? argv[1] : "";
    int have_kb = argc > 2;
    if (have_kb && cds_kb_open(&kb, argv[2]) != CDS_KB_OK) {
        fprintf(stderr, "cds-prompt: cannot open KB %s\n", argv[2]);
        return 2;
    }
    if (!strcmp(mode, "lists")) {
        printf("STOP: %s\nCUES: %s\nCODE_WORDS: %s\nCODE_CHARS: %s\nNUM_WORDS: %s\nMATH_WORDS: %s\n", cds_stop_words,
               cds_cue_words, cds_code_words, cds_code_chars, cds_num_words, cds_math_words);
        return 0;
    }
    while (get(in, sizeof in)) {
        if (!strcmp(mode, "prompt")) {
            static char ctx[CDS_CTX_CAP], prompt[CDS_PROMPT_CAP];
            if (!cds_retrieve(have_kb ? &kb : NULL, in, ctx))
                ctx[0] = 0;
            if (cds_format_prompt(in, ctx, prompt, sizeof prompt) < 0)
                strcpy(prompt, "<too long>");
            put(ctx);
            putchar('\t');
            put(prompt);
        } else if (!strcmp(mode, "calc")) {
            put(cds_apply_calc(in, out, sizeof out) ? out : in);
        } else if (!strcmp(mode, "gate")) {
            putchar('0' + cds_wants_context(in));
        } else if (!strcmp(mode, "fuzzy") && have_kb) {
            char c[CDS_FZ_CANDS][CDS_KEY_MAX + 1];
            int n = cds_kb_fuzzy(&kb, in, c);
            for (int i = 0; i < n; i++)
                printf(i ? " %s" : "%s", c[i]);
        } else {
            fprintf(stderr, "usage: cds-prompt prompt [KB] | calc | gate | fuzzy KB | lists\n");
            return 2;
        }
        putchar('\n');
    }
    if (have_kb)
        cds_kb_close(&kb);
    return 0;
}
