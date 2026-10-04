/* SPDX-License-Identifier: MIT
 * Retrieval gate, n-gram KB retrieval and the prompt: twin of corpus/chatds_runtime.py. */
#include <string.h>
#include "plumbing.h"

/* Word lists copied from chatds_runtime.py; test_parity.py diffs them against it. */
const char *const cds_stop_words =
    "what who where when why how is are was were the a an of in on to do does can i you me my your "
    "tell about whats which whom this that it its and or for with at by from be as if so please explain define "
    "describe give say know mean means there their they he she we us our "
    "continent currency language capital country city money name main official primary spoken speak located called "
    "use used people plus minus times divided multiply add subtract percent calculate much";
const char *const cds_cue_words = "who what whats where when which define explain describe";
const char *const cds_code_words = "list dict string function variable loop file command python code print return "
    "sort reverse count plus minus times divided multiplied percent";
const char *const cds_code_chars = "()[]{}=\"`_";
const char *const cds_num_words = "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety "
    "hundred thousand million";
const char *const cds_math_words = "half double twice triple quarter over sum product added add subtract "
    "subtracted multiply divide difference squared cubed and away take more less than";

#define MAX_TRIES 40
#define FUZZY_MIN 5
#define MAX_WORDS (CDS_QUESTION_MAX / 2 + 1)

typedef struct {
    char buf[CDS_QUESTION_MAX + 1];
    const char *w[MAX_WORDS];
    unsigned char len[MAX_WORDS];
    int n;
} words_t;

static int in_list(const char *list, const char *w, size_t n) {
    for (const char *p = list; *p;) {
        const char *e = strchr(p, ' ');
        size_t k = e ? (size_t)(e - p) : strlen(p);
        if (k == n && memcmp(p, w, n) == 0)
            return 1;
        p += k + (e != NULL);
    }
    return 0;
}

int cds_normalize(const char *s, char *out, size_t cap) {
    size_t n = 0;
    int gap = 0;
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c >= 0x80)
            continue; /* ponytail: kb.py transliterates (NFKD); the DS keyboard only types ASCII */
        if (c >= 'A' && c <= 'Z')
            c += 'a' - 'A';
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'))) {
            gap = n > 0;
            continue;
        }
        if (n + gap + 1 >= cap)
            return -1;
        if (gap)
            out[n++] = ' ';
        out[n++] = (char)c;
        gap = 0;
    }
    if (cap)
        out[n] = 0;
    return cap ? (int)n : -1;
}

static int split(const char *question, words_t *ws) {
    if (strlen(question) > CDS_QUESTION_MAX || cds_normalize(question, ws->buf, sizeof ws->buf) < 0)
        return -1;
    ws->n = 0;
    for (char *p = ws->buf; *p;) {
        char *e = strchr(p, ' ');
        size_t k = e ? (size_t)(e - p) : strlen(p);
        ws->w[ws->n] = p, ws->len[ws->n++] = (unsigned char)k;
        p += k + (e != NULL);
    }
    return 0;
}

static int any_in(const words_t *ws, const char *list) {
    for (int i = 0; i < ws->n; i++)
        if (in_list(list, ws->w[i], ws->len[i]))
            return 1;
    return 0;
}

static int letter(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
static int digit(char c) { return c >= '0' && c <= '9'; }
static int op(char c) { return c && strchr("-+*/%^", c); }
static int space(char c) { return c && strchr(" \t\n\r\v\f\x1c\x1d\x1e\x1f", c); } /* Python str \s, ASCII */

static int gate(const char *q, const words_t *ws) {
    if (!(ws->n && in_list(cds_cue_words, ws->w[0], ws->len[0])) && strncmp(ws->buf, "tell me about ", 14))
        return 0;
    for (const char *p = q; *p; p++)
        if (strchr(cds_code_chars, *p))
            return 0;
    if (any_in(ws, cds_code_words) || (any_in(ws, cds_num_words) && any_in(ws, cds_math_words)))
        return 0;
    for (size_t i = 0; q[i]; i++) /* a quote, but not a contraction (what's) */
        if (q[i] == '\'' && (i == 0 || !letter(q[i - 1]) || !letter(q[i + 1])))
            return 0;
    for (size_t i = 0; q[i]; i++) { // \d\s*[-+*/%^] or [-+*/%^]\s*\d
        if (!digit(q[i]) && !op(q[i]))
            continue;
        size_t j = i + 1;
        while (space(q[j]))
            j++;
        if (digit(q[i]) ? op(q[j]) : digit(q[j]))
            return 0;
    }
    return 1;
}

int cds_wants_context(const char *question) {
    words_t ws;
    return split(question, &ws) == 0 && gate(question, &ws);
}

/* Joins words[i..i+n) with spaces, word j replaced by sub (if not NULL). */
static void join(const words_t *ws, int i, int n, int j, const char *sub, char *key) {
    size_t k = 0;
    for (int x = i; x < i + n; x++) {
        const char *w = x == j && sub ? sub : ws->w[x];
        size_t l = x == j && sub ? strlen(sub) : ws->len[x];
        if (x > i)
            key[k++] = ' ';
        memcpy(key + k, w, l);
        k += l;
    }
    key[k] = 0;
}

static int fuzzy(cds_kb *kb, const char *w, size_t len, const char **cands) {
    for (int i = 0; i < kb->fz_n; i++)
        if (strlen(kb->fz_word[i]) == len && memcmp(kb->fz_word[i], w, len) == 0) {
            for (int c = 0; c < kb->fz_ncand[i]; c++)
                cands[c] = kb->fz_cand[i][c];
            return kb->fz_ncand[i];
        }
    if (kb->fz_n == CDS_FZ_WORDS)
        return 0; /* MAX_FUZZY_WORDS distinct words per question; later ones get no candidates */
    int i = kb->fz_n++;
    memcpy(kb->fz_word[i], w, len);
    kb->fz_word[i][len] = 0;
    kb->fz_ncand[i] = cds_kb_fuzzy(kb, kb->fz_word[i], kb->fz_cand[i]);
    for (int c = 0; c < kb->fz_ncand[i]; c++)
        cands[c] = kb->fz_cand[i][c];
    return kb->fz_ncand[i];
}

int cds_retrieve(cds_kb *kb, const char *question, char *out) {
    words_t ws;
    char key[CDS_QUESTION_MAX + CDS_KEY_MAX + 2];
    int content = 0, tries = 0, ftries = 0;
    if (!kb || !kb->f)
        return 0;
    kb->err = 0, kb->lookups = 0, kb->sector_reads = 0, kb->fz_n = 0;
    if (split(question, &ws) || !gate(question, &ws))
        return 0;
    for (int i = 0; i < ws.n; i++)
        content += !in_list(cds_stop_words, ws.w[i], ws.len[i]);
    for (int n = 4; n >= 1; n--) {
        unsigned char ok[MAX_WORDS];
        for (int i = 0; i + n <= ws.n; i++) {
            /* covers a strict majority of the content words; no all-stopword grams or 1-grams under 3 chars */
            int c = 0;
            for (int x = i; x < i + n; x++)
                c += !in_list(cds_stop_words, ws.w[x], ws.len[x]);
            ok[i] = 2 * c > content && c > 0 && !(n == 1 && ws.len[i] < 3);
        }
        for (int i = 0; i + n <= ws.n; i++) {
            if (!ok[i])
                continue;
            if (tries >= MAX_TRIES)
                return 0;
            tries++;
            join(&ws, i, n, -1, NULL, key);
            if (cds_kb_lookup(kb, key, out))
                return 1;
            if (kb->err)
                return 0;
        }
        for (int i = 0; i + n <= ws.n; i++) {
            int longw = ok[i];
            for (int x = i; longw && x < i + n; x++)
                longw = ws.len[x] >= FUZZY_MIN;
            if (!longw)
                continue;
            for (int j = i; j < i + n; j++) {
                const char *cands[CDS_FZ_CANDS];
                int nc = fuzzy(kb, ws.w[j], ws.len[j], cands);
                if (kb->err)
                    return 0;
                for (int c = 0; c < nc; c++) {
                    if (ftries >= MAX_TRIES)
                        return 0;
                    ftries++;
                    join(&ws, i, n, j, cands[c], key);
                    if (cds_kb_lookup(kb, key, out))
                        return 1;
                    if (kb->err)
                        return 0;
                }
            }
        }
    }
    return 0;
}

int cds_format_prompt(const char *question, const char *ctx, char *out, size_t cap) {
    int has = ctx && *ctx;
    size_t n = (has ? 3 + strlen(ctx) + 1 : 0) + 3 + strlen(question) + 3;
    if (n + 1 > cap)
        return -1;
    out[0] = 0;
    if (has) {
        strcat(out, "C: ");
        strcat(out, ctx);
        strcat(out, "\n");
    }
    strcat(out, "Q: ");
    strcat(out, question);
    strcat(out, "\nA:");
    return (int)n;
}
