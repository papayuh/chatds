/* SPDX-License-Identifier: MIT */
/* ChatDS tokenizer. Written from the sentencepiece BPE algorithm (merge the
 * highest-scoring adjacent pair, leftmost on ties) and the tok.bin layout in
 * tok.h; see ds/tokenizer/test_tok.py for the parity check. */
#include "tok.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FIRST_NORMAL (TOK_BYTE0 + 256)
#define MAX_PIECE 255

/* This model's trainer_spec.unk_surface, as sentencepiece decodes <unk>. */
static const char UNK_SURFACE[] = " \\342\\201\\207 ";
static const unsigned char FFFD[3] = {0xEF, 0xBF, 0xBD};
static unsigned char byte_tab[256];

static unsigned rd32(const unsigned char *p)
{
    return p[0] | p[1] << 8 | p[2] << 16 | (unsigned)p[3] << 24;
}

static int cmp_bytes(const unsigned char *a, int na, const unsigned char *b, int nb)
{
    int r = memcmp(a, b, na < nb ? na : nb);
    return r ? r : na - nb;
}

/* ponytail: qsort has no context arg; tok_load is not reentrant. */
static const tok_t *sort_tok;
static int cmp_ids(const void *a, const void *b)
{
    int x = *(const unsigned short *)a, y = *(const unsigned short *)b;
    return cmp_bytes(sort_tok->piece[x], sort_tok->len[x], sort_tok->piece[y], sort_tok->len[y]);
}

int tok_load(tok_t *t, const char *path)
{
    FILE *f = fopen(path, "rb");
    long size = 0;
    size_t off;
    int i;

    memset(t, 0, sizeof *t);
    if (!f)
        return -1;
    if (!fseek(f, 0, SEEK_END) && (size = ftell(f)) >= 4 && !fseek(f, 0, SEEK_SET))
        t->buf = malloc(size);
    if (!t->buf || fread(t->buf, 1, size, f) != (size_t)size) {
        fclose(f);
        goto bad;
    }
    fclose(f);

    t->max_len = rd32(t->buf);
    if (t->max_len < 1 || t->max_len > MAX_PIECE)
        goto bad;
    for (off = 4; off < (size_t)size; t->n++) {
        size_t n;
        if (size - off < 8 || t->n == 65535)
            goto bad;
        n = rd32(t->buf + off + 4);
        if (n > (size_t)t->max_len || size - off - 8 < n)
            goto bad;
        off += 8 + n;
    }
    if (t->n < FIRST_NORMAL)
        goto bad;

    t->piece = malloc(t->n * sizeof *t->piece);
    t->len = malloc(t->n * sizeof *t->len);
    t->score = malloc(t->n * sizeof *t->score);
    t->sorted = malloc(t->n * sizeof *t->sorted);
    if (!t->piece || !t->len || !t->score || !t->sorted)
        goto bad;
    for (off = 4, i = 0; i < t->n; i++) {
        memcpy(&t->score[i], t->buf + off, 4); /* unaligned on purpose */
        t->len[i] = rd32(t->buf + off + 4);
        t->piece[i] = t->buf + off + 8;
        off += 8 + t->len[i];
    }
    /* We rely on byte fallback living at 3..258. */
    for (i = 0; i < 256; i++) {
        char want[8];
        snprintf(want, sizeof want, "<0x%02X>", i);
        if (cmp_bytes(t->piece[TOK_BYTE0 + i], t->len[TOK_BYTE0 + i],
                      (const unsigned char *)want, 6))
            goto bad;
        byte_tab[i] = i;
    }
    for (i = FIRST_NORMAL; i < t->n; i++)
        t->sorted[t->n_sorted++] = i;
    sort_tok = t;
    qsort(t->sorted, t->n_sorted, sizeof *t->sorted, cmp_ids);
    return 0;
bad:
    tok_free(t);
    return -1;
}

void tok_free(tok_t *t)
{
    free(t->buf);
    free(t->piece);
    free(t->len);
    free(t->score);
    free(t->sorted);
    memset(t, 0, sizeof *t);
}

/* Normal piece id with exactly these bytes, or -1. */
static int find(const tok_t *t, const unsigned char *s, int n)
{
    int lo = 0, hi = t->n_sorted - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2, id = t->sorted[mid];
        int r = cmp_bytes(t->piece[id], t->len[id], s, n);
        if (!r)
            return id;
        if (r < 0)
            lo = mid + 1;
        else
            hi = mid - 1;
    }
    return -1;
}

static int merge_id(const tok_t *t, int a, int b)
{
    unsigned char tmp[2 * MAX_PIECE];
    if (a < FIRST_NORMAL || b < FIRST_NORMAL || t->len[a] + t->len[b] > t->max_len)
        return -1;
    memcpy(tmp, t->piece[a], t->len[a]);
    memcpy(tmp + t->len[a], t->piece[b], t->len[b]);
    return find(t, tmp, t->len[a] + t->len[b]);
}

/* Length of the well-formed UTF-8 char at p, 0 if malformed. */
static int utf8_len(const unsigned char *p, size_t n)
{
    unsigned c = p[0], lo = 0x80, hi = 0xBF;
    int k, i;

    if (c < 0x80)
        return 1;
    if (c >= 0xC2 && c <= 0xDF)
        k = 2;
    else if (c >= 0xE0 && c <= 0xEF) {
        k = 3;
        if (c == 0xE0)
            lo = 0xA0;
        if (c == 0xED)
            hi = 0x9F;
    } else if (c >= 0xF0 && c <= 0xF4) {
        k = 4;
        if (c == 0xF0)
            lo = 0x90;
        if (c == 0xF4)
            hi = 0x8F;
    } else
        return 0;
    if (n < (size_t)k || p[1] < lo || p[1] > hi)
        return 0;
    for (i = 2; i < k; i++)
        if (p[i] < 0x80 || p[i] > 0xBF)
            return 0;
    return k;
}

/* One char: its piece, else its bytes. */
static int push(const tok_t *t, int *sym, int n, const unsigned char *p, int k)
{
    int id = find(t, p, k), j;
    if (id >= 0)
        sym[n++] = id;
    else
        for (j = 0; j < k; j++)
            sym[n++] = TOK_BYTE0 + p[j];
    return n;
}

int tok_encode(const tok_t *t, const char *text, size_t len, int bos, int *ids, int max)
{
    const unsigned char *s = (const unsigned char *)text;
    size_t cap = 3 * len + 2, i = 0; /* a bad byte is U+FFFD: 3 byte ids */
    int *sym = malloc(2 * cap * sizeof *sym), *pair, n = 0, j, o = 0;

    if (!sym)
        return -1;
    pair = sym + cap;
    if (len) /* sentencepiece's dummy prefix */
        n = push(t, sym, n, (const unsigned char *)" ", 1);
    while (i < len) {
        const unsigned char *p = s + i;
        int k = utf8_len(p, len - i);
        i += k ? k : 1;
        if (!k)
            p = FFFD, k = 3;
        if (k == 3 && !memcmp(p, "\xE2\x96\x81", 3)) /* U+2581 is the space */
            p = (const unsigned char *)" ", k = 1;
        n = push(t, sym, n, p, k);
    }

    for (j = 0; j + 1 < n; j++)
        pair[j] = merge_id(t, sym[j], sym[j + 1]);
    for (;;) {
        int best = -1;
        for (j = 0; j + 1 < n; j++)
            if (pair[j] >= 0 && (best < 0 || t->score[pair[j]] > t->score[pair[best]]))
                best = j;
        if (best < 0)
            break;
        sym[best] = pair[best];
        memmove(sym + best + 1, sym + best + 2, (n - best - 2) * sizeof *sym);
        if (n - best - 3 > 0)
            memmove(pair + best + 1, pair + best + 2, (n - best - 3) * sizeof *pair);
        n--;
        if (best > 0)
            pair[best - 1] = merge_id(t, sym[best - 1], sym[best]);
        if (best + 1 < n)
            pair[best] = merge_id(t, sym[best], sym[best + 1]);
    }

    if (bos && o++ < max)
        ids[0] = TOK_BOS;
    for (j = 0; j < n; j++, o++)
        if (o < max)
            ids[o] = sym[j];
    free(sym);
    return o;
}

const char *tok_piece(const tok_t *t, int id, int *len)
{
    if (id < 0 || id >= t->n)
        return NULL;
    *len = 0;
    if (id == TOK_BOS || id == TOK_EOS)
        return "";
    if (id == TOK_UNK) {
        *len = sizeof UNK_SURFACE - 1;
        return UNK_SURFACE;
    }
    if (id < FIRST_NORMAL) {
        *len = 1;
        return (const char *)&byte_tab[id - TOK_BYTE0];
    }
    *len = t->len[id];
    return (const char *)t->piece[id];
}

static void put(char *out, int cap, int *o, const void *p, int k)
{
    int j;
    for (j = 0; j < k; j++, (*o)++)
        if (*o < cap - 1)
            out[*o] = ((const char *)p)[j];
}

/* Byte ids ids[0..m) as UTF-8, each malformed byte as U+FFFD. */
static void put_bytes(char *out, int cap, int *o, const int *ids, int m)
{
    int j = 0;
    while (j < m) {
        unsigned char b[4];
        int k, w = m - j < 4 ? m - j : 4;
        for (k = 0; k < w; k++)
            b[k] = ids[j + k] - TOK_BYTE0;
        k = utf8_len(b, w);
        put(out, cap, o, k ? b : FFFD, k ? k : 3);
        j += k ? k : 1;
    }
}

int tok_decode(const tok_t *t, const int *ids, int n, char *out, int cap)
{
    int o = 0, first = 1, run = -1, i;

    for (i = 0; i <= n; i++) {
        const char *p;
        int k, id = i < n ? ids[i] : TOK_EOS; /* EOS sentinel flushes the run */
        if (id < 0 || id >= t->n)
            return -1;
        if (id >= TOK_BYTE0 && id < FIRST_NORMAL) {
            if (run < 0)
                run = i;
            first = 0;
            continue;
        }
        if (run >= 0)
            put_bytes(out, cap, &o, ids + run, i - run);
        run = -1;
        if (id == TOK_BOS || id == TOK_EOS)
            continue;
        p = tok_piece(t, id, &k);
        if (!p)
            return -1;
        if (first && id != TOK_UNK && *p == ' ')
            p++, k--;
        first = 0;
        put(out, cap, &o, p, k);
    }
    if (cap > 0)
        out[o < cap ? o : cap - 1] = 0;
    return o;
}
