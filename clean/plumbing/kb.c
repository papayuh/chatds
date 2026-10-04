/* SPDX-License-Identifier: MIT
 * KB2/KB3 reader, twin of corpus/kb/kb.py (layout: corpus/kb/FORMAT.md). */
#include <string.h>
#include "plumbing.h"
#ifdef ARM9
#include <fat.h>
#endif

#define SECTOR 512u
#define MAX_SPAN 4096u /* real buckets hold ~2.6 entries; a garbage table must not hang a question */
#define NOFF 16        /* fuzzy offsets considered, like kb.py's heapq.nsmallest(16) */

static uint32_t le32(const unsigned char *p) {
    return p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static uint32_t fnv1a(const char *s, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++)
        h = (h ^ (unsigned char)s[i]) * 16777619u;
    return h;
}

static int fill(cds_kb *kb, uint32_t s) {
    uint32_t n = kb->size - s * SECTOR < SECTOR ? kb->size - s * SECTOR : SECTOR;
    for (int tries = 0; tries < 2; tries++) { /* one retry: a card read can fail transiently */
        clearerr(kb->f);
        if (fseek(kb->f, (long)s * SECTOR, SEEK_SET) == 0 && fread(kb->buf, 1, n, kb->f) == n) {
            kb->sector = s;
            kb->sector_reads++;
            return 0;
        }
    }
    kb->sector = UINT32_MAX;
    return -1;
}

/* Reads n bytes at off through the one-sector buffer; re-reading the buffered sector is free. */
static int rd(cds_kb *kb, uint32_t off, void *dst, uint32_t n) {
    unsigned char *d = dst;
    if (kb->err || off > kb->size || n > kb->size - off) {
        kb->err = 1;
        return -1;
    }
    while (n) {
        uint32_t s = off / SECTOR, at = off % SECTOR, k = SECTOR - at < n ? SECTOR - at : n;
        if (s != kb->sector && fill(kb, s)) {
            kb->err = 1;
            return -1;
        }
        memcpy(d, kb->buf + at, k);
        d += k, off += k, n -= k;
    }
    return 0;
}

static int pow2(uint32_t x) { return x && !(x & (x - 1)); }

cds_kb_status cds_kb_open(cds_kb *kb, const char *path) {
    unsigned char h[60];
    memset(kb, 0, sizeof *kb);
    kb->sector = UINT32_MAX;
    if (!(kb->f = fopen(path, "rb")))
        return CDS_KB_MISSING;
    setvbuf(kb->f, NULL, _IONBF, 0); /* buf is the cache; no second stdio buffer on the heap */
    long real = -1;
    if (fseek(kb->f, 0, SEEK_END) == 0)
        real = ftell(kb->f);
    if (real < (long)SECTOR || fseek(kb->f, 0, SEEK_SET) || fread(h, 1, sizeof h, kb->f) != sizeof h)
        goto bad;
    uint32_t ver = le32(h + 4);
    kb->version = ver;
    if (!((memcmp(h, "KB3\0", 4) == 0 && ver == 3) || (memcmp(h, "KB2\0", 4) == 0 && ver == 2)))
        goto bad;
    kb->nb = le32(h + 8), kb->bt = le32(h + 20), kb->ko = le32(h + 24), kb->to = le32(h + 28);
    kb->size = le32(h + 32);
    kb->fnb = le32(h + 36), kb->fbt = le32(h + 40), kb->fent = le32(h + 44), kb->fwo = le32(h + 48);
    if ((unsigned long)real != kb->size || !pow2(kb->nb) || !pow2(kb->fnb))
        goto bad;
    uint32_t off[] = {kb->bt, kb->ko, kb->to, kb->fbt, kb->fent, kb->fwo};
    for (unsigned i = 0; i < sizeof off / sizeof *off; i++)
        if (off[i] >= kb->size)
            goto bad;
    if ((kb->size - kb->bt) / 4 <= kb->nb || (kb->size - kb->fbt) / 4 <= kb->fnb)
        goto bad;
#ifdef ARM9
    fatInitLookupCacheFile(kb->f, 4096); /* fast backward seeks in a 42 MB file; optional */
#endif
    return CDS_KB_OK;
bad:
    fclose(kb->f);
    kb->f = NULL;
    return CDS_KB_BAD;
}

void cds_kb_close(cds_kb *kb) {
    if (kb->f)
        fclose(kb->f);
    kb->f = NULL;
}

/* Bucket [a, e) from a table at tab with n_buckets nb. */
static int bucket(cds_kb *kb, uint32_t tab, uint32_t b, uint32_t *a, uint32_t *e) {
    unsigned char p[8];
    if (rd(kb, tab + 4 * b, p, 8))
        return -1;
    *a = le32(p), *e = le32(p + 4);
    if (*e < *a) {
        kb->err = 1;
        return -1;
    }
    return 0;
}

int cds_kb_lookup(cds_kb *kb, const char *query, char *out) {
    char key[CDS_KEY_MAX + 1];
    unsigned char p[5];
    int kl = cds_normalize(query, key, sizeof key);
    uint32_t a, e;
    kb->lookups++;
    if (kl <= 0)
        return 0;
    uint32_t h = fnv1a(key, kl);
    if (bucket(kb, kb->bt, h & (kb->nb - 1), &a, &e))
        return 0;
    if (e - a > MAX_SPAN * (9 + CDS_KEY_MAX)) {
        kb->err = 1;
        return 0;
    }
    for (uint32_t i = a; i < e;) {
        char k[CDS_KEY_MAX + 1];
        if (rd(kb, kb->ko + i, p, 5)) /* ko + i overflow is caught by rd's size check */
            return 0;
        uint32_t ln = p[4];
        if (le32(p) == h && ln == (uint32_t)kl) {
            unsigned char t[4];
            if (rd(kb, kb->ko + i + 5, k, ln) || rd(kb, kb->ko + i + 5 + ln, t, 4))
                return 0;
            if (memcmp(k, key, ln) == 0) {
                uint32_t to = le32(t);
                unsigned char n;
                if (to > kb->size - kb->to || rd(kb, kb->to + to, &n, 1) || rd(kb, kb->to + to + 1, out, n)) {
                    kb->err = 1;
                    return 0;
                }
                out[n] = 0;
                /* Frozen KB2 ROM behavior: one sentence in an 80-byte buffer,
                 * with a truncated final word discarded. KB3 stores already
                 * curated sentences and must not receive this legacy limit. */
                if (kb->version == 2) {
                    if (n >= 80) {
                        out[79] = 0;
                        char *space = strrchr(out, ' ');
                        if (space) *space = 0;
                    }
                    char *end = strchr(out, '.');
                    if (end) end[1] = 0;
                }
                return 1;
            }
        }
        i += 9 + ln;
    }
    return 0;
}

/* variants(a) & variants(b) != {} : a and b agree after deleting at most one char from each. */
static int del_eq(const char *a, size_t na, size_t i, const char *b, size_t nb, size_t j) {
    /* a without a[i] (i == na: no deletion) equals b without b[j] */
    if ((na - (i < na)) != (nb - (j < nb)))
        return 0;
    for (size_t x = 0, y = 0; x < na || y < nb; x++, y++) {
        if (x == i) x++;
        if (y == j) y++;
        if (x >= na || y >= nb)
            return x >= na && y >= nb;
        if (a[x] != b[y])
            return 0;
    }
    return 1;
}

static int share_variant(const char *a, const char *b) {
    size_t na = strlen(a), nb = strlen(b);
    for (size_t i = 0; i <= na; i++)
        for (size_t j = 0; j <= nb; j++)
            if (del_eq(a, na, i, b, nb, j))
                return 1;
    return 0;
}

int cds_kb_fuzzy(cds_kb *kb, const char *q, char out[CDS_FZ_CANDS][CDS_KEY_MAX + 1]) {
    uint32_t offs[NOFF];
    int noff = 0, nout = 0;
    size_t nq = strlen(q);
    char v[CDS_KEY_MAX + 1];
    if (nq > CDS_KEY_MAX)
        return 0; /* ponytail: indexed words are 5..20 chars; a >63-char word has no match */
    /* q itself (d == nq), then each single deletion; deleting either of two equal neighbours
     * gives the same variant, so it is probed once (kb.py probes a set). */
    for (size_t d = nq + 1; d-- > 0;) {
        if (d < nq && d > 0 && q[d] == q[d - 1])
            continue;
        size_t n = 0;
        for (size_t i = 0; i < nq; i++)
            if (i != d)
                v[n++] = q[i];
        uint32_t h = fnv1a(v, n), a, e;
        if (bucket(kb, kb->fbt, h & (kb->fnb - 1), &a, &e))
            return 0;
        if (e - a > 8 * MAX_SPAN) {
            kb->err = 1;
            return 0;
        }
        for (uint32_t i = a; i + 8 <= e; i += 8) {
            unsigned char p[8];
            if (rd(kb, kb->fent + i, p, 8))
                return 0;
            if (le32(p) != h)
                continue;
            uint32_t wo = le32(p + 4);
            /* keep the NOFF smallest distinct offsets, ascending */
            int at = noff;
            while (at > 0 && offs[at - 1] > wo)
                at--;
            if ((at > 0 && offs[at - 1] == wo) || at == NOFF)
                continue;
            if (noff < NOFF)
                noff++;
            memmove(offs + at + 1, offs + at, (noff - 1 - at) * sizeof *offs);
            offs[at] = wo;
        }
    }
    for (int i = 0; i < noff && nout < CDS_FZ_CANDS; i++) {
        unsigned char n;
        char w[256];
        if (offs[i] > kb->size - kb->fwo || rd(kb, kb->fwo + offs[i], &n, 1) || rd(kb, kb->fwo + offs[i] + 1, w, n))
            return nout;
        w[n] = 0;
        if (n > CDS_KEY_MAX || strcmp(w, q) == 0 || !share_variant(w, q))
            continue; /* ponytail: >63-char words only exist in a corrupt KB; skipped, not counted */
        memcpy(out[nout++], w, n + 1);
    }
    return nout;
}
