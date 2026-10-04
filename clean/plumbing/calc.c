/* SPDX-License-Identifier: MIT
 * calc(): integer fixed point, scale 10^4, int64; twin of apply_calc in corpus/chatds_runtime.py
 * (grammar, rounding and overflow rules are documented there). No 128-bit types: the ARM9 has none. */
#include <stdio.h>
#include <string.h>
#include "plumbing.h"

#define S 10000
#define MAXV INT64_MAX
#define CALC_DEPTH 8

typedef struct {
    const char *e;
    size_t n, i;
    int bad, ovf;
} calc_t;

static char peek(calc_t *c) {
    while (c->i < c->n && c->e[c->i] == ' ')
        c->i++;
    return c->i < c->n ? c->e[c->i] : 0;
}

/* round half away from zero; n != INT64_MIN, d != 0 */
static int64_t rdiv(int64_t n, int64_t d) {
    uint64_t an = n < 0 ? -(uint64_t)n : (uint64_t)n, ad = d < 0 ? -(uint64_t)d : (uint64_t)d;
    uint64_t q = an / ad, r = an % ad;
    q += 2 * r >= ad; /* 2r < 2^64: r < ad <= 2^63 */
    return (n < 0) != (d < 0) ? -(int64_t)q : (int64_t)q;
}

/* |v| <= MAXV, else overflow and 0. INT64_MIN is out of range too. */
static int64_t ok(calc_t *c, int overflowed, int64_t v) {
    if (overflowed || v == INT64_MIN) {
        c->ovf = 1;
        return 0;
    }
    return v;
}

static int64_t expr(calc_t *c, int depth);

static int64_t unary(calc_t *c, int depth) {
    char ch = peek(c);
    if (depth > CALC_DEPTH) {
        c->bad = 1;
        return 0;
    }
    if (ch == '+' || ch == '-') {
        c->i++;
        int64_t v = unary(c, depth + 1);
        return ch == '-' ? -v : v;
    }
    if (ch == '(') {
        c->i++;
        int64_t v = expr(c, depth + 1);
        if (peek(c) != ')') {
            c->bad = 1;
            return 0;
        }
        c->i++;
        return v;
    }
    /* ([0-9]*)(\.([0-9]*))? */
    size_t i = c->i, ip = 0, fp = 0;
    uint64_t v = 0;
    int big = 0;
    for (; i < c->n && c->e[i] >= '0' && c->e[i] <= '9'; i++, ip++) {
        if (v > (uint64_t)MAXV / 10)
            big = 1;
        else
            v = v * 10 + (uint64_t)(c->e[i] - '0');
    }
    int64_t f = 0;
    if (i < c->n && c->e[i] == '.') {
        size_t start = ++i;
        for (; i < c->n && c->e[i] >= '0' && c->e[i] <= '9'; i++, fp++)
            if (fp < 4)
                f = f * 10 + (c->e[i] - '0');
        for (size_t k = fp; k < 4; k++)
            f *= 10;
        f += fp > 4 && c->e[start + 4] >= '5'; /* digits past the 4th round half up on the 5th */
    }
    if (!ip && !fp) {
        c->bad = 1;
        return 0;
    }
    c->i = i;
    if (big || v > (uint64_t)(MAXV - f) / S)
        return ok(c, 1, 0);
    return ok(c, 0, (int64_t)v * S + f);
}

static int64_t term(calc_t *c, int depth) {
    int64_t a = unary(c, depth);
    char o;
    while (!c->bad && ((o = peek(c)) == '*' || o == '/' || o == '%')) {
        c->i++;
        int64_t b = unary(c, depth), p;
        if (c->ovf)
            continue;
        if (o == '*') {
            int of = __builtin_mul_overflow(a, b, &p);
            a = rdiv(ok(c, of, p), S);
        } else if (b == 0) {
            c->ovf = 1;
        } else if (o == '/') {
            int of = __builtin_mul_overflow(a, (int64_t)S, &p);
            a = rdiv(ok(c, of, p), b);
        } else {
            uint64_t aa = a < 0 ? -(uint64_t)a : (uint64_t)a, ab = b < 0 ? -(uint64_t)b : (uint64_t)b;
            a = (a >= 0 ? 1 : -1) * (int64_t)(aa % ab);
        }
    }
    return a;
}

static int64_t expr(calc_t *c, int depth) {
    if (depth > CALC_DEPTH) {
        c->bad = 1;
        return 0;
    }
    int64_t a = term(c, depth), s;
    char o;
    while (!c->bad && ((o = peek(c)) == '+' || o == '-')) {
        c->i++;
        int64_t b = term(c, depth);
        int of = o == '+' ? __builtin_add_overflow(a, b, &s) : __builtin_sub_overflow(a, b, &s);
        a = ok(c, of, s);
    }
    return a;
}

static int fmt(int64_t v, char *out) {
    uint64_t av = v < 0 ? -(uint64_t)v : (uint64_t)v;
    if (av % S == 0)
        return sprintf(out, "%s%llu", v < 0 ? "-" : "", (unsigned long long)(av / S));
    int n = sprintf(out, "%s%llu.%04u", v < 0 ? "-" : "", (unsigned long long)(av / S), (unsigned)(av % S));
    while (out[n - 1] == '0')
        out[--n] = 0;
    return n;
}

int cds_apply_calc(const char *output, char *out, size_t cap) {
    static const char ws[] = " \t\n\r\v\f";
    const char *s = output, *e = output + strlen(output);
    while (s < e && strchr(ws, *s))
        s++;
    while (e > s && strchr(ws, e[-1]))
        e--;
    // ^calc\(([0-9 +\-*/().%]+)\)$
    if (e - s < 7 || strncmp(s, "calc(", 5) || e[-1] != ')')
        return 0;
    const char *x = s + 5, *y = e - 1;
    for (const char *p = x; p < y; p++)
        if (!strchr("0123456789 +-*/().%", *p))
            return 0;
    while (x < y && *x == ' ')
        x++;
    while (y > x && y[-1] == ' ')
        y--;
    calc_t c = {x, (size_t)(y - x), 0, 0, 0};
    int64_t v = expr(&c, 0);
    if (c.bad || peek(&c))
        return 0;
    char val[32];
    if (c.ovf)
        strcpy(val, "error");
    else
        fmt(v, val);
    size_t n = 5 + (size_t)(y - x) + 4 + strlen(val);
    if (n + 1 > cap)
        return 0;
    memcpy(out, "calc(", 5);
    memcpy(out + 5, x, (size_t)(y - x));
    memcpy(out + 5 + (y - x), ") = ", 4);
    strcpy(out + 9 + (y - x), val);
    return (int)n;
}
