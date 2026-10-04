/* SPDX-License-Identifier: MIT */
/* Host driver for test_tok.py. tok_host TOK.BIN, then one request per stdin
 * line, one reply per stdout line:
 *   E <bos> <max> <hex text>  ->  <ret> <ids...>
 *   D <cap> <ids...>          ->  <ret> <hex out>
 *   P <id>                    ->  <hex piece>  (or "null")
 * Exit 2 if tok.bin does not load. */
#define _POSIX_C_SOURCE 200809L
#include "tok.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void hex(const char *p, int n)
{
    int i;
    for (i = 0; i < n; i++)
        printf("%02x", (unsigned char)p[i]);
}

int main(int argc, char **argv)
{
    tok_t t;
    char *line = NULL;
    size_t lcap = 0;

    if (argc != 2 || tok_load(&t, argv[1])) {
        fprintf(stderr, "tok_host: cannot load %s\n", argc > 1 ? argv[1] : "(none)");
        return 2;
    }
    while (getline(&line, &lcap, stdin) > 0) {
        char *s = line + 2;
        int i, n = 0;
        if (line[0] == 'E') {
            int bos = strtol(s, &s, 10), max = strtol(s, &s, 10), ret, *ids;
            char *text = malloc(strlen(s) / 2 + 1);
            unsigned b;
            while (*s == ' ')
                s++;
            while (sscanf(s, "%2x", &b) == 1)
                text[n++] = b, s += 2;
            ids = malloc((max + 1) * sizeof *ids);
            ret = tok_encode(&t, text, n, bos, ids, max);
            printf("%d", ret);
            for (i = 0; i < ret && i < max; i++)
                printf(" %d", ids[i]);
            free(text);
            free(ids);
        } else if (line[0] == 'D') {
            int cap = strtol(s, &s, 10), *ids = malloc(strlen(s) * sizeof *ids), ret;
            char *end, *out = malloc(cap + 1);
            for (;;) {
                long v = strtol(s, &end, 10);
                if (end == s)
                    break;
                ids[n++] = v, s = end;
            }
            ret = tok_decode(&t, ids, n, out, cap);
            printf("%d ", ret);
            if (ret >= 0)
                hex(out, ret < cap ? ret : cap - 1);
            free(ids);
            free(out);
        } else if (line[0] == 'P') {
            const char *p = tok_piece(&t, atoi(s), &n);
            if (p)
                hex(p, n);
            else
                printf("null");
        }
        printf("\n");
    }
    free(line);
    tok_free(&t);
    return 0;
}
