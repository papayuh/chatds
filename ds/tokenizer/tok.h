/* SPDX-License-Identifier: MIT */
/* ChatDS tokenizer: SentencePiece-BPE compatible encode/decode over tok.bin.
 *
 * tok.bin (little-endian, written by tokbin.py):
 *   u32 max_len, then per piece id: f32 score, u32 len, len bytes.
 *   '▁' is stored as ' '; id 0 <unk>, 1 BOS, 2 EOS, 3..258 byte
 *   fallback "<0xXX>", 259.. normal pieces.
 *
 * Matches sentencepiece's encode/decode for models with an identity
 * normalizer, add_dummy_prefix, byte_fallback and the configured unk surface;
 * ds/tokenizer/test_tok.py checks that byte-for-byte. */
#ifndef CHATDS_TOK_H
#define CHATDS_TOK_H

#include <stddef.h>

#define TOK_UNK 0
#define TOK_BOS 1
#define TOK_EOS 2
#define TOK_BYTE0 3 /* id of byte 0x00; byte b is TOK_BYTE0 + b */

typedef struct {
    int n, max_len;
    unsigned char *buf;           /* the whole file; pieces point into it */
    const unsigned char **piece;
    unsigned short *len;
    float *score;
    unsigned short *sorted;       /* normal piece ids sorted by bytes */
    int n_sorted;
} tok_t;

/* 0 ok, -1 unreadable/malformed file or out of memory. */
int tok_load(tok_t *t, const char *path);
void tok_free(tok_t *t);

/* Encode len bytes of UTF-8 (invalid bytes become U+FFFD, as sentencepiece
 * does). Writes at most max ids, returns the full count (> max means the
 * output was cut), -1 out of memory. */
int tok_encode(const tok_t *t, const char *text, size_t len, int bos,
               int *ids, int max);

/* Raw bytes of one id, for streaming output: a byte id gives that byte,
 * BOS/EOS give "", <unk> gives sentencepiece's unk surface. NULL if the id
 * is out of range. */
const char *tok_piece(const tok_t *t, int id, int *len);

/* sentencepiece decode: first piece's leading space dropped, BOS/EOS
 * skipped, byte runs UTF-8 checked (each bad byte -> U+FFFD). snprintf-style:
 * writes at most cap-1 bytes plus a NUL, returns the full length, -1 bad id. */
int tok_decode(const tok_t *t, const int *ids, int n, char *out, int cap);

#endif
