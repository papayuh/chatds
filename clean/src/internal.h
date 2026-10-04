/* SPDX-License-Identifier: MIT */
#ifndef CHATDS_INTERNAL_H
#define CHATDS_INTERNAL_H
#include "chatds_engine.h"
#include <limits.h>
#include <stdint.h>
#include <string.h>

#if CHAR_BIT != 8 || ((-1 >> 1) != -1)
#error "ChatDS requires 8-bit bytes and arithmetic signed right shift"
#endif
#ifdef ARM9
#include <nds/ndstypes.h>
#define CDS_HOT ITCM_CODE
#else
#define CDS_HOT
#endif

#define CDS_HEADER_BYTES 512u
#define CDS_MAX_MODEL_BYTES (64u * 1024u * 1024u)
#define CDS_MAX_SESSION_BYTES (128u * 1024u * 1024u)
#define CDS_MAX_WIDTH 8192u
#define CDS_MAX_LAYERS 128u
#define CDS_MAX_VOCAB 65536u
#define CDS_MAX_CONTEXT 4096u
#define CDS_TCM_BYTES (10u * 1024u)
#define CDS_ONE 65536

typedef struct {
    const int8_t *rows;
    uint32_t height, width, stride;
} cds_matrix;

struct chatds_model {
    chatds_model_config config;
    unsigned char *bytes;
    size_t file_bytes, allocation_bytes;
    cds_matrix embedding;
    cds_matrix *matrices; /* Per layer: Q K V O gate down up. */
    int32_t *norms;
    float *frequencies;
    uint32_t head_size, kv_dim;
    int32_t inv_sqrt_head;
    uint16_t exponent[1025];
    uint32_t sigmoid[2049];
};

struct chatds_session {
    const chatds_model *model;
    chatds_kernel kernel;
    uint32_t window, position;
    unsigned char *storage, *workspace;
    int owns_tcm;
    size_t heap_bytes, tcm_bytes, kv_bytes;
    uint64_t saturations;
    int16_t *keys, *values;
    int16_t *quantized;
    int32_t *x, *z, *q, *k, *v, *a, *o, *gate, *up;
    int32_t *logits, *scores, *probabilities, *cosines, *sines;
};

/* Unaligned-safe little-endian decoding. Compilers fold these into native loads
 * on supported little-endian targets; no aliasing or alignment assumptions. */
static inline uint16_t cds_u16(const void *p) {
    const unsigned char *b = p;
    return (uint16_t)((uint16_t)b[0] | (uint16_t)b[1] << 8);
}
static inline uint32_t cds_u32(const void *p) {
    const unsigned char *b = p;
    return (uint32_t)b[0] | (uint32_t)b[1] << 8 |
           (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
}
static inline int cds_add_size(size_t a, size_t b, size_t *out) {
    if (b > SIZE_MAX - a) return 0;
    *out = a + b;
    return 1;
}
static inline int cds_mul_size(size_t a, size_t b, size_t *out) {
    if (a && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}
static inline int32_t cds_clamp(int64_t value, uint64_t *saturations) {
    if (value > INT32_MAX) { ++*saturations; return INT32_MAX; }
    if (value < INT32_MIN) { ++*saturations; return INT32_MIN; }
    return (int32_t)value;
}
static inline int32_t cds_mul_q16(int32_t a, int32_t b, uint64_t *saturations) {
    return cds_clamp(((int64_t)a * b) >> 16, saturations);
}
static inline int16_t cds_clip_q8(int32_t value, uint64_t *saturations) {
    value >>= 8;
    if (value > INT16_MAX) { ++*saturations; return INT16_MAX; }
    if (value < INT16_MIN) { ++*saturations; return INT16_MIN; }
    return (int16_t)value;
}

/* Private kernel seam. The public test surface is session_forward; direct
 * randomized kernel tests additionally pin integer bounds and rounding. */
void cds_matvec_reference(int32_t *out, const cds_matrix *matrix,
                         const int32_t *input, uint32_t group_size,
                         int16_t *scratch, uint64_t *saturations);
void cds_matvec_fast(int32_t *out, const cds_matrix *matrix,
                    const int32_t *input, uint32_t group_size,
                    int16_t *scratch, uint64_t *saturations);
#endif
