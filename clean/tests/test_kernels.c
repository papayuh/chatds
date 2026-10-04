/* SPDX-License-Identifier: MIT */
#include "internal.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static uint32_t state = UINT32_C(0x31514954);
static uint32_t random32(void) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state;
}
static void put16(unsigned char *p, uint16_t x) { p[0] = x; p[1] = x >> 8; }
static void put32(unsigned char *p, uint32_t x) {
    for (unsigned i = 0; i < 4; ++i) p[i] = (unsigned char)(x >> (i * 8));
}
/* Test oracle intentionally uses division/remainder rather than signed shifts,
 * and 128-bit arithmetic rather than sharing overflow checks with the engine. */
static __int128 floor_div(__int128 n, uint64_t denominator) {
    __int128 q = n / denominator;
    return q - (n < 0 && n % denominator != 0);
}
static int32_t oracle(const int8_t *w, const int32_t *x, unsigned width,
                      unsigned gs, unsigned shift) {
    __int128 sum = 0;
    for (unsigned g = 0; g < width / gs; ++g) {
        __int128 dot = 0;
        for (unsigned i = 0; i < gs; ++i)
            dot += w[g * gs + i] * floor_div(x[g * gs + i], UINT64_C(1) << shift);
        sum += floor_div(dot * cds_u16(w + width + g * 2), 32768);
    }
    __int128 value = floor_div(sum * cds_u32(w + width + 2 * (width / gs)), UINT64_C(1) << (24 - shift));
    return value > INT32_MAX ? INT32_MAX : value < INT32_MIN ? INT32_MIN : (int32_t)value;
}
int main(void) {
    unsigned widths[] = {8, 24, 192, 512, 1024, 1056, 4096, 8192};
    unsigned groups[] = {8, 16, 24, 32, 48, 64, 96, 128, 192, 256};
    size_t checked = 0;
    for (unsigned wi = 0; wi < sizeof(widths) / sizeof(*widths); ++wi) {
        unsigned width = widths[wi];
        for (unsigned gi = 0; gi < sizeof(groups) / sizeof(*groups); ++gi) {
            unsigned gs = groups[gi]; if (width % gs) continue;
            cds_matrix m = {NULL, 5, width, width + 2 * (width / gs) + 4};
            unsigned char *storage = malloc((size_t)m.height * m.stride);
            int32_t *x = malloc((size_t)width * sizeof(*x));
            int16_t *scratch = malloc((size_t)width * sizeof(*scratch));
            assert(storage && x && scratch); m.rows = (const int8_t *)storage;
            for (unsigned trial = 0; trial < 40; ++trial) {
                uint32_t maximum = 0;
                for (unsigned i = 0; i < width; ++i) {
                    uint32_t bits = random32(); memcpy(x + i, &bits, 4);
                    if (!trial) x[i] = 0;
                    if (trial == 1) x[i] = INT32_MIN;
                    if (trial == 2) x[i] = INT32_MAX;
                    if (trial == 3) x[i] = i & 1 ? INT32_MAX : INT32_MIN;
                    if (trial >= 4 && trial < 20) x[i] /= (1 << (trial - 3));
                    uint32_t a = x[i] < 0 ? (uint32_t)(-(int64_t)x[i]) : (uint32_t)x[i];
                    if (a > maximum) maximum = a;
                }
                unsigned shift = 0; while (maximum > 16383) { maximum /= 2; ++shift; }
                for (unsigned r = 0; r < m.height; ++r) {
                    unsigned char *w = storage + (size_t)r * m.stride;
                    for (unsigned i = 0; i < width; ++i) w[i] = trial < 4 ? 128 : (unsigned char)random32();
                    for (unsigned g = 0; g < width / gs; ++g) put16(w + width + 2 * g, trial < 4 ? 32767 : random32() & 32767);
                    put32(w + width + 2 * (width / gs), trial < 4 ? INT32_MAX : random32() & INT32_MAX);
                }
                int32_t fast[5], reference[5]; uint64_t fs = 0, rs = 0;
                cds_matvec_fast(fast, &m, x, gs, scratch, &fs);
                cds_matvec_reference(reference, &m, x, gs, scratch, &rs);
                assert(fs == rs);
                for (unsigned r = 0; r < m.height; ++r) {
                    assert(fast[r] == reference[r]);
                    assert(fast[r] == oracle(m.rows + (size_t)r * m.stride, x, width, gs, shift));
                    ++checked;
                }
            }
            free(storage); free(x); free(scratch);
        }
    }
    printf("PASS: %zu kernel rows, fast == scalar == independent int128 oracle\n", checked);
    return 0;
}
