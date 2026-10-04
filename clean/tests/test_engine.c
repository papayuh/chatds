/* SPDX-License-Identifier: MIT
 * Public-interface regression tests. Fixtures are synthetic exported data, not
 * vendored model/engine code, and do not need upstream source or private model assets.
 */
#include "chatds_engine.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { unsigned char *bytes; size_t size, embedding; } fixture;
static uint32_t rng = 0x192512;
static uint32_t random32(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static void put16(unsigned char *p, uint16_t v) { p[0] = v; p[1] = v >> 8; }
static void put32(unsigned char *p, uint32_t v) {
    for (unsigned i = 0; i < 4; ++i) p[i] = (unsigned char)(v >> (8 * i));
}
static size_t aligned(size_t n) { return (n + 511) & ~(size_t)511; }
static size_t matrix_size(uint32_t rows, uint32_t cols, uint32_t gs) {
    return (size_t)rows * (cols + cols / gs * 2 + 4);
}
static void fill_matrix(unsigned char *b, size_t *offset, uint32_t rows, uint32_t cols, uint32_t gs) {
    size_t stride = cols + cols / gs * 2 + 4;
    for (uint32_t r = 0; r < rows; ++r) {
        unsigned char *w = b + *offset + r * stride;
        for (uint32_t c = 0; c < cols; ++c) w[c] = (unsigned char)((int)(random32() % 15) - 7);
        for (uint32_t g = 0; g < cols / gs; ++g) put16(w + cols + g * 2, 32767);
        put32(w + cols + cols / gs * 2, 16384);
    }
    *offset += rows * stride;
}
static fixture make_fixture(chatds_model_config c) {
    uint32_t kv = c.dim / c.heads * c.kv_heads;
    size_t embedding = aligned(512 + (size_t)(2 * c.layers + 1) * c.dim * 4);
    size_t layer = matrix_size(c.dim, c.dim, c.group_size) * 2 +
        matrix_size(kv, c.dim, c.group_size) * 2 +
        matrix_size(c.hidden_dim, c.dim, c.group_size) * 2 +
        matrix_size(c.dim, c.hidden_dim, c.group_size);
    size_t size = embedding + matrix_size(c.vocab_size, c.dim, c.group_size);
    for (uint32_t l = 0; l < c.layers; ++l) size = aligned(size) + layer;
    unsigned char *b = calloc(1, size + 1); assert(b);
    uint32_t header[] = {0x31454954, 1, 4, 512, c.dim, c.hidden_dim, c.layers,
        c.heads, c.kv_heads, c.vocab_size, c.context_length, c.group_size, 257};
    for (unsigned i = 0; i < sizeof(header) / sizeof(*header); ++i) put32(b + i * 4, header[i]);
    for (size_t i = 0; i < (size_t)(2 * c.layers + 1) * c.dim; ++i) put32(b + 512 + i * 4, 0x3f800000);
    size_t offset = embedding;
    fill_matrix(b, &offset, c.vocab_size, c.dim, c.group_size);
    for (uint32_t l = 0; l < c.layers; ++l) {
        offset = aligned(offset);
        fill_matrix(b, &offset, c.dim, c.dim, c.group_size);
        fill_matrix(b, &offset, kv, c.dim, c.group_size);
        fill_matrix(b, &offset, kv, c.dim, c.group_size);
        fill_matrix(b, &offset, c.dim, c.dim, c.group_size);
        fill_matrix(b, &offset, c.hidden_dim, c.dim, c.group_size);
        fill_matrix(b, &offset, c.dim, c.hidden_dim, c.group_size);
        fill_matrix(b, &offset, c.hidden_dim, c.dim, c.group_size);
    }
    assert(offset == size);
    fixture result = {b, size, embedding}; return result;
}
static void equal_output(chatds_output a, chatds_output b) {
    assert(a.argmax == b.argmax && a.logits_count == b.logits_count);
    assert(!memcmp(a.logits_q16, b.logits_q16, (size_t)a.logits_count * 4));
}
static void test_sessions(chatds_model_config config) {
    fixture f = make_fixture(config);
    unsigned char *before = malloc(f.size); assert(before); memcpy(before, f.bytes, f.size);
    chatds_model *model = NULL;
    assert(chatds_model_from_memory(f.bytes, f.size, &model) == CHATDS_OK);
    assert(!memcmp(f.bytes, before, f.size)); free(before);
    assert(!memcmp(chatds_model_get_config(model), &config, sizeof(config)));
    assert(chatds_model_memory_bytes(model) > f.size);
    /* Model must own its bytes, not borrow the caller's scratch input. */
    memset(f.bytes, 0, f.size); free(f.bytes);
    chatds_session_options opts = {0, CHATDS_KERNEL_REFERENCE, 0};
    chatds_session *reference = NULL, *fast = NULL;
    assert(chatds_session_create(model, &opts, &reference) == CHATDS_OK);
    assert(chatds_session_create(model, NULL, &fast) == CHATDS_OK);
    chatds_session_info info; chatds_session_get_info(fast, &info);
    assert(info.kv_bytes == (size_t)config.layers * config.context_length * (config.dim / config.heads * config.kv_heads) * 4);
    assert(info.heap_bytes > info.kv_bytes && info.position == 0 && info.tcm_bytes == 0);
    int32_t *first = malloc((size_t)config.vocab_size * 4); assert(first);
    chatds_output a, b;
    for (uint32_t p = 0; p < config.context_length; ++p) {
        uint32_t token = (p * 3 + 1) % config.vocab_size;
        assert(chatds_session_forward(reference, token, &a) == CHATDS_OK);
        assert(chatds_session_forward(fast, token, &b) == CHATDS_OK);
        equal_output(a, b);
        if (!p) memcpy(first, b.logits_q16, (size_t)b.logits_count * 4);
    }
    chatds_session_get_info(fast, &info); assert(info.position == config.context_length);
    chatds_output saved = b;
    assert(chatds_session_forward(fast, 0, &b) == CHATDS_CONTEXT_FULL);
    assert(saved.logits_q16 == b.logits_q16 && saved.logits_count == b.logits_count && saved.argmax == b.argmax);
    chatds_session_info after; chatds_session_get_info(fast, &after);
    assert(!memcmp(&info, &after, sizeof(info)));
    chatds_session_reset(fast); chatds_session_reset(reference);
    assert(chatds_session_forward(fast, config.vocab_size, &b) == CHATDS_INVALID_ARGUMENT);
    chatds_session_get_info(fast, &info); assert(info.position == 0 && info.saturations == 0);
    assert(chatds_session_forward(fast, 1, &b) == CHATDS_OK);
    assert(!memcmp(first, b.logits_q16, (size_t)b.logits_count * 4));
    chatds_session_reset(fast);
    /* Classifier omission in prefill must not affect the subsequent logits. */
    for (uint32_t p = 0; p < config.context_length; ++p) {
        uint32_t token = (p * 3 + 1) % config.vocab_size;
        assert(chatds_session_forward(reference, token, &a) == CHATDS_OK);
        assert(chatds_session_forward(fast, token, p + 1 == config.context_length ? &b : NULL) == CHATDS_OK);
    }
    equal_output(a, b);
    chatds_session_destroy(fast); chatds_session_destroy(reference); free(first);
    opts.context_length = config.context_length + 1;
    assert(chatds_session_create(model, &opts, &fast) == CHATDS_INVALID_ARGUMENT && !fast);
    opts.context_length = 1; opts.max_memory_bytes = 1;
    assert(chatds_session_create(model, &opts, &fast) == CHATDS_OUT_OF_MEMORY && !fast);
    opts.max_memory_bytes = 0; opts.kernel = (chatds_kernel)99;
    assert(chatds_session_create(model, &opts, &fast) == CHATDS_INVALID_ARGUMENT && !fast);
    opts.kernel = CHATDS_KERNEL_FAST;
    assert(chatds_session_create(model, &opts, &fast) == CHATDS_OK);
    assert(chatds_session_create(model, NULL, &reference) == CHATDS_OK);
    assert(chatds_session_forward(fast, 1, &a) == CHATDS_OK);
    assert(chatds_session_forward(reference, 1, &b) == CHATDS_OK); equal_output(a, b);
    assert(chatds_session_forward(fast, 1, NULL) == CHATDS_CONTEXT_FULL);
    chatds_session_destroy(fast); chatds_session_destroy(reference); chatds_model_destroy(model);
}
static void reject_patch(fixture f, size_t offset, uint32_t value, chatds_status expected) {
    unsigned char saved[4]; memcpy(saved, f.bytes + offset, 4); put32(f.bytes + offset, value);
    chatds_model *model = NULL;
    assert(chatds_model_from_memory(f.bytes, f.size, &model) == expected && !model);
    memcpy(f.bytes + offset, saved, 4);
}
static void test_loader(const char *path) {
    chatds_model_config c = {32, 64, 2, 4, 2, 19, 8, 16};
    fixture f = make_fixture(c); chatds_model *m = NULL;
    assert(chatds_model_from_memory(f.bytes, 0, &m) == CHATDS_BAD_FORMAT && !m);
    assert(chatds_model_from_memory(f.bytes, 511, &m) == CHATDS_BAD_FORMAT && !m);
    assert(chatds_model_from_memory(f.bytes, f.size - 1, &m) == CHATDS_BAD_FORMAT && !m);
    assert(chatds_model_from_memory(f.bytes, f.size + 1, &m) == CHATDS_BAD_FORMAT && !m);
    reject_patch(f, 0, 0, CHATDS_BAD_FORMAT);
    reject_patch(f, 4, 2, CHATDS_UNSUPPORTED);
    reject_patch(f, 8, 5, CHATDS_UNSUPPORTED);
    reject_patch(f, 12, 256, CHATDS_UNSUPPORTED);
    reject_patch(f, 16, 0, CHATDS_BAD_FORMAT);
    reject_patch(f, 16, UINT32_MAX, CHATDS_UNSUPPORTED);
    reject_patch(f, 24, UINT32_MAX, CHATDS_UNSUPPORTED);
    reject_patch(f, 28, 0, CHATDS_BAD_FORMAT);
    reject_patch(f, 28, 3, CHATDS_BAD_FORMAT);
    reject_patch(f, 32, 3, CHATDS_BAD_FORMAT);
    reject_patch(f, 44, 0, CHATDS_BAD_FORMAT);
    reject_patch(f, 44, 7, CHATDS_BAD_FORMAT);
    reject_patch(f, 48, 256, CHATDS_UNSUPPORTED);
    reject_patch(f, 512, 0x7fc00000, CHATDS_BAD_FORMAT); /* NaN */
    reject_patch(f, 512, 0x7f800000, CHATDS_BAD_FORMAT); /* Infinity */
    reject_patch(f, 512, 0x47000000, CHATDS_BAD_FORMAT); /* +32768 */
    reject_patch(f, f.embedding + 32, 0x7fff8000, CHATDS_BAD_FORMAT); /* negative group scale */
    reject_patch(f, f.embedding + 36, UINT32_MAX, CHATDS_BAD_FORMAT);
    FILE *file = fopen(path, "wb"); assert(file);
    assert(fwrite(f.bytes, 1, f.size, file) == f.size); assert(!fclose(file));
    assert(chatds_model_load(path, &m) == CHATDS_OK);
    assert(chatds_model_get_config(m)->hidden_dim == 64); chatds_model_destroy(m);
    assert(!remove(path)); assert(chatds_model_load(path, &m) == CHATDS_IO_ERROR && !m);
    file = fopen(path, "wb"); assert(file); assert(fwrite(f.bytes, 1, 33, file) == 33); assert(!fclose(file));
    assert(chatds_model_load(path, &m) == CHATDS_BAD_FORMAT && !m); assert(!remove(path));
    /* All-zero embeddings/transformer rows: logits tie exactly; ID zero wins. */
    memset(f.bytes + f.embedding, 0, f.size - f.embedding);
    assert(chatds_model_from_memory(f.bytes, f.size, &m) == CHATDS_OK);
    chatds_session *s = NULL; assert(chatds_session_create(m, NULL, &s) == CHATDS_OK);
    chatds_output out; assert(chatds_session_forward(s, 1, &out) == CHATDS_OK);
    assert(out.argmax == 0); for (uint32_t i = 0; i < out.logits_count; ++i) assert(out.logits_q16[i] == 0);
    chatds_session_destroy(s); chatds_model_destroy(m); free(f.bytes);
    assert(chatds_model_load(NULL, &m) == CHATDS_INVALID_ARGUMENT && !m);
    assert(chatds_model_from_memory(NULL, 0, &m) == CHATDS_INVALID_ARGUMENT && !m);
    assert(chatds_model_from_memory(NULL, 0, NULL) == CHATDS_INVALID_ARGUMENT);
    assert(chatds_session_forward(NULL, 0, NULL) == CHATDS_INVALID_ARGUMENT);
    assert(!chatds_model_get_config(NULL) && !chatds_model_memory_bytes(NULL));
    chatds_session_destroy(NULL); chatds_model_destroy(NULL); chatds_session_reset(NULL);
    assert(!strcmp(chatds_status_string(CHATDS_CONTEXT_FULL), "context full"));
}
static void test_closed_form(void) {
    chatds_model_config c = {8, 8, 1, 4, 2, 3, 4, 8};
    fixture f = make_fixture(c);
    memset(f.bytes + f.embedding, 0, f.size - f.embedding);
    for (unsigned token = 1; token <= 2; ++token) {
        unsigned char *row = f.bytes + f.embedding + token * 14;
        memset(row, (int)token, 8); put16(row + 8, 32767); put32(row + 10, 1u << 24);
    }
    chatds_model *m = NULL; assert(chatds_model_from_memory(f.bytes, f.size, &m) == CHATDS_OK);
    for (int kernel = 0; kernel <= 1; ++kernel) {
        chatds_session_options options = {0, (chatds_kernel)kernel, 0};
        chatds_session *s = NULL; assert(chatds_session_create(m, &options, &s) == CHATDS_OK);
        chatds_output out; assert(chatds_session_forward(s, 1, &out) == CHATDS_OK);
        /* Zero transformer matrices leave eight embedding values at 65534.
         * RMS factor=65538 -> normalized value=65535 -> qx=16383, shift=2.
         * Group sums 131060 / 262120, then row scale and shift -> values below. */
        assert(out.logits_q16[0] == 0 && out.logits_q16[1] == 524240 && out.logits_q16[2] == 1048480);
        assert(out.argmax == 2);
        chatds_session_info info; chatds_session_get_info(s, &info); assert(info.saturations == 0);
        chatds_session_destroy(s);
    }
    chatds_model_destroy(m); free(f.bytes);
}
static void test_saturation(void) {
    chatds_model_config c = {32, 64, 2, 4, 2, 19, 8, 16};
    fixture f = make_fixture(c);
    for (size_t i = 0; i < (size_t)(2 * c.layers + 1) * c.dim; ++i)
        put32(f.bytes + 512 + i * 4, 0x46fffe00); /* finite +32767 norm */
    chatds_model *m = NULL; assert(chatds_model_from_memory(f.bytes, f.size, &m) == CHATDS_OK);
    chatds_session *fast = NULL, *ref = NULL;
    chatds_session_options options = {0, CHATDS_KERNEL_REFERENCE, 0};
    assert(chatds_session_create(m, NULL, &fast) == CHATDS_OK);
    assert(chatds_session_create(m, &options, &ref) == CHATDS_OK);
    for (uint32_t i = 0; i < c.context_length; ++i) {
        chatds_output a, b;
        assert(chatds_session_forward(fast, i, &a) == CHATDS_OK);
        assert(chatds_session_forward(ref, i, &b) == CHATDS_OK); equal_output(a, b);
    }
    chatds_session_info a, b; chatds_session_get_info(fast, &a); chatds_session_get_info(ref, &b);
    assert(a.saturations > 0 && a.saturations == b.saturations);
    chatds_session_reset(fast); chatds_session_get_info(fast, &a); assert(!a.saturations);
    chatds_session_destroy(fast); chatds_session_destroy(ref); chatds_model_destroy(m); free(f.bytes);
}
int main(int argc, char **argv) {
    assert(argc == 2);
    test_closed_form();
    test_saturation();
    chatds_model_config configs[] = {
        {24, 40, 2, 3, 1, 17, 9, 8}, /* Two-byte-aligned row scales, non-power-of-two shape. */
        {32, 64, 1, 4, 2, 19, 8, 16},
        {96, 128, 3, 6, 2, 31, 17, 32},
        {48, 96, 1, 6, 2, 17, 4, 24},
        {256, 512, 1, 8, 4, 11, 4, 256}
    };
    for (unsigned i = 0; i < sizeof(configs) / sizeof(*configs); ++i) test_sessions(configs[i]);
    test_loader(argv[1]);
    puts("PASS: generalized loader, file/memory ownership, malformed inputs, exact fast/scalar logits, reset, prefill, context and budgets");
    return 0;
}
