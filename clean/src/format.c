/* SPDX-License-Identifier: MIT
 * Layout inferred from allowed export data and Diego's format descriptions.
 * No unlicensed upstream parser or converter source was consulted.
 */
#include "internal.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static chatds_status layout(const unsigned char *header, size_t file_bytes,
                            chatds_model_config *c, size_t *expected) {
    if (file_bytes < CDS_HEADER_BYTES || cds_u32(header) != UINT32_C(0x31454954))
        return CHATDS_BAD_FORMAT;
    if (cds_u32(header + 4) != 1 || cds_u32(header + 8) != 4 ||
        cds_u32(header + 12) != CDS_HEADER_BYTES || cds_u32(header + 48) != 257)
        return CHATDS_UNSUPPORTED;
    c->dim = cds_u32(header + 16);
    c->hidden_dim = cds_u32(header + 20);
    c->layers = cds_u32(header + 24);
    c->heads = cds_u32(header + 28);
    c->kv_heads = cds_u32(header + 32);
    c->vocab_size = cds_u32(header + 36);
    c->context_length = cds_u32(header + 40);
    c->group_size = cds_u32(header + 44);
    if (!c->dim || !c->hidden_dim || !c->layers || !c->heads || !c->kv_heads ||
        !c->vocab_size || !c->context_length || !c->group_size)
        return CHATDS_BAD_FORMAT;
    if (c->dim > CDS_MAX_WIDTH || c->hidden_dim > CDS_MAX_WIDTH ||
        c->layers > CDS_MAX_LAYERS || c->vocab_size > CDS_MAX_VOCAB ||
        c->context_length > CDS_MAX_CONTEXT || c->group_size > 256 ||
        file_bytes > CDS_MAX_MODEL_BYTES)
        return CHATDS_UNSUPPORTED;
    if (c->dim % c->heads || c->heads % c->kv_heads ||
        (c->dim / c->heads) % 2 || c->group_size % 8 ||
        c->dim % c->group_size || c->hidden_dim % c->group_size)
        return CHATDS_BAD_FORMAT;

    size_t offset = CDS_HEADER_BYTES + (size_t)(2 * c->layers + 1) * c->dim * 4;
    offset = (offset + 511) & ~(size_t)511;
    size_t stride_d = c->dim + 2 * (c->dim / c->group_size) + 4;
    size_t stride_h = c->hidden_dim + 2 * (c->hidden_dim / c->group_size) + 4;
    size_t bytes, layer_bytes;
    if (!cds_mul_size(c->vocab_size, stride_d, &bytes) ||
        !cds_add_size(offset, bytes, &offset)) return CHATDS_BAD_FORMAT;
    uint32_t kv = (c->dim / c->heads) * c->kv_heads;
    size_t rows_d = (size_t)c->dim * 2 + kv * 2 + (size_t)c->hidden_dim * 2;
    if (!cds_mul_size(rows_d, stride_d, &layer_bytes) ||
        !cds_mul_size(c->dim, stride_h, &bytes) ||
        !cds_add_size(layer_bytes, bytes, &layer_bytes)) return CHATDS_BAD_FORMAT;
    for (uint32_t l = 0; l < c->layers; ++l) {
        if (!cds_add_size(offset, 511, &offset)) return CHATDS_BAD_FORMAT;
        offset &= ~(size_t)511;
        if (!cds_add_size(offset, layer_bytes, &offset)) return CHATDS_BAD_FORMAT;
    }
    if (offset != file_bytes) return CHATDS_BAD_FORMAT;
    *expected = offset;
    return CHATDS_OK;
}

static cds_matrix take_matrix(chatds_model *m, size_t *offset,
                              uint32_t rows, uint32_t cols) {
    cds_matrix result;
    result.rows = (const int8_t *)(m->bytes + *offset);
    result.height = rows;
    result.width = cols;
    result.stride = cols + 2 * (cols / m->config.group_size) + 4;
    *offset += (size_t)rows * result.stride;
    return result;
}

static int valid_scales(const cds_matrix *m, uint32_t gs) {
    for (uint32_t r = 0; r < m->height; ++r) {
        const int8_t *row = m->rows + (size_t)r * m->stride;
        for (uint32_t j = 0; j < m->width / gs; ++j)
            if (cds_u16(row + m->width + 2 * j) > 32767) return 0;
        if (cds_u32(row + m->width + 2 * (m->width / gs)) > INT32_MAX) return 0;
    }
    return 1;
}

/* Adopts bytes on every path, including failure. */
static chatds_status adopt(unsigned char *bytes, size_t size,
                           const chatds_model_config *config, chatds_model **out) {
    chatds_model *m = calloc(1, sizeof(*m));
    if (!m) { free(bytes); return CHATDS_OUT_OF_MEMORY; }
    m->bytes = bytes;
    m->file_bytes = size;
    m->config = *config;
    m->head_size = config->dim / config->heads;
    m->kv_dim = m->head_size * config->kv_heads;
    m->matrices = calloc((size_t)config->layers * 7, sizeof(*m->matrices));
    m->frequencies = malloc((size_t)(m->head_size / 2) * sizeof(float));
    if (!m->matrices || !m->frequencies) {
        chatds_model_destroy(m); return CHATDS_OUT_OF_MEMORY;
    }
    m->allocation_bytes = sizeof(*m) + size +
        (size_t)config->layers * 7 * sizeof(*m->matrices) +
        (size_t)(m->head_size / 2) * sizeof(float);
    size_t norm_count = (size_t)(2 * config->layers + 1) * config->dim;
    m->norms = (int32_t *)(bytes + CDS_HEADER_BYTES);
    for (size_t i = 0; i < norm_count; ++i) {
        uint32_t bits = cds_u32(bytes + CDS_HEADER_BYTES + i * 4);
        float weight;
        memcpy(&weight, &bits, sizeof(weight));
        double fixed = (double)weight * 65536.0;
        if (!isfinite(weight) || fixed > INT32_MAX || fixed < INT32_MIN) {
            chatds_model_destroy(m); return CHATDS_BAD_FORMAT;
        }
        double rounded = fixed < 0 ? ceil(fixed - 0.5) : floor(fixed + 0.5);
        if (rounded > INT32_MAX || rounded < INT32_MIN) {
            chatds_model_destroy(m); return CHATDS_BAD_FORMAT;
        }
        m->norms[i] = (int32_t)rounded;
    }
    size_t offset = (CDS_HEADER_BYTES + norm_count * 4 + 511) & ~(size_t)511;
    m->embedding = take_matrix(m, &offset, config->vocab_size, config->dim);
    for (uint32_t l = 0; l < config->layers; ++l) {
        offset = (offset + 511) & ~(size_t)511;
        cds_matrix *a = m->matrices + l * 7;
        a[0] = take_matrix(m, &offset, config->dim, config->dim);
        a[1] = take_matrix(m, &offset, m->kv_dim, config->dim);
        a[2] = take_matrix(m, &offset, m->kv_dim, config->dim);
        a[3] = take_matrix(m, &offset, config->dim, config->dim);
        a[4] = take_matrix(m, &offset, config->hidden_dim, config->dim);
        a[5] = take_matrix(m, &offset, config->dim, config->hidden_dim);
        a[6] = take_matrix(m, &offset, config->hidden_dim, config->dim);
    }
    if (offset != size || !valid_scales(&m->embedding, config->group_size)) {
        chatds_model_destroy(m); return CHATDS_BAD_FORMAT;
    }
    for (uint32_t i = 0; i < config->layers * 7; ++i) {
        if (!valid_scales(m->matrices + i, config->group_size)) {
            chatds_model_destroy(m); return CHATDS_BAD_FORMAT;
        }
    }
    m->inv_sqrt_head = (int32_t)roundf(65536.0f / sqrtf((float)m->head_size));
    for (uint32_t j = 0; j < m->head_size / 2; ++j)
        m->frequencies[j] = powf(10000.0f, (float)(2 * j) / m->head_size);
    for (int i = 0; i <= 1024; ++i)
        m->exponent[i] = (uint16_t)roundf(expf(-i / 64.0f) * 32768.0f);
    for (int i = 0; i <= 2048; ++i)
        m->sigmoid[i] = (uint32_t)roundf(65536.0f / (1 + expf(-(i - 1024) / 64.0f)));
    *out = m;
    return CHATDS_OK;
}

chatds_status chatds_model_from_memory(const void *bytes, size_t size,
                                       chatds_model **out) {
    if (!out) return CHATDS_INVALID_ARGUMENT;
    *out = NULL;
    if (!bytes) return CHATDS_INVALID_ARGUMENT;
    if (sizeof(float) != 4 || FLT_RADIX != 2 || FLT_MANT_DIG != 24)
        return CHATDS_UNSUPPORTED;
    chatds_model_config config;
    size_t expected;
    chatds_status status = layout(bytes, size, &config, &expected);
    if (status != CHATDS_OK) return status;
    unsigned char *owned = malloc(expected);
    if (!owned) return CHATDS_OUT_OF_MEMORY;
    memcpy(owned, bytes, expected);
    return adopt(owned, expected, &config, out);
}

chatds_status chatds_model_load(const char *path, chatds_model **out) {
    if (!out) return CHATDS_INVALID_ARGUMENT;
    *out = NULL;
    if (!path) return CHATDS_INVALID_ARGUMENT;
    if (sizeof(float) != 4 || FLT_RADIX != 2 || FLT_MANT_DIG != 24)
        return CHATDS_UNSUPPORTED;
    FILE *file = fopen(path, "rb");
    if (!file) return CHATDS_IO_ERROR;
    unsigned char header[CDS_HEADER_BYTES];
    chatds_status status = CHATDS_IO_ERROR;
    unsigned char *bytes = NULL;
    chatds_model_config config;
    size_t expected = 0;
    if (fread(header, 1, sizeof(header), file) != sizeof(header)) {
        status = ferror(file) ? CHATDS_IO_ERROR : CHATDS_BAD_FORMAT;
        goto finish;
    }
    if (fseek(file, 0, SEEK_END)) goto finish;
    long length = ftell(file);
    if (length < 0) goto finish;
    status = layout(header, (size_t)length, &config, &expected);
    if (status != CHATDS_OK) goto finish;
    bytes = malloc(expected);
    if (!bytes) { status = CHATDS_OUT_OF_MEMORY; goto finish; }
    if (fseek(file, 0, SEEK_SET) || fread(bytes, 1, expected, file) != expected) {
        status = CHATDS_IO_ERROR; goto finish;
    }
    status = memcmp(bytes, header, sizeof(header)) ? CHATDS_BAD_FORMAT : CHATDS_OK;
finish:
    if (fclose(file) && status == CHATDS_OK) status = CHATDS_IO_ERROR;
    if (status != CHATDS_OK) { free(bytes); return status; }
    return adopt(bytes, expected, &config, out);
}

const chatds_model_config *chatds_model_get_config(const chatds_model *m) {
    return m ? &m->config : NULL;
}
size_t chatds_model_memory_bytes(const chatds_model *m) {
    return m ? m->allocation_bytes : 0;
}
void chatds_model_destroy(chatds_model *m) {
    if (!m) return;
    free(m->bytes); free(m->matrices); free(m->frequencies); free(m);
}
const char *chatds_status_string(chatds_status status) {
    switch (status) {
        case CHATDS_OK: return "OK";
        case CHATDS_INVALID_ARGUMENT: return "invalid argument";
        case CHATDS_IO_ERROR: return "I/O error";
        case CHATDS_BAD_FORMAT: return "malformed DSQ8 model";
        case CHATDS_UNSUPPORTED: return "unsupported format or dimensions";
        case CHATDS_OUT_OF_MEMORY: return "memory budget/allocation exceeded";
        case CHATDS_CONTEXT_FULL: return "context full";
    }
    return "unknown status";
}
