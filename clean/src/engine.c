/* SPDX-License-Identifier: MIT */
#include "internal.h"
#include <math.h>
#include <stdlib.h>

#ifdef CDS_DTCM_POOL
/* Separate typed regions avoid aliasing an int32 object as int16 scratch. */
#define TCM_QUANTIZED_CAPACITY 768u
#define TCM_ACT_CAPACITY ((CDS_TCM_BYTES - TCM_QUANTIZED_CAPACITY * 2) / 4)
static DTCM_DATA struct {
    int32_t activations[TCM_ACT_CAPACITY];
    int16_t quantized[TCM_QUANTIZED_CAPACITY];
} tcm_workspace;
static int tcm_owned; /* ARM9 lifetime operations are explicitly serialized. */
#endif

chatds_status chatds_session_create(const chatds_model *m,
                                   const chatds_session_options *options,
                                   chatds_session **out) {
    if (!out) return CHATDS_INVALID_ARGUMENT;
    *out = NULL;
    if (!m) return CHATDS_INVALID_ARGUMENT;
    uint32_t window = options && options->context_length ? options->context_length : m->config.context_length;
    chatds_kernel kernel = options ? options->kernel : CHATDS_KERNEL_FAST;
    size_t limit = options && options->max_memory_bytes ? options->max_memory_bytes : CDS_MAX_SESSION_BYTES;
    if (!window || window > m->config.context_length ||
        (kernel != CHATDS_KERNEL_REFERENCE && kernel != CHATDS_KERNEL_FAST))
        return CHATDS_INVALID_ARGUMENT;
    const chatds_model_config *c = &m->config;
    size_t kv_elements, kv_bytes;
    if (!cds_mul_size((size_t)c->layers * m->kv_dim, window, &kv_elements) ||
        !cds_mul_size(kv_elements, 2 * sizeof(int16_t), &kv_bytes))
        return CHATDS_OUT_OF_MEMORY;
    uint32_t width = c->dim > c->hidden_dim ? c->dim : c->hidden_dim;
    size_t work_words = (size_t)c->dim * 5 + m->kv_dim * 2 + (size_t)c->hidden_dim * 2;
    size_t work_bytes = work_words * 4 + (size_t)width * 2;
    work_bytes = (work_bytes + 7) & ~(size_t)7;
    size_t extra_bytes = ((size_t)c->vocab_size + (size_t)window * 2 + m->head_size) * 4;
    size_t storage_bytes, total_bytes;
    if (!cds_add_size(kv_bytes, extra_bytes, &storage_bytes) ||
        !cds_add_size(storage_bytes, 7, &storage_bytes)) return CHATDS_OUT_OF_MEMORY;
    storage_bytes &= ~(size_t)7;
    if (!cds_add_size(storage_bytes, work_bytes, &total_bytes) ||
        !cds_add_size(total_bytes, sizeof(chatds_session), &total_bytes) ||
        total_bytes > limit || total_bytes > CDS_MAX_SESSION_BYTES)
        return CHATDS_OUT_OF_MEMORY;

    int use_tcm = 0;
#ifdef CDS_DTCM_POOL
    use_tcm = kernel == CHATDS_KERNEL_FAST && work_words <= TCM_ACT_CAPACITY &&
              width <= TCM_QUANTIZED_CAPACITY && !tcm_owned;
#endif
    chatds_session *s = calloc(1, sizeof(*s));
    if (!s) return CHATDS_OUT_OF_MEMORY;
    size_t heap_storage = storage_bytes + (use_tcm ? 0 : work_bytes);
    s->storage = calloc(1, heap_storage);
    if (!s->storage) { free(s); return CHATDS_OUT_OF_MEMORY; }
    s->model = m; s->kernel = kernel; s->window = window;
    s->kv_bytes = kv_bytes; s->heap_bytes = heap_storage + sizeof(*s);
    s->workspace = s->storage + storage_bytes;
#ifdef CDS_DTCM_POOL
    if (use_tcm) {
        tcm_owned = 1; s->owns_tcm = 1; s->tcm_bytes = work_bytes;
        s->workspace = (unsigned char *)tcm_workspace.activations;
        memset(&tcm_workspace, 0, sizeof(tcm_workspace));
    }
#endif
    s->keys = (int16_t *)s->storage;
    s->values = s->keys + kv_elements;
    s->logits = (int32_t *)(s->values + kv_elements);
    s->scores = s->logits + c->vocab_size;
    s->probabilities = s->scores + window;
    s->cosines = s->probabilities + window;
    s->sines = s->cosines + m->head_size / 2;
    s->x = (int32_t *)s->workspace;
    s->z = s->x + c->dim;
    s->q = s->z + c->dim;
    s->a = s->q + c->dim;
    s->o = s->a + c->dim;
    s->k = s->o + c->dim;
    s->v = s->k + m->kv_dim;
    s->gate = s->v + m->kv_dim;
    s->up = s->gate + c->hidden_dim;
    s->quantized = (int16_t *)(s->up + c->hidden_dim);
#ifdef CDS_DTCM_POOL
    if (use_tcm) s->quantized = tcm_workspace.quantized;
#endif
    *out = s;
    return CHATDS_OK;
}

void chatds_session_destroy(chatds_session *s) {
    if (!s) return;
#ifdef CDS_DTCM_POOL
    if (s->owns_tcm) { memset(&tcm_workspace, 0, sizeof(tcm_workspace)); tcm_owned = 0; }
#endif
    free(s->storage); free(s);
}
void chatds_session_reset(chatds_session *s) {
    if (!s) return;
    memset(s->keys, 0, s->kv_bytes);
    memset(s->logits, 0, (size_t)s->model->config.vocab_size * sizeof(int32_t));
    s->position = 0; s->saturations = 0;
}
void chatds_session_get_info(const chatds_session *s, chatds_session_info *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!s) return;
    out->position = s->position; out->context_length = s->window;
    out->heap_bytes = s->heap_bytes; out->tcm_bytes = s->tcm_bytes;
    out->kv_bytes = s->kv_bytes; out->saturations = s->saturations;
}

static void matvec(chatds_session *s, int32_t *out, const cds_matrix *m, const int32_t *in) {
    if (s->kernel == CHATDS_KERNEL_FAST)
        cds_matvec_fast(out, m, in, s->model->config.group_size, s->quantized, &s->saturations);
    else
        cds_matvec_reference(out, m, in, s->model->config.group_size, s->quantized, &s->saturations);
}

static void rms(chatds_session *s, int32_t *out, const int32_t *in, const int32_t *w) {
    uint32_t d = s->model->config.dim, maximum = 0;
    for (uint32_t i = 0; i < d; ++i) {
        uint32_t a = in[i] < 0 ? (uint32_t)(-(int64_t)in[i]) : (uint32_t)in[i];
        if (a > maximum) maximum = a;
    }
    unsigned shift = 0;
    while (maximum > UINT32_C(0xffffff)) { maximum >>= 1; ++shift; }
    /* Includes the extra unit from flooring a negative shifted input:
     * <=8192 * (2^24)^2 = 2^61, hence no signed overflow. */
    int64_t sum = 0;
    for (uint32_t i = 0; i < d; ++i) {
        int32_t scaled = in[i] >> shift;
        sum += (int64_t)scaled * scaled;
    }
    float mean = ldexpf((float)sum / d, (int)(2 * shift) - 32) + 1e-5f;
    int32_t factor = (int32_t)roundf(65536.0f / sqrtf(mean));
    for (uint32_t i = 0; i < d; ++i)
        out[i] = cds_mul_q16(cds_mul_q16(in[i], factor, &s->saturations), w[i], &s->saturations);
}

static void embedding(chatds_session *s, uint32_t token) {
    const chatds_model *m = s->model;
    uint32_t d = m->config.dim, gs = m->config.group_size;
    const int8_t *w = m->embedding.rows + (size_t)token * m->embedding.stride;
    uint32_t row_scale = cds_u32(w + d + 2 * (d / gs));
    for (uint32_t i = 0; i < d; ++i)
        s->x[i] = cds_clamp(((int64_t)w[i] * cds_u16(w + d + 2 * (i / gs)) * row_scale) >> 23,
                             &s->saturations);
}

static void rope(chatds_session *s, int32_t *values, uint32_t count) {
    for (uint32_t i = 0; i < count; i += 2) {
        uint32_t j = (i % s->model->head_size) / 2;
        int32_t x = values[i], y = values[i + 1];
        int32_t xc = cds_mul_q16(x, s->cosines[j], &s->saturations);
        int32_t ys = cds_mul_q16(y, s->sines[j], &s->saturations);
        int32_t xs = cds_mul_q16(x, s->sines[j], &s->saturations);
        int32_t yc = cds_mul_q16(y, s->cosines[j], &s->saturations);
        values[i] = cds_clamp((int64_t)xc - ys, &s->saturations);
        values[i + 1] = cds_clamp((int64_t)xs + yc, &s->saturations);
    }
}

static CDS_HOT void attention(chatds_session *s, uint32_t layer) {
    const chatds_model *m = s->model;
    uint32_t pos = s->position, kv = m->kv_dim, hs = m->head_size;
    int16_t *keys = s->keys + (size_t)layer * s->window * kv;
    int16_t *values = s->values + (size_t)layer * s->window * kv;
    for (uint32_t i = 0; i < kv; ++i) {
        keys[(size_t)pos * kv + i] = cds_clip_q8(s->k[i], &s->saturations);
        values[(size_t)pos * kv + i] = cds_clip_q8(s->v[i], &s->saturations);
    }
    uint32_t sharing = m->config.heads / m->config.kv_heads;
    for (uint32_t h = 0; h < m->config.heads; ++h) {
        uint32_t kh = (h / sharing) * hs;
        for (uint32_t i = 0; i < hs; ++i)
            s->quantized[i] = cds_clip_q8(s->q[h * hs + i], &s->saturations);
        int32_t maximum = INT32_MIN;
        for (uint32_t t = 0; t <= pos; ++t) {
            int64_t sum = 0;
            for (uint32_t i = 0; i < hs; ++i)
                sum += (int32_t)s->quantized[i] * keys[(size_t)t * kv + kh + i];
            s->scores[t] = (int32_t)((sum * m->inv_sqrt_head) >> 26);
            if (s->scores[t] > maximum) maximum = s->scores[t];
        }
        uint32_t denominator = 0;
        for (uint32_t t = 0; t <= pos; ++t) {
            int64_t delta = (int64_t)maximum - s->scores[t];
            s->probabilities[t] = m->exponent[delta > 1024 ? 1024 : (uint32_t)delta];
            denominator += (uint32_t)s->probabilities[t];
        }
        uint32_t reciprocal = (uint32_t)((UINT64_C(1) << 31) / denominator);
        for (uint32_t t = 0; t <= pos; ++t)
            s->probabilities[t] = (int32_t)(((uint64_t)s->probabilities[t] * reciprocal) >> 16);
        for (uint32_t i = 0; i < hs; ++i) {
            int32_t sum = 0;
            for (uint32_t t = 0; t <= pos; ++t)
                sum += s->probabilities[t] * values[(size_t)t * kv + kh + i];
            s->a[h * hs + i] = sum >> 7;
        }
    }
}

chatds_status chatds_session_forward(chatds_session *s, uint32_t token,
                                    chatds_output *out) {
    if (!s || token >= s->model->config.vocab_size) return CHATDS_INVALID_ARGUMENT;
    if (s->position >= s->window) return CHATDS_CONTEXT_FULL;
    const chatds_model *m = s->model;
    uint32_t d = m->config.dim, hidden = m->config.hidden_dim;
    embedding(s, token);
    for (uint32_t j = 0; j < m->head_size / 2; ++j) {
        float angle = s->position / m->frequencies[j];
        s->cosines[j] = (int32_t)roundf(cosf(angle) * 65536.0f);
        s->sines[j] = (int32_t)roundf(sinf(angle) * 65536.0f);
    }
    for (uint32_t l = 0; l < m->config.layers; ++l) {
        const cds_matrix *a = m->matrices + l * 7;
        rms(s, s->z, s->x, m->norms + (size_t)l * d);
        matvec(s, s->q, a, s->z);
        matvec(s, s->k, a + 1, s->z);
        matvec(s, s->v, a + 2, s->z);
        rope(s, s->q, d); rope(s, s->k, m->kv_dim);
        attention(s, l);
        matvec(s, s->o, a + 3, s->a);
        for (uint32_t i = 0; i < d; ++i) s->x[i] = cds_clamp((int64_t)s->x[i] + s->o[i], &s->saturations);
        rms(s, s->z, s->x, m->norms + (size_t)(m->config.layers + l) * d);
        matvec(s, s->gate, a + 4, s->z);
        matvec(s, s->up, a + 6, s->z);
        for (uint32_t i = 0; i < hidden; ++i) {
            int32_t index = (s->gate[i] >> 10) + 1024;
            if (index < 0) index = 0;
            if (index > 2048) index = 2048;
            s->gate[i] = cds_mul_q16(cds_mul_q16(s->gate[i], (int32_t)m->sigmoid[index], &s->saturations),
                                     s->up[i], &s->saturations);
        }
        matvec(s, s->o, a + 5, s->gate);
        for (uint32_t i = 0; i < d; ++i) s->x[i] = cds_clamp((int64_t)s->x[i] + s->o[i], &s->saturations);
    }
    if (out) {
        rms(s, s->z, s->x, m->norms + (size_t)(2 * m->config.layers) * d);
        matvec(s, s->logits, &m->embedding, s->z);
        uint32_t best = 0;
        for (uint32_t i = 1; i < m->config.vocab_size; ++i)
            if (s->logits[i] > s->logits[best]) best = i;
        out->logits_q16 = s->logits;
        out->logits_count = m->config.vocab_size;
        out->argmax = best;
    }
    ++s->position;
    return CHATDS_OK;
}
