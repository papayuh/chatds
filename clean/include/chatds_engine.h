/* SPDX-License-Identifier: MIT
 * Resident DSQ8 inference, independently implemented without upstream engine source.
 * This interface does not tokenize, retrieve, sample, render, or own a UI.
 */
#ifndef CHATDS_ENGINE_H
#define CHATDS_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct chatds_model chatds_model;
typedef struct chatds_session chatds_session;

typedef enum {
    CHATDS_OK = 0,
    CHATDS_INVALID_ARGUMENT,
    CHATDS_IO_ERROR,
    CHATDS_BAD_FORMAT,
    CHATDS_UNSUPPORTED,
    CHATDS_OUT_OF_MEMORY,
    CHATDS_CONTEXT_FULL
} chatds_status;

typedef enum {
    CHATDS_KERNEL_REFERENCE = 0,
    CHATDS_KERNEL_FAST = 1
} chatds_kernel;

typedef struct {
    uint32_t dim, hidden_dim, layers, heads, kv_heads;
    uint32_t vocab_size, context_length, group_size;
} chatds_model_config;

typedef struct {
    /* Zero uses the model's window. Must not exceed it. No sliding window. */
    uint32_t context_length;
    chatds_kernel kernel;
    /* Zero uses the implementation's safety ceiling. Counts session heap +
     * TCM storage, not shared model storage; see session_info below. */
    size_t max_memory_bytes;
} chatds_session_options;

typedef struct {
    /* Signed Q16.16 logits; borrowed until the next forward/reset/destroy.
     * Lowest token ID wins an exact tie. Caller handles BOS/EOS and budget. */
    const int32_t *logits_q16;
    uint32_t logits_count;
    uint32_t argmax;
} chatds_output;

typedef struct {
    uint32_t position, context_length;
    size_t heap_bytes, tcm_bytes, kv_bytes;
    /* Cumulative saturating arithmetic / KV clipping events since reset.
     * Nonzero is an accuracy warning, not memory corruption or an error. */
    uint64_t saturations;
} chatds_session_info;

/* Copies/owns the model bytes, never modifies the file/caller buffer. On failure
 * *out is NULL. Supports TIE1 v1 DSQ8, layer-major, shared classifier only.
 * All dimensions come from the file; supported bounds are in ARITHMETIC.md.
 * Model is immutable and may be shared by sessions; it MUST outlive them. */
chatds_status chatds_model_load(const char *path, chatds_model **out);
chatds_status chatds_model_from_memory(const void *bytes, size_t size,
                                       chatds_model **out);
const chatds_model_config *chatds_model_get_config(const chatds_model *model);
size_t chatds_model_memory_bytes(const chatds_model *model);
void chatds_model_destroy(chatds_model *model);

/* NULL options: FAST, full model window, default safety ceiling.
 * No inference-time allocation. FAST and REFERENCE use identical arithmetic.
 * On ARM9 built with DTCM_POOL=1, one fitting FAST session can own bounded DTCM
 * scratch; other sessions fall back to heap. Calls/creation/destruction must be serialized on ARM9.
 * Separate host sessions may execute independently; a session is not reentrant.
 */
chatds_status chatds_session_create(const chatds_model *model,
                                   const chatds_session_options *options,
                                   chatds_session **out);
void chatds_session_destroy(chatds_session *session);
void chatds_session_reset(chatds_session *session);
void chatds_session_get_info(const chatds_session *session,
                             chatds_session_info *out);

/* Consumes one token at the current position and advances by one on success.
 * out==NULL skips the classifier (prefill), but computes all layers and KV.
 * Invalid token/context-full calls leave session state and out unchanged.
 * Exact full-window input is allowed; the NEXT forward returns CONTEXT_FULL.
 * No callbacks, I/O, cancellation checks, or hidden token/stop rules here. */
chatds_status chatds_session_forward(chatds_session *session, uint32_t token,
                                    chatds_output *out);
const char *chatds_status_string(chatds_status status);

#ifdef __cplusplus
}
#endif
#endif
