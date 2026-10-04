/* SPDX-License-Identifier: MIT
 * Stub of chatds_engine.h for host tests: replays a fixed answer and records
 * what the caller did. No model, no arithmetic. */
#ifndef STUB_ENGINE_H
#define STUB_ENGINE_H

#include "chatds_engine.h"

typedef struct {
    const uint32_t *answer; /* argmax returned by the 1st, 2nd... classified forward */
    uint32_t answer_len, eos;
    uint32_t context_length;
    /* observations */
    uint32_t forwards, prefill_forwards, classified, resets, pos;
    uint32_t last_token;
} stub_engine;

extern stub_engine g_stub;

#endif
