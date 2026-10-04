/* SPDX-License-Identifier: MIT
 * Wall time for the UI; independent of rendering and generator steps.
 */
#ifndef __NDS__
#define _POSIX_C_SOURCE 200809L
#endif
#include "chatds_ui.h"

#ifdef __NDS__
#include <nds.h>

uint32_t chatds_ui_millis(void *ctx)
{
    static int started;
    static uint32_t previous;
    static uint64_t ticks;
    (void)ctx;
    if (!started) {
        /* Reserve timers 0/1: a free-running 32-bit counter at bus / 1024. */
        TIMER_CR(0) = TIMER_CR(1) = 0;
        TIMER_DATA(0) = TIMER_DATA(1) = 0;
        TIMER_CR(1) = TIMER_ENABLE | TIMER_CASCADE;
        TIMER_CR(0) = TIMER_ENABLE | TIMER_DIV_1024;
        started = 1;
    }
    uint16_t hi, lo, again;
    do {
        hi = TIMER_DATA(1);
        lo = TIMER_DATA(0);
        again = TIMER_DATA(1);
    } while (hi != again);
    uint32_t now = ((uint32_t)hi << 16) | lo;
    ticks += (uint32_t)(now - previous);
    previous = now;
    return (uint32_t)(ticks * 1024 * 1000 / BUS_CLOCK);
}
#else
#include <time.h>
#include <stdlib.h>

uint32_t chatds_ui_millis(void *ctx)
{
    struct timespec t;
    (void)ctx;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) abort();
    return (uint32_t)((uint64_t)t.tv_sec * 1000 + (uint32_t)t.tv_nsec / 1000000);
}
#endif
