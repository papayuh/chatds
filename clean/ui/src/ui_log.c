/* SPDX-License-Identifier: MIT
 * Chat history: one byte buffer of messages (role byte + text), plus a table
 * of wrapped lines that is rebuilt for the newest message as it streams in.
 */
#include <string.h>

#include "ui_internal.h"

size_t ui_wrap(const char *s, size_t n, size_t base, size_t cols, uint8_t flags,
               chatds_ui_line *out, size_t cap)
{
    size_t count = 0, pos = 0;
    uint8_t first = LINE_FIRST;
    for (;;) {
        size_t pe = pos;
        while (pe < n && s[pe] != '\n') pe++;
        size_t p = pos;
        int lead = 1; /* keep leading spaces of a paragraph, drop them after a wrap */
        do {
            if (!lead) {
                while (p < pe && s[p] == ' ') p++;
                if (p == pe) break; /* only trailing spaces were left */
            }
            lead = 0;
            size_t end;
            size_t next;
            if (pe - p <= cols) {
                end = pe;
                next = pe;
            } else {
                size_t q = p + cols; /* last candidate break: a space at p+cols fits */
                while (q > p && s[q] != ' ') q--;
                if (q > p) {
                    end = q;
                    next = q + 1;
                } else {
                    end = p + cols;
                    next = end;
                }
            }
            if (count == cap) return count;
            out[count].off = (uint16_t)(base + p);
            out[count].len = (uint8_t)(end - p);
            out[count].flags = (uint8_t)(flags | first);
            count++;
            first = 0;
            p = next;
        } while (p < pe);
        if (pe >= n) break;
        pos = pe + 1;
    }
    return count;
}

void ui_log_clear(chatds_ui_log *l)
{
    l->used = l->nlines = l->open_off = l->open_line = 0;
    l->has_open = 0;
}

static size_t msg_end(const chatds_ui_log *l, size_t off)
{
    size_t e = off + 1;
    while (e < l->used && (uint8_t)l->text[e] > CHATDS_ROLE_NOTE) e++;
    return e;
}

/* Re-wrap every message from byte offset `from` on; lines before it stay. */
static void rewrap_from(chatds_ui_log *l, size_t from, size_t line0)
{
    size_t off = from, n = line0;
    while (off < l->used) {
        size_t e = msg_end(l, off);
        n += ui_wrap(l->text + off + 1, e - off - 1, off + 1, TOP_COLS,
                     (uint8_t)l->text[off], l->lines + n, CHATDS_UI_LOG_LINES - n);
        off = e;
    }
    l->nlines = (uint16_t)n;
}

/* Drop the oldest whole message. Never drops the newest. */
static int evict_oldest(chatds_ui_log *l)
{
    size_t e = msg_end(l, 0);
    if (e >= l->used) return 0;
    memmove(l->text, l->text + e, l->used - e);
    l->used = (uint16_t)(l->used - e);
    l->open_off = (uint16_t)(l->open_off - e);
    rewrap_from(l, 0, 0);
    /* open_line = first line of the newest message */
    size_t i = l->nlines;
    while (i > 0 && !(l->lines[i - 1].flags & LINE_FIRST)) i--;
    l->open_line = (uint16_t)(i ? i - 1 : 0);
    return 1;
}

void ui_log_begin(chatds_ui_log *l, chatds_ui_role role)
{
    /* Keep room for the role byte plus the line for an empty message. */
    while ((l->used + 1 > CHATDS_UI_LOG_BYTES || l->nlines + 1 > CHATDS_UI_LOG_LINES) &&
           evict_oldest(l)) {}
    if (l->used + 1 > CHATDS_UI_LOG_BYTES) return; /* cannot happen: BYTES > 1 */
    l->open_off = l->used;
    l->open_line = l->nlines;
    l->text[l->used++] = (char)role;
    l->has_open = 1;
    rewrap_from(l, l->open_off, l->open_line);
}

size_t ui_log_msg_len(const chatds_ui_log *l)
{
    return l->has_open ? l->used - l->open_off - 1 : 0;
}

int ui_log_append(chatds_ui_log *l, const char *s, size_t n)
{
    if (!l->has_open) return 0;
    int before = l->nlines - l->open_line;
    for (size_t i = 0; i < n; i++) {
        unsigned char ch = (unsigned char)s[i];
        if (ch == '\r') continue;
        if (ch == '\t') ch = ' ';
        if (ch != '\n' && (ch < 32 || ch > 126)) ch = '?';
        /* Make room by dropping older history; the open message is never dropped. */
        while (l->used + 1 > CHATDS_UI_LOG_BYTES && evict_oldest(l)) {}
        if (l->used + 1 > CHATDS_UI_LOG_BYTES) break;
        l->text[l->used++] = (char)ch;
    }
    rewrap_from(l, l->open_off, l->open_line);
    while (l->nlines >= CHATDS_UI_LOG_LINES && evict_oldest(l)) {}
    return (int)(l->nlines - l->open_line) - before;
}
