/* SPDX-License-Identifier: MIT
 * Card I/O: run.txt, file preflight, atomic result files (temp + rename, status= last).
 * Result format: what ds/hwkit/smoke-test.sh reads. */
#include <stdlib.h>
#include <string.h>
#include "plumbing.h"

#define PATH_CAP 256

int cds_config_read(const char *path, cds_config *c) {
    char line[256];
    FILE *f;
    int rc = 0;
    memset(c, 0, sizeof *c);
    c->steps = 64, c->kbd = 1, c->instruct = 1;
    if (!(f = fopen(path, "rb")))
        return -1;
    while (fgets(line, sizeof line, f)) {
        size_t n = strcspn(line, "\r\n");
        int whole = line[n] != 0 || feof(f);
        line[n] = 0;
        if (!whole) { /* over-long line: skip the rest of it */
            int ch;
            while ((ch = fgetc(f)) != EOF && ch != '\n')
                ;
            rc = -2;
            continue;
        }
        char *v = strchr(line, '=');
        if (line[0] == '#' || !v)
            continue;
        *v++ = 0;
        char *end;
        if (!strcmp(line, "prompt")) {
            if (strlen(v) > CDS_QUESTION_MAX)
                rc = -2;
            else
                strcpy(c->prompt, v);
        } else if (!strcmp(line, "steps") || !strcmp(line, "kbd") || !strcmp(line, "instruct")) {
            unsigned long x = strtoul(v, &end, 10);
            if (!*v || *end || x > 100000) {
                rc = -2;
                continue;
            }
            if (line[0] == 's')
                c->steps = (uint32_t)x;
            else if (line[0] == 'k')
                c->kbd = x != 0;
            else
                c->instruct = x != 0;
        }
    }
    fclose(f);
    return rc;
}

chatds_status cds_preflight(const char *path, const char *magic, size_t magic_len,
                            uint32_t min_size, uint32_t *size) {
    char head[16];
    FILE *f = fopen(path, "rb");
    long n = -1;
    *size = 0;
    if (!f)
        return CHATDS_IO_ERROR;
    if (fseek(f, 0, SEEK_END) == 0)
        n = ftell(f);
    int bad = n < 0 || (unsigned long)n < min_size || magic_len > sizeof head;
    if (!bad && magic_len)
        bad = fseek(f, 0, SEEK_SET) || fread(head, 1, magic_len, f) != magic_len || memcmp(head, magic, magic_len);
    fclose(f);
    if (n < 0)
        return CHATDS_IO_ERROR;
    *size = (uint32_t)n;
    return bad ? CHATDS_BAD_FORMAT : CHATDS_OK;
}

static int tmp_path(const char *path, char *out) {
    return snprintf(out, PATH_CAP, "%s.tmp", path) < PATH_CAP ? 0 : -1;
}

FILE *cds_atomic_open(const char *path) {
    char tmp[PATH_CAP];
    return tmp_path(path, tmp) ? NULL : fopen(tmp, "wb");
}

int cds_atomic_commit(FILE *f, const char *path) {
    char tmp[PATH_CAP];
    if (!f)
        return -1;
    int bad = ferror(f) | fflush(f);
    bad |= fclose(f);
    if (bad || tmp_path(path, tmp))
        return -1;
    remove(path); /* FAT rename does not replace an existing file */
    return rename(tmp, path) ? -1 : 0;
}

static FILE *open_in(const char *dir, const char *name, char *path) {
    if (snprintf(path, PATH_CAP, "%s%s", dir, name) >= PATH_CAP - 4)
        return NULL;
    return cds_atomic_open(path);
}

int cds_publish(const char *dir, const cds_result *r) {
    char path[PATH_CAP];
    const cds_answer *a = r->answer;
    const cds_generate_result *g = &a->gen;
    int bad = 0;
    FILE *f;

    if ((f = open_in(dir, "ids.txt", path)))
        for (uint32_t i = 0; r->ids && i < g->generated; i++)
            fprintf(f, "%lu\n", (unsigned long)r->ids[i]);
    bad |= cds_atomic_commit(f, path);

    if ((f = open_in(dir, "ctx.txt", path))) {
        fprintf(f, "ctx=%s\n", a->ctx);
        if (r->kb)
            fprintf(f, "kb=%s lookups=%lu sector_reads=%lu\n", r->kb->err ? "error" : "ok",
                    (unsigned long)r->kb->lookups, (unsigned long)r->kb->sector_reads);
        else
            fprintf(f, "kb=none\n");
    }
    bad |= cds_atomic_commit(f, path);

    if (r->instruct) {
        if ((f = open_in(dir, "calc.txt", path)))
            fprintf(f, "calc=%s\n", a->calc);
        bad |= cds_atomic_commit(f, path);
    }

    if ((f = open_in(dir, "heap.txt", path))) {
        fprintf(f, "model_bytes=%lu\n", (unsigned long)r->model_bytes);
        if (r->session) {
            chatds_session_info in;
            chatds_session_get_info(r->session, &in);
            fprintf(f, "session_heap_bytes=%lu\nsession_tcm_bytes=%lu\nkv_bytes=%lu\ncontext=%lu\nsaturations=%llu\n",
                    (unsigned long)in.heap_bytes, (unsigned long)in.tcm_bytes, (unsigned long)in.kv_bytes,
                    (unsigned long)in.context_length, (unsigned long long)in.saturations);
        }
        if (r->heap_extra)
            fputs(r->heap_extra, f);
    }
    bad |= cds_atomic_commit(f, path);

    /* out.txt last: pollers treat it as the completion signal; status= is its final line. */
    if ((f = open_in(dir, "out.txt", path))) {
        uint32_t tokens = g->prompt_tokens ? g->prompt_tokens - 1 + g->generated : 0;
        fprintf(f, "%s\ntokens=%lu\nprompt_tokens=%lu\ngenerated=%lu\nstop=%s\nms=%lu\nstatus=%s\n",
                r->text ? r->text : "", (unsigned long)tokens, (unsigned long)g->prompt_tokens,
                (unsigned long)g->generated, cds_stop_string(g->stop), (unsigned long)r->ms,
                r->ok && !bad ? "OK" : "FAIL");
    }
    bad |= cds_atomic_commit(f, path);
    return bad ? -1 : 0;
}
