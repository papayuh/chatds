/* SPDX-License-Identifier: MIT
 * One launch-selected card; never follow a default drive onto NAND.
 * Header-only so diagnostic ROMs and standalone host probes share the policy. */
#ifndef CHATDS_STORAGE_H
#define CHATDS_STORAGE_H
#include <errno.h>
#include <stdio.h>
#include <string.h>

static inline const char *cds_storage_drive(const char *drive) {
    if (drive && (!strcmp(drive, "fat:/") || !strcmp(drive, "sd:/"))) return drive;
    errno = ENODEV;
    return NULL;
}
static inline const char *cds_storage_root(const char *drive) {
    if (!cds_storage_drive(drive)) return NULL;
    return !strcmp(drive, "sd:/") ? "sd:/chatds/" : "fat:/chatds/";
}
/* Non-overlapping buffers. Reject overflow without leaving a truncated path. */
static inline int cds_path_join(char *out, size_t cap, const char *dir, const char *name) {
    size_t d = strlen(dir), n = strlen(name);
    if (cap) out[0] = 0;
    if (d >= cap || n >= cap - d) { errno = ENAMETOOLONG; return -1; }
    memcpy(out, dir, d);
    memcpy(out + d, name, n + 1);
    return 0;
}
static inline int cds_storage_error(const char *operation, const char *path) {
    int error = errno ? errno : EIO;
    fprintf(stderr, "%s: %s: %s (errno=%d)\n", operation, path ? path : "(no path)", strerror(error), error);
    errno = error;
    return -1;
}
#endif
