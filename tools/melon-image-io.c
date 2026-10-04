/* Linux-only melonDS 1.1 harness shim, never linked into a DS ROM.
 *
 * Qt's Platform::OpenFile uses fdopen(dup(QFile::handle())). FATStorage's
 * WriteSectorsInternal calls fwrite without fflush, leaving the last FAT
 * directory update buffered until another seek or a clean emulator exit.
 * SIGTERM discards it; mtools polling can therefore see out.tmp forever even
 * though the DS completed rename successfully.
 *
 * Disable stdio buffering ONLY for the exact disposable image named by the
 * launcher. OS page-cache visibility is enough for mtools (not power-loss
 * durability). No global preload installation, no other files affected.
 * Upstream: melonDS tag 1.1, src/FATStorage.cpp and
 * src/frontend/qt_sdl/Platform.cpp. Remove when upstream flushes DLDI writes.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

FILE *fdopen(int fd, const char *mode) {
    FILE *(*real_fdopen)(int, const char *) = dlsym(RTLD_NEXT, "fdopen");
    if (!real_fdopen) abort();
    FILE *file = real_fdopen(fd, mode);
    const char *image = getenv("MELON_UNBUFFERED_IMAGE");
    if (file && image) {
        char link[64], path[4096];
        snprintf(link, sizeof(link), "/proc/self/fd/%d", fd);
        ssize_t n = readlink(link, path, sizeof(path) - 1);
        if (n >= 0) {
            path[n] = '\0';
            if (!strcmp(path, image)) {
                if (setvbuf(file, NULL, _IONBF, 0)) abort();
                fprintf(stderr, "[melon-image] unbuffered: %s\n", path);
            }
        }
    }
    return file;
}
