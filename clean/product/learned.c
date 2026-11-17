/* SPDX-License-Identifier: MIT
 * Product-private session seam: no engine ABI changes. Cache validated heads
 * in idle KV values across deterministic turns. Invalidate before generator
 * reset/prefill overwrites that arena; never borrow halfword-only VRAM keys. */
#include <stdio.h>
#include <string.h>
#include "internal.h"
#include "learned.h"
#include "chatds_storage.h"
static const char *model_directory;
static chatds_session *owner;
static cds_round3 cached;
static unsigned loaded;
enum { ACT_BYTES = 22660, OP_BYTES = 11452, TAG_BYTES = 5224,
       BLOB_BYTES = ACT_BYTES + OP_BYTES + TAG_BYTES };
static const char *const names[] = {"round3-act.cdsr", "round3-op.cdsr", "round3-tag.cdsr"};
static const size_t sizes[] = {ACT_BYTES, OP_BYTES, TAG_BYTES};
static const size_t offsets[] = {0, ACT_BYTES, ACT_BYTES + OP_BYTES};
static const uint8_t hashes[3][32] = {
 {0xe4,0x67,0x81,0xff,0xcd,0xbf,0xf4,0x85,0xf3,0x3f,0xcc,0x91,0x8d,0xff,0xc5,0x99,0x4d,0x3d,0xa8,0x26,0x24,0x1a,0xe5,0x4a,0x71,0x2b,0xef,0xb3,0x8a,0x1a,0x7a,0x86},
 {0xaa,0x82,0x5e,0x49,0x2e,0x45,0xb5,0x37,0xdf,0x09,0x15,0x0e,0x64,0x91,0x56,0x36,0xcb,0xa7,0x77,0xca,0x7e,0x9e,0x2d,0xdb,0xaa,0xdc,0x2c,0x9e,0x30,0xdb,0x5d,0xd1},
 {0x6b,0x7b,0xd4,0xdb,0x55,0x9f,0xd9,0x36,0xf1,0xa2,0x60,0xa4,0x50,0x03,0xda,0x88,0x13,0x5e,0x69,0xdc,0x61,0x81,0x4b,0x2e,0xc5,0x14,0x51,0xba,0x5c,0xcc,0xac,0xf4}
};
static int arena(chatds_session *s, uint8_t **blob, cds_router_work **work) {
    if (!s || s->kv_bytes / 2 < BLOB_BYTES + sizeof(cds_router_work)) return -1;
    *blob = (uint8_t *)s->values;
    *work = (cds_router_work *)(*blob + BLOB_BYTES);
    return 0;
}
static int load(int head, uint8_t *blob, cds_router *r) {
    char path[512]; uint8_t hash[32];
    if (!model_directory) return -1;
    if (cds_path_join(path, sizeof path, model_directory, names[head]))
        return cds_storage_error("learned path failed in directory", model_directory);
    FILE *f = fopen(path, "rb");
    if (!f) return cds_storage_error("learned open failed", path);
    /* No per-file stdio read buffer: avoid a new high-water heap allocation. */
    setvbuf(f, NULL, _IONBF, 0);
    int bad = fread(blob, 1, sizes[head], f) != sizes[head];
    bad |= fgetc(f) != EOF || ferror(f); bad |= fclose(f) != 0;
    if (bad) return -1;
    cds_router_sha256(blob, sizes[head], hash);
    /* The pinned whole-file SHA covers header and body, including the CDSR
     * body digest. Do not hash the same bytes again in cds_router_open. */
    if (memcmp(hash, hashes[head], 32) || cds_router_open(r, blob, sizes[head], 0)) return -1;
    return 0;
}
static int ensure(void *ctx, int head) {
    chatds_session *s = ctx;
    uint8_t *blob; cds_router_work *work;
    if (s != owner || arena(s, &blob, &work)) return -1;
    if (loaded & (1u << head)) return 0;
    cds_router *r = head == 0 ? &cached.act : head == 1 ? &cached.op : &cached.tag;
    if (load(head, blob + offsets[head], r)) return -1;
    loaded |= 1u << head;
    return 0;
}
void c24_models_invalidate(chatds_session *s) {
    if (s == owner) {
        loaded = 0;
        memset(&cached.act, 0, sizeof cached.act);
        memset(&cached.op, 0, sizeof cached.op);
        memset(&cached.tag, 0, sizeof cached.tag);
    }
}
int c24_models_init(chatds_session *s, const char *directory) {
    uint8_t *blob; cds_router_work *work;
    owner = s; model_directory = directory; loaded = 0;
    memset(&cached, 0, sizeof cached);
    cached.act_threshold = 1793; cached.op_threshold = 386; cached.tag_threshold = 0;
    return arena(s, &blob, &work);
}
int c24_act(chatds_session *s, const char *q, int32_t *margin) {
    uint8_t *blob; cds_router_work *work;
    if (!cds_round3_act_eligible(q)) return 0;
    if (arena(s, &blob, &work) || ensure(s, 0)) return -1;
    return cds_round3_act(&cached, work, q, 0, margin);
}
int c24_text(chatds_session *s, const char *q, cds_round3_text *result, char *out, size_t cap) {
    uint8_t *blob; cds_router_work *work;
    if (s != owner || arena(s, &blob, &work)) return -1;
    return cds_round3_textop_lazy(&cached, work, q, result, out, cap, ensure, s);
}
