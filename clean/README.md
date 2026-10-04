# Clean resident inference core (phase 1)

Independent C99 DSQ8 engine for ChatDS, with host and ARM9 static libraries.
The default product ROM now integrates this library through `clean/product/`;
see the [product README](../README.md). Tokenizer, UI, retrieval, calculator,
sampling, stop rules and golden-runner integration stay outside the core module.

- Interface: [`include/chatds_engine.h`](include/chatds_engine.h).
- Numeric/format contract: [`ARITHMETIC.md`](ARITHMETIC.md).
- Clean-room audit: [`PROVENANCE.md`](PROVENANCE.md).

## Build

From the repository root (no upstream source or private model files required):

```sh
make -C clean                         # build/host/libchatds_engine.a
make -C clean TARGET=nds              # build/nds/libchatds_engine.a
make -C clean TARGET=nds DTCM_POOL=1  # build/nds-dtcm/, opt-in DTCM scratch pool
make -C clean test
make -C clean sanitize               # ASan + UBSan, separate host-san objects
```

These targets build only the core library. Root `make` builds the complete
clean product ROM; `ds/hwkit/make-kit.sh` packages it as `chatds.nds` with a
`chatds/` data directory.

Host: GCC/Clang, `make`, libm. DS: BlocksDS/Wonderful,
with `WONDERFUL_TOOLCHAIN`, `BLOCKSDS` and optionally `PREFIX` overridable.
Link an ARM9 caller against `clean/build/nds/libchatds_engine.a`, libm, libc and
libnds using the normal BlocksDS CRT/linker script so `.itcm`/`.dtcm` relocate
properly. Compile callers with their SDK-supported language mode; the core only
includes the narrow `nds/ndstypes.h` attribute header and itself builds as C99.
No `-ffast-math`. The fast kernel uses ARM mode, not Thumb. Scalar and fast paths
are selected **at session creation**, without changing the model format.

For a host caller:

```sh
cc -std=c99 -Iclean/include caller.c clean/build/host/libchatds_engine.a -lm -o caller
```

The library build has no main function, timer, stdout output, filesystem writes,
or emulator process. Its only I/O is the optional model-file loader. Every
emulator integration must use `QT_QPA_PLATFORM=offscreen` and a timeout.

## Interface usage

```c
#include "chatds_engine.h"

/* token_ids and count come from the separate tokenizer module. */
chatds_status consume_prompt(const char *model_path, const uint32_t *token_ids,
                             uint32_t count, uint32_t *next_id) {
    if (!token_ids || !count || !next_id) return CHATDS_INVALID_ARGUMENT;
    chatds_model *model = NULL;
    chatds_session *session = NULL;
    chatds_status status = chatds_model_load(model_path, &model);
    if (status != CHATDS_OK) return status;

    /* NULL options means fast kernel and the model's full context length.
     * For a smaller budget, explicitly initialize kernel=CHATDS_KERNEL_FAST
     * in chatds_session_options as well as context_length/max_memory_bytes. */
    status = chatds_session_create(model, NULL, &session);
    chatds_output output;
    for (uint32_t i = 0; status == CHATDS_OK && i < count; ++i)
        status = chatds_session_forward(session, token_ids[i],
                                        i + 1 == count ? &output : NULL);
    if (status == CHATDS_OK) *next_id = output.argmax;
    chatds_session_destroy(session);
    chatds_model_destroy(model);
    return status;
}
```

Keep model/session alive for generation: feed the returned ID into the next
`forward`, stopping on caller-owned BOS/EOS/budget/cancel rules. `out==NULL`
prefills KV without paying for discarded classifier logits. `logits_q16` is
borrowed; copy it if needed beyond the next call. Divide by 65536 to compare
real-valued logits. `session_reset` starts an independent question without
reloading weights. Invalid tokens/full context do not advance or mutate state.

A model can be shared by independent sessions and must outlive them. Sessions
have no shared mutable scratch unless the ARM9 library is built with
`DTCM_POOL=1`; then one fitting fast session claims a 10 KiB typed DTCM pool and
subsequent/non-fitting sessions fall back to heap. Serialize ARM9
lifetime/forward calls and do not invoke the core in IRQs. Inspect
`session_info` for requested heap/TCM/KV sizes and cumulative saturation
warnings. These figures exclude allocator overhead, stack and unrelated UI/SDK
allocations.

Stack budget: BlocksDS puts the user stack in the same 16 KiB DTCM as the
256 B SVC and 256 B IRQ stacks and any `.dtcm` data. The default build reserves
no DTCM, leaving about 15.5 KiB of user stack. `DTCM_POOL=1` leaves about
5.5 KiB minus other `.dtcm` data. The core's own deepest ARM9 frame chain is
under 1 KiB (`-fstack-usage`: forward 152 B + matvec 80 B, plus libm; model file
load 576 B). Enable the pool only when the whole application's measured
worst-case stack, engine included, is at most 4 KiB, keeping at least 1.5 KiB of
margin, and keep a stack canary in the integration build.

## Validation and current limits

`make test` checks:

- Five synthetic header-derived shapes, different GQA/head/group sizes,
  non-power-of-two groups and unaligned rows.
- Full scalar/fast logits, reset and classified/unclassified prefill equality;
  exact last-position success, context-full and invalid-token no-mutation.
- A closed-form zero-transformer model with analytically known Q16 logits.
- File/memory loading, ownership, malformed headers/scales/norms/truncation,
  budgets, zero-logit tie handling and saturation accounting.
- 8,200 matrix rows against an independent int128/division oracle, including
  int32 extremes, wide accumulators and overflow-saturating row scales.

These tests need no private model, ROM, emulator, or network. Local frozen-P2c
host checks additionally retained exact three-question answer IDs + EOS with
zero saturation. That is not a broad accuracy certification.

Only resident DSQ8 layer-major shared-classifier models are accepted. Dimensions
are parsed and bounded, not baked into the ROM. KV is int16 Q8.8, so this first
core uses more cache RAM than the legacy Q8 KV. It does not promise bit-exact
legacy logits, universal cross-libm parity, or a publicly releasable complete
ChatDS package.

The product ROM built on this core is the default ChatDS build. Its headless
melonDS speed, TTFT, RAM and parity measurements against the legacy ROM are in
[`docs/ENGINE-BENCHMARK.md`](../docs/ENGINE-BENCHMARK.md).
