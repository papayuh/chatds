# Resident DSQ8 arithmetic and tolerance contract

This contract defines **our** arithmetic, not a legacy engine's.
The implementation is independent; no unlicensed upstream source is an authority or a
permitted implementation reference. See [PROVENANCE.md](PROVENANCE.md).

## Supported input and ownership

The loader accepts the existing little-endian TIE1 header, version 1, payload 4
(DSQ8), 512-byte header, flags `0x101` (layer-major, shared embedding/classifier).
Other payloads, versions, flags and layouts return `CHATDS_UNSUPPORTED`; notably
there is no float/Q4 reader, untied classifier, paging, or format conversion.

All dimensions come from the header. Safety ceilings, **not model dimensions**:

- `dim`, `hidden_dim`: 1..8192; layers: 1..128; vocab: 1..65536;
  model context: 1..4096; serialized model: at most 64 MiB.
- Group size: positive multiple of 8, at most 256. Both matrix input widths
  (`dim`, `hidden_dim`) must be divisible by it.
- `dim % heads == 0`, `heads % kv_heads == 0`; heads and KV heads positive;
  each head has an even width. GQA maps contiguous query-head groups to KV heads.
- Negative group/row scales, nonfinite or unrepresentable normalization weights,
  truncated payloads and trailing bytes are rejected. Zero scales are valid.
- A session can request a shorter context, never a longer one. Its default
  allocation safety ceiling is 128 MiB, further restricted by the caller's
  `max_memory_bytes`. These ceilings are not a claim that such models fit a DS.
  DS callers must choose model/window budgets appropriate to their remaining
  application heap and reject allocation failures.

The model owns its buffer. `from_memory` copies input; `load` allocates and reads
once without a second full model copy. Normalization weights are converted to
Q16.16 **inside that owned buffer**, never in the file or caller's input. It is
immutable after construction and may be shared; callers destroy sessions first.

The header's dimensions determine every offset. Normalization vectors are
contiguous from offset 512: attention norms for all layers, FFN norms for all
layers, then the final norm. Embedding begins on the next 512-byte boundary.
Each layer begins on a 512-byte boundary; Q/K/V/O/gate/down/up matrices are
contiguous inside that layer. No per-matrix padding. A row has `cols` int8 values,
`cols/group_size` int16 group scales, and one int32 row scale. The exact computed
file size must match the actual file size. Unaligned row-scale reads are safe.

## Numeric representation

Requires 8-bit bytes, binary32 `float`, and arithmetic signed right shift (checked
at build/load). No `-ffast-math`. ARM9 uses software floating point. Host builds
are C99; only the optional randomized test oracle uses GCC/Clang `__int128`.

All activations and exposed logits are signed **Q16.16**. Weight values are int8;
group scales are nonnegative Q1.15 (`0..32767`); row scales are nonnegative Q8.24
(`0..INT32_MAX`). Negative right shifts round **toward minus infinity**, not zero.
Conversion from finite float to fixed point rounds halfway **away from zero**.

### Matrix-vector product

For one input vector `x`, choose the smallest `shift >= 0` such that
`max(abs(x)) >> shift <= 16383`. Set `qx[i] = x[i] >> shift`; values lie in
`[-16384,16383]`, including the extra unit from flooring negative inputs.

For every row:

```
dot[g] = sum(weight[g*gs+i] * qx[g*gs+i], i=0..gs-1)
acc    = sum((dot[g] * group_scale[g]) >> 15, g=0..groups-1)
y      = saturate_int32((acc * row_scale) >> (24-shift))
```

Group scaling happens **before** summing groups. Reordering this rounding changes
the numeric contract. The scalar path uses int64 dots/sums. The fast path unrolls
32-wide groups and uses int32 dots (at most 2^29 magnitude for gs<=256); widths
<=1024 also have a provably bounded int32 scaled sum. Larger rows use int64 sums.
All scale products use int64. Products that would overflow int64 are detected
before multiplication and necessarily saturate even after the largest final
shift. Every int32 input, including `INT32_MIN`, is handled without signed UB.

Embedding lookup uses
`floor(weight * group_scale * row_scale / 2^23)`, saturated to int32.

### Normalization, rotary position and nonlinearities

- RMSNorm: epsilon `1e-5`. Squares are accumulated in int64. Inputs exceeding
  `0xffffff` magnitude are right-shifted until bounded before squaring; the mean
  restores this scale with `ldexpf`. This keeps the sum <=2^61 for supported
  widths. One soft-float `sqrtf` produces a Q16.16 reciprocal. The two products
  (input × reciprocal, then × norm weight) each floor at 16 bits and saturate.
- RoPE: base 10000, adjacent pairs, frequency exponent `2*j/head_size`.
  `powf` initializes frequencies once; `cosf`/`sinf` evaluate each position.
  Coefficients round to Q16.16; products and add/subtract saturate in sequence.
- SiLU: sigmoid table indexed by `(gate >> 10) + 1024`, clamped to `0..2048`.
  Table values are `round(65536 / (1+exp(-(i-1024)/64)))`. Gate×sigmoid and
  then ×up each floor at 16 bits and saturate.
- Residual additions saturate int32; no wraparound is permitted.

### Attention and KV

KV is **int16 Q8.8**, not legacy-compatible dynamically scaled int8 KV.
Queries and cached K/V use `clip_int16(value_q16 >> 8)`. Clipping increments the
session's saturation counter. This doubles raw KV storage relative to int8; do
not infer bigger-model capacity from the engine's compute speed alone.

Scores are accumulated in int64, multiplied by
`round(65536/sqrt(head_size))`, then shifted 26 to obtain 1/64-unit table indices.
Subtract the maximum. Exponent table: `round(32768*exp(-i/64))`, indices 0..1024;
larger negative deltas clamp to 1024 (whose quantized exponent is zero).
The maximum's exponent is 32768, so the denominator cannot be zero.

Probability uses `reciprocal = floor(2^31/sum_exp)` and
`p[t] = floor(exp[t]*reciprocal/2^16)` (Q15). Their sum is at most 32768.
Weighted Q8.8 values accumulate without overflow in int32, then shift seven to
return Q16.16. There is no sliding window: only positions already consumed in
that session participate. Reset clears KV, position, logits and saturation count.

## Equality and tolerances

1. **Fast versus scalar:** exact Q16.16 logits, argmax, saturation counts and
   session position on the same platform/build. Tolerance is **zero**, not a
   floating epsilon. Optimizations must preserve accumulation/rounding order.
   Tests cover full sessions with five header-derived shapes and 8,200 randomized
   matrix rows against an independent int128/division oracle, including extreme
   inputs, scales, non-power-of-two groups and two-byte-aligned rows.
2. **Same session after reset / classified versus unclassified prefill:** exact
   subsequent logits. Invalid-token/context-full calls must leave state and
   output untouched. The last valid window position succeeds; only the next call
   returns `CHATDS_CONTEXT_FULL`. Lowest ID wins exact argmax ties. Token IDs 1/2
   have no special treatment inside the core; stop rules belong to the caller.
3. **Host versus ARM9:** fixed-point operations have the same semantics, but
   libc `sqrtf`/trig/exp/pow rounding can change coefficients. Cross-platform
   token/logit gates belong to the separate golden-runner suite; this core does
   not silently promise bit-identical libm across toolchains.
4. **Versus the legacy black box:** no bit-exact numeric contract. The scout's frozen P2c
   three-question fixture matched all 46 generated IDs and three EOS decisions,
   but only 122/127 teacher-forced argmax decisions; differing prefill positions
   included BOS. Sequence-wide logit RMSE was 0.076..0.103 and maximum absolute
   error 1.453 (real-valued logits, divide Q16 integers by 65536). A conservative
   fixture-specific reporting envelope is RMSE <=0.15 / max error <=1.6, plus
   exact generated IDs and EOS. **Those are not universal tolerances for new
   models/prompts.** New fixtures require their own explicit accuracy gate; do
   not hide disagreements by broadening this envelope.

Nonzero `saturations` is an accuracy warning to surface, not a failed forward or
silent success criterion. The accepted extreme-scale synthetic tests exercise
this counter. Frozen P2c host checks of this generalized core still produce the
same three answer ID sequences + EOS with zero saturation. Physical timing and
broader cross-platform parity are not certified by these checks.

## TCM and memory

ARM9 fast matvec and attention code reside in ITCM. Only when the ARM9 library
is built with `DTCM_POOL=1`, one fitting FAST session may claim a statically
reserved **10 KiB DTCM pool**, split into 2176 int32 activation
slots and 768 int16 quantization slots. All dimensions remain dynamic; sessions
that do not fit, and additional sessions, use heap workspace without changing
results. Scalar sessions always use heap workspace. ARM9 calls and lifetime
operations must be serialized and must not run in IRQ handlers.

`session_info.heap_bytes` / `tcm_bytes` report requested live storage, not malloc
metadata, stack space, or the unused portion of the opt-in TCM reservation.
When enabled, that reservation exists in the linked image even when a session
falls back to heap; the default build reserves no DTCM. See the stack budget in
[README.md](README.md). KV bytes are included in heap bytes, not an additional allocation. Model
storage is reported separately. Allocation occurs only during model/session
creation; forward/reset allocate nothing. There is no DMA, ARM7 co-processing,
VRAM use, model I/O, or logging in a forward call. Application-level stack canary,
heap/VRAM ownership and physical timing validation remain integration gates.
