# Engine swap: exact-parity investigation (not a release result)

Historical investigation, followed by the captain's 2026-10-03 decision to
accept 31/33: `near-tie-uppercase` and `raw-length-0` are documented numerical
exceptions, with every other fixture still enforced. The default product is now
the clean engine. No golden was changed, no prompt-specific output substitution
was added, and no candidate below was adopted into production arithmetic.
Fresh measurements are in [ENGINE-BENCHMARK.md](ENGINE-BENCHMARK.md).

## Reproduction boundary

No unlicensed upstream source was opened, searched, copied, compiled, or disassembled.
Inputs were the clean engine/scout sources, the frozen suite, public model bytes,
and public outputs from prebuilt black boxes. Synthetic model files were written
under `build/swap-diagnosis/`; original assets were never modified.

The prebuilt host CLI is used only through its documented `--model`,
`--tokenizer`, `--profile`, `--ids`, and `--logits-out` options. The profile is
passed opaquely, not opened. This CLI's SHA-256 is
`57e694d941a9f602fd31e6e13d4aea325fa025c653bb447fee335b0b91cd8481`.
It differs from the frozen host executable's identity, but its original-model
BOS logit vector is byte-identical to the scout's frozen 2048-float observation.
Synthetic device probes additionally used the frozen ROM, SHA-256
`bfdd06e783d7d7f6b0e6f8e8406295b0ee5d2cca319e163d46b0b3aa9404f9b1`.
Every device probe was headless, using `tools/engine_golden.py`'s private
config/image/ROM-copy boundary and process cleanup.

## Confirmed facts

1. The clean-only product ROM builds and connects engine, tokenizer, plumbing,
   UI, and result publication. The final product ships `chatds.nds` and uses
   `fat:/chatds/`. Only the historical black-box oracle uses the old card path.
2. The scout and generalized engine agree exactly on both failing inputs:
   **2,622 operation records** for the uppercase question and **69** for empty
   BOS. Records include embedding, every RMSNorm/matvec/attention/residual, and
   classifier logits. This is not a generalization regression.
3. After reproducing the frozen KB2 sentence/80-byte truncation and raw prompt
   echo, the host product passes **31/33** fixtures. The two failures also occur
   on the headless product ROM:
   - `near-tie-uppercase`: clean/scout generate `  Shard correct.` instead of
     `  also kncopess.`.
   - `raw-length-0`: input is exactly `[1]`; clean/scout predict token 366 and
     budget-stop instead of predicting BOS and emitting nothing.
4. Tokenization is independently ruled out. A synthetic echo model uses equal-
   norm random sign codewords, zero transformer weights, and a tied classifier.
   Its per-position logit argmax reveals the actual input token ID. All **28
   successful-input fixtures** have identical clean and black-box prompt IDs,
   not merely matching token counts. Five oversized-input safety fixtures are
   outside this inference comparison and still reject cleanly in the product.
5. The arithmetic experiments evaluated **224 valid configurations** against
   all 28 successful-input sequences, retaining the five unchanged safety-case
   results in the /33 totals. None reached 33/33. The best count was 32/33;
   uppercase remained wrong. An initial diagnostic batch with an unintended
   signed/unsigned multiplication conversion is excluded from this count.

## Hypotheses tested

The local scripts and individual results are under
`build/swap-diagnosis/hypotheses*/` and `hypotheses*.py`/`.log`.

- Lowest/last-ID argmax tie direction.
- Matvec activation bounds 127 through 32767; fixed versus dynamic shifts;
  truncation versus nearest rounding; linear versus power-of-two quantization;
  early effective-weight scale rounding and group accumulation.
- Q16 multiplication rounding, RMS epsilon, mean-square precision, and RMS
  input/output quantization.
- Exact SiLU and alternative direct/table ranges and resolutions.
- Contiguous versus interleaved GQA heads, KV quantization block sizes and
  rounding, int16 fractional widths, and a diagnostic full-Q16 reconstructed
  cache.
- Attention probability/output scaling, wrapped reciprocal hypotheses, RoPE
  lookup resolutions/rounding, and exponential-table score precision.
- Gate/up projection order.

Reducing the matvec activation bound to 511 reaches **32/33**, preserving the
previous 31 cases, but does not improve the BOS logit error (RMSE 0.405 versus
0.404). It is an experimental token agreement, not evidence of the original
numeric contract, and was not adopted.

## Strongest black-box lead: a small-value attention discontinuity

Zeroing transformer components in copies of the model isolates the discrepancy:

| BOS experiment | Clean vs black-box full-logit RMSE |
|---|---:|
| No transformer weights | 0.0156 |
| First layer FFN only | 0.0192 |
| First layer attention only | 0.9944 |
| All six FFNs only | 0.0280 |
| Original model | 0.4042 |

For the first-layer attention-only model, the known tied classifier and final
normalization weights allow a least-squares recovery of the relative attention
contribution. There are 2048 observed logits and only 13 coefficients (embedding
plus each query-head/KV-head pairing). The fit RMSE is about 0.010. It recovers
contiguous GQA routing but an attention gain near **0.078**, not 1.

Scaling only V projection row scales gives a discontinuity; scaling O is linear:

| V row-scale multiplier | Recovered contribution relative to original unscaled V |
|---|---:|
| 1.00 | 0.0781 |
| 1.01 | 0.0702 |
| 1.05 | 0.0331 |
| 1.09 | 1.0891 |
| 1.10 | 1.1003 |
| 2.00 | 2.0023 |

A synthetic identity-projection model exposes the same cliff through public
outputs. Two copies with V row-scale integers 6000 and 6500 and a threshold
classifier produce **token 0** versus **token 1000**, respectively, on the frozen
headless ROM. The first result is token 0, not BOS: the synthetic classifier has
a deliberate equal-score axis tie won by the lower ID.

A wrapped unsigned reciprocal is a plausible explanation: for example,
`uint32((2^39)/scale)` predicts gains near `10/128` at scale 118, `4/128` at 124,
and normal gain above 128. This is a hypothesis inferred from observations,
**not a claim about unseen implementation source**. Simulating this family
reduces original-model BOS RMSE to **0.0255** and recovers BOS, but introduces
regressions (`python-max`, `raw-length-1`) and still misses uppercase: **30/33**.
Putting the attenuation after attention aggregation or into a generic matvec
also fails. The exact quantization location/rounding remains unresolved.

Additional identity-classifier probes show coordinate-level quantization errors
without KV or nonlinear transformer operations. Thus the attention cliff is not
necessarily the only numerical difference. Small classifier inputs and a
synthetic FFN-up scaling test stay approximately linear, ruling out a universal
small-input attenuation rule.

## Local evidence and commands

These diagnostic files are intentionally under ignored `build/`, not production:

- `compare.py`, `instrument.py`, `core-driver.c`, `*.trace`: scout/core equality.
- `hypotheses.py` through `hypotheses11.py`, `hypotheses/*/result.json`: arithmetic
  experiments. Each probe teacher-forces the real prompt then freely generates;
  it checks IDs, prompt count, and stop, never injects expected answer tokens.
- `truncate_probe.py`, `invert_attention.py`, `scale_probe.py`,
  `threshold_probe.py`, `quant_probe.py`, `ffn_scale_probe.py`,
  `identity_probe.py`: synthetic black-box isolation.
- `tokenizer_oracle.py`, `tokenizer-oracle/*.json`: actual input-ID equality.
- `synthetic/device-cliff-{6000,6500}.json`: headless device confirmation.
- `../swap-measure/kv16/measurements.json`: initial product device timing and
  the two reproduced failures. These are not a completed release benchmark.

Examples from the repository root:

```sh
python3 build/swap-diagnosis/compare.py
python3 build/swap-diagnosis/instrument.py
python3 build/swap-diagnosis/hypotheses.py
OPENBLAS_NUM_THREADS=1 python3 build/swap-diagnosis/scale_probe.py
python3 build/swap-diagnosis/tokenizer_oracle.py
```

The default arithmetic remains int16 Q8.8. The experimental fixed Q4.4 int8
option passed only 26/33 host fixtures, is not a shipping candidate, and has
been removed from the engine source.
This investigation did not establish 33/33 equivalence or a speed claim. The
later acceptance decision and fresh ROM measurements, not these experiments,
authorize the default-engine switch. Reaching 33/33 in future would require a
more exact black-box reconstruction of the legacy attention/quantization
contract, not a scout-kernel repair or a special case for the failing prompts.
