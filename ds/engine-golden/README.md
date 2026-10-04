# device observation fixtures

`fixtures.json` contains inputs and black-box observations, not engine source:
token IDs, stop decisions, prompt counts, context text and host top-two logits.
33 cases cover retrieval, arithmetic, code, budgets, near ties, repeated boots
and context boundaries.

Successful output observations are unchanged. Public-prep renamed the safety
class to `baseline-fails` and removed historical card paths from diagnostics.
The original file hash remains in provenance; the policy's hash now pins the
normalized file. Historical benchmark hashes still identify their original
inputs. No model, ROM or upstream implementation is embedded in either file.
Context sentences derived from Wikipedia need its attribution; see
[third-party notices](../../THIRD_PARTY_NOTICES.md).

- Successful cases require exact IDs, stop, count, context and text.
- `baseline-fails` requires completed inference or an explicit diagnostic
  failure; timeouts, crashes and malformed outputs never pass.
- The product retains 31/33 behaviors, with two exact pinned numerical outputs
  in `clean-known-divergences.json`. This policy does not waive other changes.

See [runner commands](../../docs/ENGINE-GOLDEN.md) and
[measurements](../../docs/ENGINE-BENCHMARK.md). These are compatibility
observations, not a model quality certification.
