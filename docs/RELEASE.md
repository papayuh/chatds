# release gates

This branch prepares the MIT-licensed, code-only public repository `chatds`.
Publishing it is separate: the private repository will be named `chatds-private`.
Use one initial
commit from the reviewed tracked tree; keep full history private. Do not change
the private repository's visibility, mirror its refs, or copy the worktree with
`cp -r`: ignored assets, logs and private history are not publication inputs.

MIT is the approved public code license. Trained tokenizers, weights, KB
indexes, FAT images and ROM binaries are excluded from the source snapshot.
Review
[PUBLIC_RELEASE_AUDIT.md](PUBLIC_RELEASE_AUDIT.md) before export.

## binary/data release is a separate gate

- establish model-weight and tokenizer redistribution rights;
- attach the KB's Wikipedia attribution, CC BY-SA terms and modification notice;
- inventory linked libnds, FatFs, picolibc/libm and ARM7 runtime notices;
- boot the exact clean kit on physical DS lite + DSpico, including stack canary;
- obtain explicit publication approval. A local kit is not a public release.

```sh
source tools/env.sh
make
bash ds/hwkit/make-kit.sh --mode assistant \
  --model path/to/model.bin --tokenizer path/to/tok.bin \
  --kb path/to/kb.bin --kb-attribution path/to/ATTRIBUTION.txt \
  --out build/release-kit
bash ds/hwkit/smoke-test.sh "$PWD/build/release-kit"
bash ds/hwkit/test-publication.sh "$PWD/build/release-kit"
```

All emulator gates are isolated/offscreen. The publication test is a one-token
file-write gate, not an upload. The default ROM is `chatds.nds`, with only the
independent engine, tokenizer, plumbing and UI. Its data directory is `chatds/`.
No unsupported older ROM can be relabeled as the clean product.

Use [ENGINE-BENCHMARK.md](ENGINE-BENCHMARK.md) for timing boundaries and identities.
The demo uses release KB3 (`f0326927caf6d8e508aa19e3efe65a3f615c53295c24841a8bf4125eb22faefc`);
compatibility uses frozen KB2 (`4eff3ac8d99a738cc0789809764e7f5b197e2eee4ed7f7628bd3a3ec9e803203`).
Do not substitute one while retaining the other's claims.

Host workflow tests do not supply private assets, emulator or physical hardware.
The requested shipping pipeline skips CI; report that as skipped, not a claim
that device gates ran in GitHub Actions.
