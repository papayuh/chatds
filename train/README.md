# training

`train_instruct.py` trains instruction models with PyTorch, grouped-query
attention and masked prompt loss. It needs the MIT `RileyGreiff/ds-llm` model,
export and tokenizer modules under `reference/ds-llm/` (ignored by git).
`tools/fetch-refs.sh` fetches that optional dependency. See
[third-party notices](../THIRD_PARTY_NOTICES.md).

```sh
python3 -m pip install torch numpy sentencepiece
bash tools/fetch-refs.sh
python3 train/train_instruct.py --selftest=True --device=cpu --compile=False
```

Train a tokenizer on licensed UTF-8 text using `tools/train_tokenizer.py`,
then export its runtime binary with `ds/tokenizer/tokbin.py`; see the root
README. No pretrained tokenizer, model weights or full dataset is shipped.
Prepare instruction pairs with `corpus/` and export training shards with
`corpus/export_train.py` using that same `.model`. The trainer accepts `--key=value` overrides; its
module documentation lists data, checkpoint and tokenizer-provenance checks.
Do not load untrusted PyTorch checkpoints.

`retain-checkpoints.py`, `select-checkpoint.py` and `soup_ckpt.py` handle local
checkpoint selection. `qat4.py` is experimental fake-quant math; the product
supports DSQ8, not DSQ4. No device-format converter, capacity planner or
end-to-end training campaign wrapper is shipped. Training can produce float
checkpoints, but converting new weights to DSQ8 needs an independently licensed
exporter. Existing compatible DSQ8 assets can be evaluated with
`eval/run_eval.py` and packaged with `ds/hwkit/make-kit.sh`.
