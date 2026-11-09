# Published datasets

One directory per dataset, each with `train.jsonl`, `val.jsonl`, `test.jsonl` in the
instruction format written by `corpus/make_dataset.py`. A dataset is **frozen once a
published model has been trained or scored on it**: its `SHA256SUMS` is checked in CI,
and a rebuilt or extended split goes into a new directory instead.

| dataset | directions | train / val / test | used by |
|---|---|---|---|
| `v1-io-eo` | ido↔epo | 34,813 / 1,002 / 1,000 | [komapc/nllb-ido-epo-600m](https://huggingface.co/komapc/nllb-ido-epo-600m) (chrF 73.4) |

To publish a new one: build into `corpus/out/`, then
`mkdir datasets/<name> && cp corpus/out/{train,val,test}.jsonl datasets/<name>/`,
`(cd datasets/<name> && sha256sum train.jsonl val.jsonl test.jsonl > SHA256SUMS)`,
and run `python3 corpus/check_split.py <name>`.
