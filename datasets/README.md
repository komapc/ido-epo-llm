# Published datasets

One directory per dataset, each with `train.jsonl`, `val.jsonl`, `test.jsonl` in the
instruction format written by `corpus/make_dataset.py`. A dataset is **frozen once a
published model has been trained or scored on it**: its `SHA256SUMS` is checked in CI,
and a rebuilt or extended split goes into a new directory instead.

| dataset | directions | train / val / test | used by |
|---|---|---|---|
| `v1-io-eo` | ido↔epo | 34,813 / 1,002 / 1,000 | [komapc/nllb-ido-epo-600m](https://huggingface.co/komapc/nllb-ido-epo-600m) (chrF 73.4) |
| `eval-ido-x-v1` | ido↔eng/spa/fra/deu | eval only: dev 1,003 / test 3,592 rows | baseline of the split-decoder browser build ([table](../eval/README_ido_x.md)) |
| `v2-ido-mono` | Ido monolingual | 217,841 sentences (`mono.jsonl.gz`) | back-translation pool of `finetune_ido_decoder.ipynb` |

**`eval-ido-x-v1`** (`corpus/build_eval_ido_x.py`): Tatoeba Ido sentences with a direct link to an
English/Spanish/French/German sentence, none of whose link neighbourhood (translations and their
translations) is in v1 train. Ido side from v1 val → `dev`; from v1 test or never used → `test`.
For English and German (over 500 never-used sentences each), half of the never-used pairs are held
back as future training data, minus any pair sharing a Tatoeba id or text with dev/test: 859 pairs
in `corpus/out/eval-ido-x-v1_reserved_train.jsonl` (unpublished). Rows over 200 characters are
dropped. Rows carry Tatoeba ids.
Test sizes per direction: ido→eng 803, deu 597, fra 256, spa 176 (X→ido slightly fewer:
one row per X sentence). Spanish and French are too small to resolve differences of about 2 chrF.

**`v2-ido-mono`** (`corpus/build_ido_mono.py`): sentences from the Ido Wikipedia dump (CC BY-SA) and
the non-Wikipedia part of FineWeb-2 `ido_Latn` (ODC-By), filtered for shape, Esperanto letters,
Ido vocabulary share, bot-stub templates, and any text in the datasets above. This is third-party
text redistributed under CC BY-SA 4.0 (Wikipedia part) and ODC-By 1.0 (FineWeb-2 part); sources
and attribution in [`v2-ido-mono/README.md`](v2-ido-mono/README.md).

`corpus/check_split.py` also checks across datasets: no val/dev/test text of any dataset may
appear in any `train.jsonl` or `mono.jsonl.gz`.

To publish a new one: build into `corpus/out/`, then
`mkdir datasets/<name> && cp corpus/out/{train,val,test}.jsonl datasets/<name>/`,
`(cd datasets/<name> && sha256sum train.jsonl val.jsonl test.jsonl > SHA256SUMS)`,
and run `python3 corpus/check_split.py <name>`.
