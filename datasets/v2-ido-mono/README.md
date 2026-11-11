# v2-ido-mono: sources and licences

`mono.jsonl.gz` holds 217,841 Ido sentences, one JSON object per line: `{"text", "source"}`.
It was built by `corpus/build_ido_mono.py` and is frozen by `SHA256SUMS`. The sentences are
third-party text, redistributed under the terms of their sources:

| `source` | sentences | origin | licence |
|---|---|---|---|
| `iowiki` | 173,944 | Ido Wikipedia ([io.wikipedia.org](https://io.wikipedia.org)), `iowiki-latest-pages-articles.xml.bz2` dump downloaded 2026-05-01 | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), © Wikipedia contributors |
| `fineweb2:<host>` | 43,897 | [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), `data/ido_Latn/train/000_00000.parquet` (downloaded 2026-11-10), only documents not from wikipedia.org; `<host>` is the original site | [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/); the underlying web text is subject to the terms of the original sites (see the FineWeb-2 card) |

Changes made to the source text: extracted from wiki markup or web pages, split into sentences,
filtered (shape, Esperanto letters, Ido vocabulary share, bot-stub templates), deduplicated, and
every sentence occurring in a published evaluation or training set removed. The `iowiki` part,
as an adaptation of Wikipedia text, is shared under CC BY-SA 4.0. Attribution for an individual
sentence is the Wikipedia article history (iowiki) or the host named in `source` (FineWeb-2).
