# ido-epo-llm — neural translation for Ido

Parallel data, training notebooks and evaluation for neural Ido translation
(clone into the apertium-dev workspace as `llm/`). The first model is
Ido↔Esperanto, [komapc/nllb-ido-epo-600m](https://huggingface.co/komapc/nllb-ido-epo-600m),
served in the browser on [tradukilo.lipu.dev](https://tradukilo.lipu.dev/) next to Apertium.

## Layout

| path | what |
|---|---|
| `mtlib/` | shared code: paths, the language registry (`langs.py`: Tatoeba code ↔ NLLB token ↔ Apertium mode), chrF/BLEU |
| `corpus/` | builders: Tatoeba pairs, Apertium synthetic pairs, the leakage-safe split; `cache/` and `out/` are local and gitignored |
| `datasets/<name>/` | published splits; frozen once a published model used them ([list](datasets/README.md)) |
| `eval/` | `eval_chrf.py` scorer; `preds/<dataset>/` model predictions |
| `train/` | generator for `finetune_nllb.ipynb` |
| `space/` | browser build (vocab pruning, ONNX) and the HF Space |

Adding a language means a line in `mtlib/langs.py`; the pair builder, the scorer
and the notebook generator take languages and datasets as arguments.

Current dataset `datasets/v1-io-eo/`:
- `train.jsonl` — 34,813 instruction pairs (Tatoeba real ×3 weight + Apertium synthetic)
- `val.jsonl`   — 1,002 real held-out pairs
- `test.jsonl`  — 1,000 real held-out pairs (eval split)

Real pairs derive from [Tatoeba](https://tatoeba.org) (CC-BY 2.0 FR — attribution
required). Synthetic pairs are high-precision Apertium ido-epo output (zero
failure markers). Eval is real-only with no source leakage into train.

## Why a neural engine at all

Apertium is high-precision but brittle: unknown words come back as `*token`,
generation failures as `#token`, and it has no notion of fluency or context.
An LLM's value is exactly there — robustness on inputs Apertium fails, and
fluency. If the LLM only ever matches Apertium, it isn't worth serving, so the
**eval metric is head-to-head chrF/BLEU on real human pairs**, not synthetic.

## Data strategy

There is essentially **no sentence-level IO↔EO parallel corpus in the world**,
so we build one from three sources of decreasing trust:

1. **Tatoeba** (`corpus/build_tatoeba_pairs.py`) — real human IO↔EO sentence
   pairs, direct and pivoted (IO↔EN↔EO, IO↔fre↔EO, etc.). Small but the only
   ground truth. Used for the **held-out test/val split** and as high-weight
   training data. Never let synthetic data leak into eval.

2. **Apertium-filtered synthetic** (`corpus/build_apertium_synthetic.py`) — run
   Apertium over iowiki/sourceswiki Ido text, keep **only** sentences whose
   translation contains zero `*` (unknown) and `#` (gen-fail) markers. This is
   high-precision weak supervision: bulk volume + grammaticality. It **caps the
   model at Apertium quality**, so it is mixed with (not substituted for) real
   data, and excluded from eval.

3. **Bidix glossary** — the 96k-pair `apertium-ido-epo` bidix is word-level.
   Used to (a) inject relevant term hints into prompts and (b) sanity-filter
   synthetic pairs. Not training data on its own.

`corpus/make_dataset.py` merges, deduplicates (by normalized source), holds out a
real-pairs-only test/val split, and emits instruction-formatted jsonl.

### Instruction format
```
{"instruction": "Translate Ido to Esperanto:", "input": "<src>", "output": "<tgt>", "src_lang": "ido", "tgt_lang": "epo", "provenance": "tatoeba|apertium_synth"}
```
One bidirectional model handles both directions; direction is encoded in the
instruction string.

## Model & training

Two candidates, same data and same test split, so scores compare 1:1:

- `finetune_colab.ipynb` — **LLaMA 3.2 3B Instruct**, QLoRA 4-bit with plain
  transformers+peft+trl (Unsloth dropped: incompatible with transformers 5.x).
  bf16 on T4 is software-emulated, so it is capped at 300 steps (~3h).
- `finetune_nllb.ipynb` — **NLLB-200 distilled 600M**, full fine-tune in fp16
  AMP. Dedicated encoder-decoder MT model that already knows Esperanto; Ido gets
  a new `ido_Latn` language token (init = mean of epo/ita/spa). Small enough to
  serve on CPU. Weights are CC-BY-NC-4.0. Generated from
  `train/build_nllb_notebook.py` (edit there; `SMOKE=1 ... --smoke` runs the
  cells on CPU on a tiny subset).

GPU work runs on **Kaggle** (T4 x2, Save & Run All) or Colab, never EC2.

## Eval

`eval/eval_chrf.py` — chrF + BLEU on the held-out **real** Tatoeba split, LLM
vs. Apertium side by side. Ship only if the LLM wins (or wins on the subset
where Apertium emits `*`/`#`).

### Results (2026-11-09, clean split, 1000 real pairs)

| | chrF | BLEU | chrF where Apertium fails (25%) |
|---|---|---|---|
| Apertium ido-epo (current) | 65.2 | 45.4 | 45.0 |
| **NLLB-200 600M** | **73.4** | **53.9** | **66.9** |
| Llama-3.2-3B QLoRA | 60.9 | 38.7 | 51.9 |

NLLB wins in both directions (ido→epo 74.0 vs 68.3; epo→ido 72.7 vs 62.0), and it
also wins on the inputs Apertium handles cleanly. Llama loses to Apertium and is
dropped. The Kaggle predictions are in `eval/preds/`. To re-score:
`python3 eval/eval_chrf.py --apertium --pred eval/preds/v1-io-eo/nllb-600m.jsonl`.

**Published model:** [komapc/nllb-ido-epo-600m](https://huggingface.co/komapc/nllb-ido-epo-600m)
(CC-BY-NC-4.0, with a usage snippet in the model card).
**Try it:** [browser demo Space](https://huggingface.co/spaces/komapc/ido-epo-translator)
(runs locally via transformers.js; build in [`space/`](space/README.md)).

## Serving (deferred)

The NLLB model is published (see above) but not served yet. To serve it, add an
`engine` field to the worker's `/api/translate` handler and fan out to whatever
backend hosts the model. 600M params run on CPU; the backend is decided then. The frontend already has room for a side-by-side
compare UI to gather preference data for a later DPO pass.

## Run order
```bash
python3 corpus/build_tatoeba_pairs.py      # → corpus/out/tatoeba_ido_epo.jsonl (--src/--tgt for other pairs)
python3 corpus/build_apertium_synthetic.py # → corpus/out/apertium_synth.jsonl
python3 corpus/make_dataset.py             # → corpus/out/{train,val,test}.jsonl
```

### Publish a dataset
This repo is cloned into the apertium-dev workspace as `llm/` (`mtlib/paths.py`
finds `apertium-ido-epo` as a sibling). The notebook `git clone`s this repo and
reads `datasets/<DATASET>/{train,val,test}.jsonl`. A published dataset is never
rebuilt in place; a new split gets a new directory (steps in
[datasets/README.md](datasets/README.md)), then
`DATASET=<name> python3 train/build_nllb_notebook.py` points the notebook at it.
Reproducibility: with the Tatoeba dump cached in June 2026 and the
`apertium_synth.jsonl` built then (both local, not committed), `corpus/make_dataset.py`
writes `v1-io-eo` byte for byte. A fresh Apertium run gives different synthetic
pairs, because the Apertium pair has changed since.

### Score the result
```bash
python3 eval/eval_chrf.py --apertium --pred preds.jsonl   # LLM vs Apertium baseline (default dataset v1-io-eo)
```
The scorer refuses preds whose `input`s don't line up row-by-row with
the dataset's `test.jsonl`, so predictions made on another split can't be scored by accident.

`python3 corpus/check_split.py` (also run in CI) fails if a published dataset's
splits share any sentence text; CI also checks each dataset's `SHA256SUMS` and
that the v1 predictions still score chrF 73.40. Before 2026-09-26 the
committed split predated the union-find fix and ~90% of val/test texts were in
train, so any model trained on it has inflated scores.
