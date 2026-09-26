# ido-epo-llm — dataset

Ido↔Esperanto parallel data for fine-tuning a neural translation engine
plus the pipeline that builds it (clone into the apertium-dev workspace as `llm/`).

- `data/train.jsonl` — 34,813 instruction pairs (Tatoeba real ×3 weight + Apertium synthetic)
- `data/val.jsonl`   — 1,002 real held-out pairs
- `data/test.jsonl`  — 1,000 real held-out pairs (eval split)

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

1. **Tatoeba** (`data/build_tatoeba_pairs.py`) — real human IO↔EO sentence
   pairs, direct and pivoted (IO↔EN↔EO, IO↔fre↔EO, etc.). Small but the only
   ground truth. Used for the **held-out test/val split** and as high-weight
   training data. Never let synthetic data leak into eval.

2. **Apertium-filtered synthetic** (`data/build_apertium_synthetic.py`) — run
   Apertium over iowiki/sourceswiki Ido text, keep **only** sentences whose
   translation contains zero `*` (unknown) and `#` (gen-fail) markers. This is
   high-precision weak supervision: bulk volume + grammaticality. It **caps the
   model at Apertium quality**, so it is mixed with (not substituted for) real
   data, and excluded from eval.

3. **Bidix glossary** — the 96k-pair `apertium-ido-epo` bidix is word-level.
   Used to (a) inject relevant term hints into prompts and (b) sanity-filter
   synthetic pairs. Not training data on its own.

`data/make_dataset.py` merges, deduplicates (by normalized source), holds out a
real-pairs-only test/val split, and emits instruction-formatted jsonl.

### Instruction format
```
{"instruction": "Translate Ido to Esperanto:", "input": "<src>", "output": "<tgt>", "src_lang": "ido", "tgt_lang": "epo", "provenance": "tatoeba|apertium_synth"}
```
One bidirectional model handles both directions; direction is encoded in the
instruction string.

## Model & training

- Base: **LLaMA 3.2 3B Instruct** (path to 8B documented in the notebook).
- Method: **QLoRA 4-bit via Unsloth**, fits free-Colab T4. GPU work runs on
  **Colab**, never EC2 (project steering rule).
- `finetune_colab.ipynb` — self-contained: pulls the jsonl, trains, pushes
  adapter to HF Hub (or Drive).

## Eval

`eval/eval_chrf.py` — chrF + BLEU on the held-out **real** Tatoeba split, LLM
vs. Apertium side by side. Ship only if the LLM wins (or wins on the subset
where Apertium emits `*`/`#`).

## Serving (deferred)

When a model exists, add an `engine` field to the worker's `/api/translate`
handler and fan out to whatever backend hosts the adapter (serverless GPU vs.
hosted API decided then). The frontend already has room for a side-by-side
compare UI to gather preference data for a later DPO pass.

## Run order
```bash
python3 data/build_tatoeba_pairs.py      # → data/out/tatoeba_io_eo.jsonl
python3 data/build_apertium_synthetic.py # → data/out/apertium_synth.jsonl
python3 data/make_dataset.py             # → data/out/{train,val,test}.jsonl
```

### Publish the dataset for Colab
This repo is cloned into the apertium-dev workspace as `llm/` (the scripts
resolve `apertium-ido-epo` etc. via `parents[1]`). The notebook `git clone`s
this repo and reads `data/{train,val,test}.jsonl`. To refresh:
```bash
cp data/out/{train,val,test}.jsonl data/ && python3 data/check_split.py && git commit -am "dataset: refresh" && git push
```
Then open `finetune_colab.ipynb` on Colab (T4), run all cells → `preds.jsonl`.

### Score the result
```bash
python3 eval/eval_chrf.py --apertium --pred preds.jsonl   # LLM vs Apertium baseline
```
The scorer refuses preds whose `input`s don't line up row-by-row with
`data/test.jsonl`, so predictions made on another split can't be scored by accident.

`python3 data/check_split.py` (also run in CI) fails if the published
`data/{train,val,test}.jsonl` share any sentence text. Before 2026-09-26 the
committed split predated the union-find fix and ~90% of val/test texts were in
train, so any model trained on it has inflated scores.
