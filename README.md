# ido-epo-llm — dataset

Ido↔Esperanto parallel data for fine-tuning a neural translation engine
(see the `llm/` pipeline in the apertium-dev workspace).

- `data/train.jsonl` — 33,742 instruction pairs (Tatoeba real ×3 weight + Apertium synthetic)
- `data/val.jsonl`   — 1,000 real held-out pairs
- `data/test.jsonl`  — 1,000 real held-out pairs (eval split)

Real pairs derive from [Tatoeba](https://tatoeba.org) (CC-BY 2.0 FR — attribution
required). Synthetic pairs are high-precision Apertium ido-epo output (zero
failure markers). Eval is real-only with no source leakage into train.
