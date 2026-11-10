# Zero-shot / pivot exploration (2026-11-10)

Step A of "Ido for NLLB": what the current io↔eo model does for other languages, before any training.

**Exploration set, not a benchmark.** Ido side = v1-io-eo val+test Ido sentences (never in v1 train) with
their direct Tatoeba translation; it overlaps v1 val, is small and the sentences are short.
Greedy decoding everywhere (as in the browser build). Models: ours = `ido-epo-nllb`
(komapc/nllb-ido-epo-600m), base = `facebook/nllb-200-distilled-600M`. Runtime 94 min on CPU.

## Ido ↔ X (chrF / BLEU)

| direction | n | zero-shot (ours) | pivot via epo, ours only | pivot via epo, ours + base |
|---|---|---|---|---|
| ido→eng | 200 | 73.0 / 58.3 | 68.2 / 50.2 | 71.6 / 54.3 |
| eng→ido | 200 | 73.8 / 51.6 | 73.5 / 50.6 | 74.2 / 53.0 |
| ido→spa | 114 | 31.2 / 7.2 | 22.8 / 0.0 | 72.5 / 52.6 |
| spa→ido | 114 | 68.1 / 43.1 | 79.6 / 61.0 | 81.0 / 63.7 |
| ido→fra | 172 | 59.9 / 36.7 | 46.1 / 22.6 | 77.6 / 60.3 |
| fra→ido | 172 | 76.4 / 53.5 | 78.4 / 60.7 | 79.6 / 62.4 |
| ido→deu | 200 | 66.6 / 49.6 | 54.1 / 33.3 | 81.1 / 69.1 |
| deu→ido | 200 | 74.9 / 51.4 | 80.4 / 62.3 | 80.9 / 64.6 |

## Forgetting check: Esperanto ↔ X, fine-tuned vs base (chrF / BLEU)

| direction | n | ours | base | Δ chrF |
|---|---|---|---|---|
| epo→eng | 100 | 65.4 / 44.4 | 72.8 / 53.1 | -7.4 |
| eng→epo | 100 | 65.8 / 43.3 | 66.6 / 44.9 | -0.8 |
| epo→spa | 100 | 22.5 / 0.0 | 68.9 / 46.1 | -46.4 |
| spa→epo | 100 | 66.4 / 39.6 | 65.6 / 33.8 | +0.9 |
| epo→fra | 100 | 40.7 / 12.2 | 67.1 / 48.5 | -26.3 |
| fra→epo | 100 | 64.6 / 41.8 | 65.4 / 45.6 | -0.8 |
| epo→deu | 100 | 45.5 / 12.9 | 69.6 / 46.3 | -24.1 |
| deu→epo | 100 | 68.2 / 45.0 | 68.4 / 43.6 | -0.2 |

Conditions: *zero-shot* = ours with the source/target language tokens set directly;
*pivot, ours only* = ours ido→epo then ours epo→X (one model, what the browser could do today);
*pivot, ours + base* = the epo↔X hop by base NLLB (a second ~600 MB model).
Predictions are in `preds/`, inputs in `sets/`.
## Findings

1. **Into Ido works from every language, zero-shot** (chrF 68–76), and pivoting
   through Esperanto inside our own model is even better (73–80). No training needed
   for X→Ido.
2. **Out of Ido / Esperanto, the fine-tune forgot every target except English.**
   Esperanto→X vs base: Spanish 68.9→22.5, French 67.1→40.7, German 69.6→45.5,
   English 72.8→65.4. The failure differs by language: Spanish outputs copy the
   input (51% of ido→spa outputs have chrF ≥ 60 against the *input*), French mixes in
   Ido words ("Tu aurais pu esar felica"), German loses quality and 2/100 outputs fall
   into repetition loops (without them: 49.6, still −20).
3. **Two models work today:** ours for ido→epo, base NLLB for epo→X gives 72–81
   for all four languages, but costs a second ~600 MB model.

### Where the damage is (`weight_delta.py`, `weight_delta.log`)

99% of the squared weight change is in the shared embedding matrix (tied with the
output projection; relative change 25% vs ~2% in the encoder/decoder layers). Rows of
tokens that never occur in v1 train moved 26%, rows of seen tokens 4%: the softmax
gradient touches every vocabulary row; that the optimizer (Adafactor, wd 0.01) amplifies those
tiny gradients is a hypothesis, not measured.

### Can it be undone without training? Partly (`embedding_surgery.py`, `.log`)

Putting base rows back for the 247k tokens unseen in v1 train: ido→deu 70.0→75.5,
ido→spa 32.3→42.1, epo→spa 22.4→30.9, v1 ido→epo unchanged (73.4), v1 epo→ido 73.2→70.9
(first 60/100 rows only). Both Spanish and German improve, neither recovers (two-model pivot: 72.5 / 81.1),
so the encoder/decoder layers are damaged too. Restoring *all* embeddings except
`ido_Latn` costs ~0 on ido→epo and ~2 on epo→ido: v2 can freeze embeddings almost for free.
A single model needs retraining with replay data; its gate must include this
epo→X forgetting check, not only the v1 io↔eo test. Differences of 1–2 chrF at
n = 100–200 are noise.

### Reading vs writing Ido (`body_blend.py`, `body_blend.log`)

Embeddings = base + our `ido_Latn` row; encoder/decoder layers = base + α·(ours − base),
same small sets as above (60 rows per X, 100 per v1 direction; exploration only):

| variant | ido→spa | epo→spa | ido→deu | v1 ido→epo | v1 epo→ido |
|---|---|---|---|---|---|
| ours (v1) | 32.3 | 22.4 | 70.0 | 73.4 | 73.2 |
| α = 0.5 | 73.4 | 64.9 | 78.4 | 73.1 | 49.3 |
| α = 0.75 | 68.4 | 59.6 | 78.5 | 74.1 | 64.6 |
| α = 0.9 | 57.5 | 46.3 | 76.2 | 73.6 | 70.2 |
| **encoder ours, decoder base** | **77.1** | 64.7 | **82.1** | **75.1** | 18.7 |
| encoder ours, decoder α = 0.5 | 75.1 | 62.5 | 79.2 | 74.5 | 52.6 |

**Reading Ido lives in the encoder; writing Ido lives in the decoder, and the decoder
change is what destroyed the other targets.** Our encoder with the untouched base decoder
translates Ido into every language tested (ido→epo included) at least as well as the
two-model pivot, while only our decoder can write Ido. Consequences:
- one package with shared embeddings (base + `ido_Latn`), our encoder and two decoders
  (base for Ido→X, ours for X→Ido) is ~770M parameters instead of two 615M models;
- for v2 training: Ido-as-source data only needs the encoder; Ido-as-target data needs
  the decoder, and that is where replay of other targets is required.
