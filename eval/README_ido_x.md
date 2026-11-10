# Ido ↔ English, Spanish, French, German: baseline (2026-11-10)

Browser build [komapc/nllb-ido-multi-onnx](https://huggingface.co/komapc/nllb-ido-multi-onnx) (variant A),
transformers.js q8, greedy decoding, as shipped on tradukilo.lipu.dev. Sets: `datasets/eval-ido-x-v1`
(dev / test) and v1-io-eo test. chrF with 95% bootstrap intervals (`eval/score_ido_decoder.py`, 1,000
samples). Predictions: `eval/preds/eval-ido-x-v1/browser-va-q8.jsonl`.

- **base** = the shipped path. Ido→X: read decoder. X→Ido: X→epo (read decoder), then epo→Ido (Ido decoder).
- **direct** = X→Ido with the Ido decoder alone, no Esperanto step. Δ is direct − base on the same sentences.

```
set        src  model       n   chrF  95% CI         Δ vs base (95% CI)
evx-dev    deu  base       99   78.4  [73.1, 83.4]
evx-dev    deu  direct     99   72.4  [66.9, 77.5]   -6.1 [-10.7, -1.9]
evx-dev    eng  base      238   74.7  [71.6, 78.4]
evx-dev    eng  direct    238   75.7  [72.3, 79.3]   +0.9 [-1.3, +3.2]
evx-dev    fra  base       76   78.2  [72.5, 84.3]
evx-dev    fra  direct     76   73.5  [67.5, 79.4]   -4.7 [-10.0, +0.4]
evx-dev    ido->deu base      108   77.6  [72.3, 82.4]
evx-dev    ido->eng base      289   76.4  [73.3, 79.7]
evx-dev    ido->fra base       84   75.8  [70.0, 81.4]
evx-dev    ido->spa base       57   80.8  [73.8, 87.6]
evx-dev    spa  base       52   77.5  [70.5, 84.8]
evx-dev    spa  direct     52   69.0  [60.8, 76.9]   -8.5 [-15.8, -1.6]
evx-test   deu  base      589   76.3  [74.6, 78.1]
evx-test   deu  direct    589   70.0  [67.9, 72.0]   -6.3 [-7.9, -4.9]
evx-test   eng  base      755   65.9  [64.1, 67.7]
evx-test   eng  direct    755   65.5  [63.7, 67.3]   -0.4 [-1.3, +0.6]
evx-test   fra  base      249   74.9  [71.6, 78.1]
evx-test   fra  direct    249   71.8  [68.4, 75.2]   -3.1 [-5.2, -0.9]
evx-test   ido->deu base      597   78.7  [76.6, 80.6]
evx-test   ido->eng base      803   71.2  [69.5, 72.9]
evx-test   ido->fra base      256   74.1  [70.4, 77.6]
evx-test   ido->spa base      176   68.7  [65.3, 72.5]
evx-test   spa  base      167   64.0  [60.5, 67.5]
evx-test   spa  direct    167   54.7  [51.2, 58.1]   -9.3 [-11.7, -6.7]
v1-test    epo  base      500   72.2  [69.7, 74.7]
v1-test    ido->epo base      500   75.4  [73.0, 77.9]
```

**Reading it**
- The pivot through Esperanto beats direct X→Ido for German, Spanish and French on test (−6.3, −9.3,
  −3.1 for direct, intervals clear of 0). For English the two are level, so the site's choice to pivot holds.
- **Test is harder than dev, and than v1 test, mostly because of sentence length.**
  v1 drew its eval split from the shortest third of the pairs (`corpus/make_dataset.py`), so
  sentences that come from v1 test average about 16 characters. Sentences that never reached v1
  average 26–36. Test by origin (base path):

```
direction origin           n  chrF Ido chars
deu       from v1 test    89  83.8        16
deu       new            500  75.5        26
eng       from v1 test   246  76.8        16
eng       new            509  63.5        31
fra       from v1 test    82  80.4        16
fra       new            167  73.5        27
ido->deu  from v1 test    93  84.1        16
ido->deu  new            504  78.0        26
ido->eng  from v1 test   283  75.8        16
ido->eng  new            520  70.1        31
ido->fra  from v1 test    85  75.9        16
ido->fra  new            171  73.6        27
ido->spa  from v1 test    57  73.1        17
ido->spa  new            119  67.8        36
spa       from v1 test    53  81.2        17
spa       new            114  60.7        36
```

  The v1-io-eo test score (epo→ido 72.2) is likewise a short-sentence score, so a v2 decoder should
  be judged on the X→Ido test rows (through Esperanto) as well, not on v1 test alone.
- **French and Spanish sets are small** (±3–4 chrF intervals), so decisions rest on epo (v1 test), eng and deu.
- **Esperanto→Ido** (72.2) is the step every X→Ido translation ends with. It is the target of
  `finetune_ido_decoder.ipynb` (back-translation, v2 Ido decoder).
