---
license: cc-by-nc-4.0
language:
- io
- eo
- en
- es
- fr
- de
base_model:
- komapc/nllb-ido-epo-600m
- facebook/nllb-200-distilled-600M
pipeline_tag: translation
library_name: transformers.js
tags:
- translation
- ido
- esperanto
- nllb
- onnx
---

# NLLB Ido ↔ Esperanto, English, Spanish, French, German (browser build, ONNX int8)

This is a [transformers.js](https://huggingface.co/docs/transformers.js) build for translating
between **Ido** and Esperanto, English, Spanish, French and German in the browser. It needs no
retraining: it combines [komapc/nllb-ido-epo-600m](https://huggingface.co/komapc/nllb-ido-epo-600m)
with the untouched [NLLB-200 distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M).
It is built for the neural engine of [tradukilo.lipu.dev](https://tradukilo.lipu.dev/).

## Why two decoders

The Ido↔Esperanto fine-tune learned to *read* Ido in the encoder and to *write* Ido in the
decoder. While learning to write Ido, its decoder forgot most other languages: Esperanto→Spanish
dropped from chrF 68.9 to 22.5. So this build has one encoder and two decoders:

| file | weights | use |
|---|---|---|
| `onnx/encoder_model_quantized.onnx` (237 MB) | fine-tuned encoder; base NLLB embeddings plus the trained `ido_Latn` row | every direction |
| `onnx/decoder_model_merged_quantized.onnx` (375 MB) | base NLLB decoder | anything → epo / eng / spa / fra / deu |
| `onnx_ido/decoder_model_merged_quantized.onnx` (375 MB) | fine-tuned decoder | anything → Ido |
| `onnx_ido/encoder_model_quantized.onnx` (327 bytes) | placeholder that outputs zeros | lets transformers.js load `onnx_ido/` as a model; replace its session with the real encoder (see below) |

- **Downloads:** 612 MB for Ido → other languages; the Ido decoder adds 375 MB.
- **Vocabulary:** pruned to the 79,460 tokens that text in the six languages uses.
- **Into Ido:** quality is best when you pivot through Esperanto (X → epo → Ido).

## Scores

chrF, transformers.js q8, greedy decoding:

| direction | ido-epo v1 browser build | this build |
|---|---|---|
| ido→epo (held-out test, 500) | ≈74 | **75.4** |
| epo→ido (held-out test, 500) | ≈72 | 72.2 |
| ido→eng / spa / fra / deu | 73.0 / 31.2 / 59.9 / 66.6 | **76.4 / 77.2 / 75.8 / 80.2** |
| eng / spa / fra / deu → ido, through epo | – | **75.4 / 79.2 / 78.8 / 79.2** |

The Ido↔X sets are small (114–200 short Tatoeba sentences) and overlap the v1 validation
split, so read them as indicative, not as a benchmark.

## Usage (transformers.js)

```js
import { AutoTokenizer, AutoModelForSeq2SeqLM, Tensor } from "@huggingface/transformers";

const id = "komapc/nllb-ido-multi-onnx";
const tok = await AutoTokenizer.from_pretrained(id);
const fromIdo = await AutoModelForSeq2SeqLM.from_pretrained(id, { dtype: "q8" });
// Only needed for translating into Ido:
const toIdo = await AutoModelForSeq2SeqLM.from_pretrained(id, { dtype: "q8", subfolder: "onnx_ido" });
await toIdo.sessions.model.release?.();
toIdo.sessions.model = fromIdo.sessions.model;   // share the real encoder

const L = (c) => tok.convert_tokens_to_ids(c);   // "ido_Latn", "epo_Latn", "spa_Latn", ...
async function run(model, text, src, tgt) {
  const ids = [L(src), ...tok.encode(text, { add_special_tokens: false }), tok.eos_token_id];
  const shape = [1, ids.length];
  const out = await model.generate({
    inputs: new Tensor("int64", BigInt64Array.from(ids.map(BigInt)), shape),
    attention_mask: new Tensor("int64", new BigInt64Array(ids.length).fill(1n), shape),
    forced_bos_token_id: L(tgt), max_new_tokens: 256,
  });
  return tok.decode(out[0], { skip_special_tokens: true });
}
await run(fromIdo, "Me amas lektar libri.", "ido_Latn", "spa_Latn");          // Ido → Spanish
const eo = await run(fromIdo, "Me gusta leer libros.", "spa_Latn", "epo_Latn");
await run(toIdo, eo, "epo_Latn", "ido_Latn");                                  // Spanish → Ido via epo
```

Build scripts and recipe: `space/` in [komapc/ido-epo-llm](https://github.com/komapc/ido-epo-llm)
(`build_read_model.py`, `prune_vocab.py --langs`, `make_stub_encoder.py`).

**License:** CC-BY-NC 4.0, inherited from NLLB-200. For attribution, see the base model cards.
