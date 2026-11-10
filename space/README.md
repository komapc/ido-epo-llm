# Demo and browser build

- **Live demo:** [hf.co/spaces/komapc/ido-epo-translator](https://huggingface.co/spaces/komapc/ido-epo-translator)
  is a static Space, and translation runs in the visitor's browser through transformers.js.
- **Model it loads:** [komapc/nllb-ido-epo-600m-onnx](https://huggingface.co/komapc/nllb-ido-epo-600m-onnx).

| path | what |
|---|---|
| `static/` | the Space itself (`index.html` + `worker.js` + Space README). Upload these three files. |
| `prune_vocab.py` | shrinks the NLLB vocab from 256k to the ~72k tokens IO/EO text uses. The result is lossless on the test set. |
| `gradio/` | server-side Gradio version of the demo. It isn't used, because free Gradio/Docker Spaces now need HF PRO. |

## Rebuilding the browser model

Starting point: the fine-tuned model in `models/ido-epo-nllb/` (Kaggle `ido-epo-nllb.zip`).

```bash
# 1. prune the vocabulary (env: transformers 5.x)
python space/prune_vocab.py models/ido-epo-nllb models/ido-epo-nllb-pruned

# 2. export to ONNX (separate env: optimum-onnx[onnxruntime] + CPU torch; it pins transformers 4.x)
optimum-cli export onnx --model models/ido-epo-nllb-pruned \
    --task text2text-generation-with-past --opset 17 models/onnx-fp32

# 3. int8 dynamic quantization. EnableSubgraph is required, otherwise the merged
#    decoder's If-branches stay fp32.
python - <<'EOF'
from onnxruntime.quantization import quantize_dynamic, QuantType
for n, sub in (("encoder_model", False), ("decoder_model_merged", True)):
    quantize_dynamic(f"models/onnx-fp32/{n}.onnx", f"models/web/onnx/{n}_quantized.onnx",
                     weight_type=QuantType.QUInt8, extra_options={"EnableSubgraph": sub})
EOF
cp models/onnx-fp32/{config,generation_config}.json models/ido-epo-nllb-pruned/tokenizer*.json models/web/
```

Measured on 2026-11-09 against `datasets/v1-io-eo/test.jsonl`, with greedy decoding in transformers.js 4.3.1 under Node:
**chrF 73.0 / BLEU 53.3**. The full model with beam 4 scores 73.4 / 53.9. The download is 562 MB.

## Split-decoder build: Ido ↔ eng/spa/fra/deu (variant A, 2026-11-10)

The v1 fine-tune can read Ido but forgot how to write anything except Ido/Esperanto
(see `eval/explore/2026-11-10/results.md`). Reading Ido lives in the encoder and writing
Ido in the decoder, so this build ships **one encoder and two decoders**, with no new training:

| file | weights | used for |
|---|---|---|
| `onnx/encoder_model_quantized.onnx` (237 MB) | our encoder; base NLLB embeddings + our `ido_Latn` row | every direction |
| `onnx/decoder_model_merged_quantized.onnx` (375 MB) | untouched base NLLB decoder | anything → epo/eng/spa/fra/deu |
| `onnx/decoder_ido_model_merged_quantized.onnx` (375 MB) | v1 decoder with its v1 embeddings | → Ido only |

Vocab is pruned to the tokens of six languages (79,460, vs 72k for io/eo only); both
models are pruned with the same `--langs`, so their tokenizers are byte-identical.
X→Ido works best as a pivot X→epo (read decoder) → Ido (Ido decoder).

```bash
# envs as above (transformers 5.x for steps 1–2, optimum env for 3–4)
python space/build_read_model.py models/ido-epo-nllb models/va-read-full
python space/prune_vocab.py models/va-read-full  models/va-read-pruned  --langs ido epo eng spa fra deu
python space/prune_vocab.py models/ido-epo-nllb models/va-write-pruned --langs ido epo eng spa fra deu
for m in read write; do optimum-cli export onnx --model models/va-$m-pruned \
    --task text2text-generation-with-past --opset 17 models/va-$m-onnx32; done
python - <<'PY'
from onnxruntime.quantization import quantize_dynamic, QuantType
for src, dst, sub in [("va-read-onnx32/encoder_model", "encoder_model", False),
                      ("va-read-onnx32/decoder_model_merged", "decoder_model_merged", True),
                      ("va-write-onnx32/decoder_model_merged", "decoder_ido_model_merged", True)]:
    quantize_dynamic(f"models/{src}.onnx", f"models/web-multi/onnx/{dst}_quantized.onnx",
                     weight_type=QuantType.QUInt8, extra_options={"EnableSubgraph": sub})
PY
cp models/va-read-onnx32/{config,generation_config}.json models/va-read-pruned/tokenizer*.json models/web-multi/
```

transformers.js loads `encoder_model` + `decoder_model_merged` by name, so for scoring
`view-read/` and `view-write/` are folders of symlinks (shared encoder, one decoder each);
`space/eval_web.mjs` scores them under Node. Measured 2026-11-10, q8, greedy, ~250 ms/sentence
per pass on CPU (Ido↔X rows are the exploration sets, which overlap v1 val; not a benchmark):

| direction | v1 browser build | split build |
|---|---|---|
| ido→epo (v1 test, 500) | ≈74 | **75.4** |
| epo→ido (v1 test, 500) | ≈72 | 72.2 |
| ido→eng / spa / fra / deu | 73.0 / 31.2 / 59.9 / 66.6 | **76.4 / 77.2 / 75.8 / 80.2** |
| eng / spa / fra / deu → ido, direct | 73.8 / 68.1 / 76.4 / 74.9 | 73.9 / 68.0 / 75.9 / 73.6 |
| eng / spa / fra / deu → ido, via epo | – | **75.4 / 79.2 / 78.8 / 79.2** |

Download: 612 MB for Ido→any (encoder + one decoder), 987 MB for both directions.
Not published yet; the tradukilo worker would need to load the two decoders against one
cached encoder.
