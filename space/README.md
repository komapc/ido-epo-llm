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
