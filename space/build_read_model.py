#!/usr/bin/env python3
"""Build the "read Ido" model of the split-decoder browser build (variant A):
our fine-tuned encoder + the untouched base NLLB decoder, with the base embedding
matrix plus our trained `ido_Latn` row (embeddings are tied, so the decoder can
still write every NLLB language). See eval/explore/2026-11-10/results.md.

Usage: python space/build_read_model.py OURS_DIR OUT_DIR [--base facebook/nllb-200-distilled-600M]
Then prune it with the same --langs as the write model so token ids agree.
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("ours", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--base", default="facebook/nllb-200-distilled-600M")
args = ap.parse_args()

ours = AutoModelForSeq2SeqLM.from_pretrained(args.ours, dtype=torch.float32)
base = AutoModelForSeq2SeqLM.from_pretrained(args.base, dtype=torch.float32)

ido = AutoTokenizer.from_pretrained(args.ours).convert_tokens_to_ids("ido_Latn")

with torch.no_grad():
    E = ours.model.shared.weight
    n = min(E.shape[0], base.model.shared.weight.shape[0])
    keep_ido = E[ido].clone()
    E[:n] = base.model.shared.weight[:n]
    E[ido] = keep_ido
    dec = {k: v for k, v in base.model.decoder.state_dict().items() if "embed_tokens" not in k}
    missing, unexpected = ours.model.decoder.load_state_dict(dec, strict=False)
    assert not unexpected and all("embed_tokens" in k for k in missing), (missing, unexpected)
assert ours.lm_head.weight.data_ptr() == ours.model.shared.weight.data_ptr(), "lm_head must stay tied"
ours.save_pretrained(args.out)
AutoTokenizer.from_pretrained(args.ours).save_pretrained(args.out)
print("saved", args.out)
