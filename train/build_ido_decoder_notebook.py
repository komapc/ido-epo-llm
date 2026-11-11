#!/usr/bin/env python3
"""Generate finetune_ido_decoder.ipynb (repo root): v2 of the Ido-writing decoder,
trained with back-translation on top of the frozen variant-A encoder.

Edit the cells here, then: python3 train/build_ido_decoder_notebook.py
Smoke-test the same cells on CPU (tiny subsets, 3 steps):
    python3 train/build_ido_decoder_notebook.py --smoke
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MONO = os.environ.get("MONO_DATASET", "v2-ido-mono")

INTRO = f"""# v2 Ido decoder: back-translation on a frozen encoder

The browser build ([komapc/nllb-ido-multi-onnx](https://huggingface.co/komapc/nllb-ido-multi-onnx))
has one encoder and two decoders. This notebook retrains only the decoder that **writes Ido**
(`onnx_ido/`), so a new version changes 375 MB and the rest of the build stays as it is.

- **Encoder:** frozen and exactly as shipped: the v1 encoder with base NLLB embeddings plus the
  trained `ido_Latn` row.
- **Trained:** the v1 decoder layers. Its tied embeddings stay as in v1 unless `TRAIN_EMB=1`.
- **Data:** every pair targets Ido.
  - v1 real Tatoeba epo→ido pairs (weight 3) and the v1 Apertium synthetic pairs (weight 1);
  - back-translated pairs: real Ido sentences from `datasets/{MONO}/` (Ido Wikipedia and
    FineWeb-2), translated to Esperanto by the read path of the same build (ido→epo, chrF 75).
    Their Ido side is genuine text, which is what the decoder learns to write.
- **Model selection:** best checkpoint by loss on v1 val epo→ido.
- **Outputs**, all with greedy decoding as in the browser, before (`base`) and after (`v2`) training:
  - v1 test epo→ido;
  - `eval-ido-x-v1` dev/test X→ido through Esperanto.

## Kaggle (or Colab)
1. **File → Import Notebook** → `https://github.com/komapc/ido-epo-llm/blob/main/finetune_ido_decoder.ipynb`
2. Accelerator **GPU T4** (one is enough), **Internet On**. **Save & Run All**. About 2 h.
   The first prediction cell prints the base scores next to the expected ones; if they are far off,
   stop the run.
3. Download from the Output tab:
   - `preds.jsonl`;
   - `bt_pairs.jsonl` (the back-translations, for the record);
   - `ido-decoder-v2.zip` (fp16 model; only its decoder and embeddings are new).
4. Score locally: `python3 eval/score_ido_decoder.py preds.jsonl`.

Knobs (environment variables, or edit the cells): `BT_RATIO` (back-translated examples per real
weighted example, default 1.0), `TRAIN_EMB=1` (also train the embeddings; **untested**, the default
keeps them frozen), `REPO_BRANCH` (clone a branch other than `main`).

License: NLLB-200 weights are CC-BY-NC-4.0; FineWeb-2 is ODC-By; Ido Wikipedia is CC BY-SA.
"""

CELLS = [
    # 1 install
    """%%capture
!pip install -q -U transformers accelerate datasets sentencepiece sacrebleu""",

    # 2 repo + data paths
    """import os, subprocess, json, random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
REPO_DIR = os.environ.get("REPO_DIR", "ido-epo-llm")
REPO_BRANCH = os.environ.get("REPO_BRANCH", "main")   # a PR branch, to run before it is merged
if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_BRANCH,
                    "https://github.com/komapc/ido-epo-llm.git", REPO_DIR], check=True)
DS = os.path.join(REPO_DIR, "datasets")
def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
v1 = {s: load_jsonl(f"{DS}/v1-io-eo/{s}.jsonl") for s in ("train", "val", "test")}
evx = {s: load_jsonl(f"{DS}/eval-ido-x-v1/{s}.jsonl") for s in ("dev", "test")}
import gzip
with gzip.open(f"{DS}/%s/mono.jsonl.gz", "rt", encoding="utf-8") as f:
    mono = [json.loads(l) for l in f if l.strip()]
print({k: len(v) for k, v in v1.items()}, {k: len(v) for k, v in evx.items()}, "mono", len(mono))""" % MONO,

    # 3 models
    """import torch
SMOKE = os.environ.get("SMOKE") == "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE != "cuda" and not SMOKE:
    raise RuntimeError("No GPU attached: Settings -> Accelerator -> GPU T4, then run all again.")
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

OURS, BASE = "komapc/nllb-ido-epo-600m", "facebook/nllb-200-distilled-600M"
MAX_LEN = 128
tok = AutoTokenizer.from_pretrained(OURS)
write = AutoModelForSeq2SeqLM.from_pretrained(OURS, dtype=torch.float32)   # v1: decoder to train
read = AutoModelForSeq2SeqLM.from_pretrained(OURS, dtype=torch.float32)    # becomes the read model
base = AutoModelForSeq2SeqLM.from_pretrained(BASE, dtype=torch.float32)
LANG = {l: tok.convert_tokens_to_ids(f"{l}_Latn") for l in ("ido", "epo", "eng", "spa", "fra", "deu")}

# Read model, as space/build_read_model.py: our encoder, base decoder, base embeddings + our ido row.
with torch.no_grad():
    E = read.model.shared.weight
    n = min(E.shape[0], base.model.shared.weight.shape[0])
    keep = E[LANG["ido"]].clone()
    E[:n] = base.model.shared.weight[:n]
    E[LANG["ido"]] = keep
    dec = {k: v for k, v in base.model.decoder.state_dict().items() if "embed_tokens" not in k}
    missing, unexpected = read.model.decoder.load_state_dict(dec, strict=False)
    assert not unexpected and all("embed_tokens" in k for k in missing), (missing, unexpected)
del base
read.eval().requires_grad_(False).to(DEVICE)
if DEVICE == "cuda":
    read.half()
encoder = read.model.encoder        # the shipped encoder; frozen
# write's own encoder never runs (states come from `encoder`). Freezing it also freezes the tied
# embeddings / lm_head, since they are one tensor; TRAIN_EMB=1 trains them too.
write.model.encoder.requires_grad_(False)
if os.environ.get("TRAIN_EMB") == "1":
    write.model.shared.requires_grad_(True)
write.generation_config.max_length = None
read.generation_config.max_length = None
write.to(DEVICE)
print("trainable params:", sum(p.numel() for p in write.parameters() if p.requires_grad) / 1e6, "M")""",

    # 4 helpers: encode + generate
    """EOS = tok.eos_token_id
def enc(text, lang):
    return [LANG[lang]] + tok(text, add_special_tokens=False)["input_ids"][:MAX_LEN - 2] + [EOS]

@torch.no_grad()
def translate(model, texts, src, tgt, beams=1, batch=48):
    \"\"\"Translate with `model`'s decoder; the encoder is always the shipped one.\"\"\"
    model.eval()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [None] * len(texts)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        feats = tok.pad([{"input_ids": enc(texts[i], src)} for i in idx], return_tensors="pt").to(DEVICE)
        with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
            states = encoder(input_ids=feats["input_ids"], attention_mask=feats["attention_mask"])
            gen = model.generate(encoder_outputs=states, attention_mask=feats["attention_mask"],
                                 forced_bos_token_id=LANG[tgt], num_beams=beams, max_new_tokens=MAX_LEN)
        for i, o in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            out[i] = o.strip()
    return out

def predict_into_ido(model, tag):
    \"\"\"v1 test epo->ido, and eval-ido-x-v1 X->ido through Esperanto (greedy, as in the browser).\"\"\"
    rows = [{"set": "v1-test", "src_lang": "epo", "input": r["input"], "ref": r["output"]}
            for r in v1["test"] if r["tgt_lang"] == "ido"]
    for split in ("dev", "test"):
        rows += [{"set": f"evx-{split}", "src_lang": r["src_lang"], "input": r["input"], "ref": r["output"]}
                 for r in evx[split] if r["tgt_lang"] == "ido"]
    if SMOKE:
        rows = rows[:3] + [r for r in rows if r["set"] == "evx-dev"][:3]
    for r in rows:
        r["model"] = tag
    for x in ("eng", "spa", "fra", "deu"):
        sel = [r for r in rows if r["src_lang"] == x]
        if not sel:
            continue
        if "pivot" not in sel[0]:   # X->epo is the read model's job and the same for base and v2
            for r, p in zip(sel, translate(read, [r["input"] for r in sel], x, "epo")):
                r["pivot"] = p
    for r in rows:
        r.setdefault("pivot", r["input"])
    for r, p in zip(rows, translate(model, [r["pivot"] for r in rows], "epo", "ido")):
        r["output"] = p
    return rows

import sacrebleu
from collections import defaultdict
def report(preds):
    groups = defaultdict(lambda: defaultdict(lambda: ([], [])))
    for r in preds:
        h, refs = groups[(r["set"], r["src_lang"])][r["model"]]
        h.append(r["output"]); refs.append(r["ref"])
    models = sorted({r["model"] for r in preds}, key=lambda m: m != "base")
    print(f"{'set':10} {'src':4} {'n':>4} " + " ".join(f"{m:>6}" for m in models))
    for (s, l), m in sorted(groups.items()):
        sc = [sacrebleu.corpus_chrf(*([m[k][0], [m[k][1]]])).score if k in m else 0 for k in models]
        print(f"{s:10} {l:4} {len(m[models[0]][0]):4} " + " ".join(f"{x:6.1f}" for x in sc))

base_preds = predict_into_ido(write, "base")
report(base_preds)
# Expected (browser build, q8, eval/README_ido_x.md): v1-test epo 72.2 (PyTorch fp32: 72.5);
# evx-test eng 65.9, deu 76.3, fra 74.9, spa 64.0. Far lower means the encoder setup is broken: stop here.""",

    # 5 back-translation
    """import re
BT_RATIO = float(os.environ.get("BT_RATIO", "1.0"))
real = [r for r in v1["train"] if r["tgt_lang"] == "ido"]
real_weighted = sum(int(r.get("weight", 1)) for r in real)
n_bt = min(len(mono), int(BT_RATIO * real_weighted))
if SMOKE:
    real, n_bt = real[:40], 24
rng = random.Random(42)
pool = rng.sample(mono, n_bt)
ido_texts = [r["text"] for r in pool]
epo_texts = translate(read, ido_texts, "ido", "epo", beams=4)

def norm(s):
    return re.sub(r"\\W+", " ", s.lower()).strip()
def looping(s):
    w = s.split()
    grams = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    return len(grams) > 3 and len(set(grams)) < 0.6 * len(grams)
bt, dropped = [], {"copy": 0, "ratio": 0, "loop": 0, "empty": 0}
for src_row, i, e in zip(pool, ido_texts, epo_texts):
    if not e:
        dropped["empty"] += 1
    elif norm(e) == norm(i):
        dropped["copy"] += 1
    elif not 0.5 <= len(e) / max(len(i), 1) <= 2.0:
        dropped["ratio"] += 1
    elif looping(e):
        dropped["loop"] += 1
    else:
        bt.append({"input": e, "output": i, "src_lang": "epo", "tgt_lang": "ido",
                   "provenance": "backtranslation", "source": src_row["source"], "weight": 1})
with open("bt_pairs.jsonl", "w", encoding="utf-8") as f:
    for r in bt:
        f.write(json.dumps(r, ensure_ascii=False) + "\\n")
print(f"real {len(real)} rows ({real_weighted} weighted); back-translated kept {len(bt)}/{n_bt}, dropped {dropped}")
for r in bt[:3]:
    print(r["input"], "=>", r["output"])""",

    # 6 datasets
    """from datasets import Dataset
def featurize(rows):
    out = []
    for r in rows:
        ex = {"input_ids": enc(r["input"], r["src_lang"]), "labels": enc(r["output"], "ido")}
        out.extend([ex] * int(r.get("weight", 1)))
    return out
train_ds = Dataset.from_list(featurize(real + bt)).shuffle(seed=42)
dev_rows = [r for r in v1["val"] if r["tgt_lang"] == "ido"][: (16 if SMOKE else None)]
val_ds = Dataset.from_list([{"input_ids": enc(r["input"], "epo"), "labels": enc(r["output"], "ido")} for r in dev_rows])
print(len(train_ds), "train examples,", len(val_ds), "val")""",

    # 7 train
    """from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

class FrozenEncoderTrainer(Seq2SeqTrainer):
    \"\"\"Encoder states come from the shipped (frozen) encoder, not from `write`'s own.\"\"\"
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        with torch.no_grad():
            states = encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        hidden = states.last_hidden_state.float()
        out = model(encoder_outputs=(hidden,), attention_mask=inputs["attention_mask"], labels=inputs["labels"])
        return (out.loss, out) if return_outputs else out.loss

args = Seq2SeqTrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=16, gradient_accumulation_steps=2, per_device_eval_batch_size=32,
    learning_rate=5e-5, num_train_epochs=1, warmup_steps=100, lr_scheduler_type="linear",
    weight_decay=0.01, optim="adafactor", fp16=(DEVICE == "cuda"),
    eval_strategy="steps", eval_steps=400, save_strategy="steps", save_steps=400,
    save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
    logging_steps=100, report_to="none", seed=42, remove_unused_columns=False,
)
if SMOKE:
    args.max_steps, args.eval_steps, args.save_steps, args.logging_steps = 3, 3, 3, 1
    args.per_device_train_batch_size, args.gradient_accumulation_steps, args.per_device_eval_batch_size = 2, 1, 4
trainer = FrozenEncoderTrainer(
    model=write, args=args, train_dataset=train_ds, eval_dataset=val_ds,
    data_collator=DataCollatorForSeq2Seq(tok, model=write, label_pad_token_id=-100),
    processing_class=tok,
)
print("val loss before:", trainer.evaluate()["eval_loss"])
trainer.train()
print("best:", trainer.state.best_model_checkpoint, trainer.state.best_metric)""",

    # 8 save
    """write = trainer.model.eval()
write.half().save_pretrained("ido-decoder-v2")
tok.save_pretrained("ido-decoder-v2")
if not SMOKE:
    os.system("zip -r -0 -q ido-decoder-v2.zip ido-decoder-v2 && echo saved ido-decoder-v2.zip")
write.float()""",

    # 9 predict + quick score
    """v2_preds = predict_into_ido(write, "v2")
with open("preds.jsonl", "w", encoding="utf-8") as f:
    for r in base_preds + v2_preds:
        f.write(json.dumps(r, ensure_ascii=False) + "\\n")
report(base_preds + v2_preds)
print("wrote preds.jsonl")""",
]


def build():
    def cell(kind, src):
        c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        return c
    nb = {
        "cells": [cell("markdown", INTRO)] + [cell("code", s) for s in CELLS],
        "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = ROOT / "finetune_ido_decoder.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("wrote", out)


def smoke():
    """Run cells 2-9 on CPU against this checkout (no pip, no clone)."""
    os.environ["SMOKE"] = "1"
    os.environ["REPO_DIR"] = str(ROOT)
    workdir = Path(os.environ.get("SMOKE_DIR", "/tmp/ido_dec_smoke"))
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)
    g = {}
    for i, src in enumerate(CELLS[1:], start=2):
        print(f"===== cell {i} =====", flush=True)
        exec(compile(src, f"cell{i}", "exec"), g)


if __name__ == "__main__":
    smoke() if "--smoke" in sys.argv else build()
