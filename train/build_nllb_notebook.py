#!/usr/bin/env python3
"""Generate finetune_nllb.ipynb (repo root) from the cell sources below.

Edit the cells here, then: python3 train/build_nllb_notebook.py
Smoke-test the same cells on CPU: SMOKE=1 python3 train/build_nllb_notebook.py --smoke
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mtlib.paths import DEFAULT_DATASET  # noqa: E402

# Split under datasets/ that the notebook trains and predicts on.
DATASET = os.environ.get("DATASET", DEFAULT_DATASET)

INTRO = f"""# Fine-tune NLLB-200 (600M) for Ido↔Esperanto

Second candidate next to the Llama-3.2-3B QLoRA notebook (`finetune_colab.ipynb`).
NLLB is a dedicated encoder-decoder MT model; it already knows Esperanto
(`epo_Latn`). Ido is not one of its 200 languages, so we add a new language
token `ido_Latn`, initialised from the mean of Esperanto/Italian/Spanish.

Full fine-tune (not LoRA) in **fp16 AMP with fp32 master weights** — the stock
T4 path, no bf16 emulation, so it is far faster than the Llama run.
Same data and same test split as the Llama notebook, so the scores compare 1:1.

## Kaggle
1. **File → Import Notebook** → `https://github.com/komapc/ido-epo-llm/blob/main/finetune_nllb.ipynb`
2. Settings: Accelerator = **GPU T4 x2**, **Internet = On**.
3. **Save Version → Save & Run All (Commit)**. ~2–3h.
4. Output tab → download `preds.jsonl` (+ `ido-epo-nllb.zip`, the fp16 model).

Dataset: `datasets/{DATASET}/` of the repo.
Score locally: `python3 eval/eval_chrf.py --dataset {DATASET} --apertium --pred preds.jsonl`.

License note: NLLB-200 weights are CC-BY-NC-4.0 (non-commercial).
"""

CELLS = [
    # 1 install
    """%%capture
# Kaggle already ships torch/transformers; make sure the NLLB tokenizer deps exist.
!pip install -q -U transformers accelerate datasets sentencepiece""",

    # 2 data
    """# --- Pull dataset from the repo ---------------------------------------------
import os, subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # one GPU; Trainer's DataParallel adds nothing here

REPO_URL = "https://github.com/komapc/ido-epo-llm.git"
REPO_DIR = "ido-epo-llm"
if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL], check=True)
DATA_DIR = os.path.join(REPO_DIR, "datasets", "%s")
for split in ("train", "val", "test"):
    p = os.path.join(DATA_DIR, f"{split}.jsonl")
    assert os.path.exists(p), f"missing {p}"
    print(split, sum(1 for _ in open(p)), "rows")""" % DATASET,

    # 3 model + language token
    """import os, json, torch
SMOKE = os.environ.get("SMOKE") == "1"     # CPU smoke test: tiny subset, 3 steps
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE != "cuda" and not SMOKE:
    raise RuntimeError("No GPU attached. Settings -> Accelerator -> 'GPU T4 x2', then re-run all.")
if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

BASE = "facebook/nllb-200-distilled-600M"
MAX_LEN = 128
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE)   # fp32 master weights; fp16 only via AMP

# Ido is not an NLLB language: add a language token for it.
if "ido_Latn" not in tok.get_vocab():
    tok.add_tokens(["ido_Latn"], special_tokens=True)
old_rows = model.get_input_embeddings().num_embeddings
model.resize_token_embeddings(max(len(tok), old_rows))   # never shrink
LANG_ID = {"epo": tok.convert_tokens_to_ids("epo_Latn"), "ido": tok.convert_tokens_to_ids("ido_Latn")}
seeds = [tok.convert_tokens_to_ids(t) for t in ("epo_Latn", "ita_Latn", "spa_Latn")]
with torch.no_grad():
    for w in {id(model.get_input_embeddings().weight): model.get_input_embeddings().weight,
              id(model.get_output_embeddings().weight): model.get_output_embeddings().weight}.values():
        w[LANG_ID["ido"]] = w[seeds].mean(0)
assert model.config.decoder_start_token_id == tok.eos_token_id, model.config.decoder_start_token_id
print("lang ids:", LANG_ID, "| vocab rows:", old_rows, "->", model.get_input_embeddings().num_embeddings)""",

    # 4 dataset
    """from datasets import Dataset

def load_jsonl(name):
    return [json.loads(l) for l in open(os.path.join(DATA_DIR, name), encoding="utf-8") if l.strip()]

EOS = tok.eos_token_id
def enc(text, lang):
    # NLLB (non-legacy) layout: [lang_code] tokens </s>, built by hand so it does
    # not depend on the tokenizer knowing 'ido_Latn' as a src_lang.
    return [LANG_ID[lang]] + tok(text, add_special_tokens=False)["input_ids"][:MAX_LEN - 2] + [EOS]

def featurize(rows):
    out = []
    for r in rows:
        ex = {"input_ids": enc(r["input"], r["src_lang"]), "labels": enc(r["output"], r["tgt_lang"])}
        out.extend([ex] * int(r.get("weight", 1)))   # real pairs (w=3) seen 3x vs synthetic (w=1)
    return out

train_rows, val_rows = load_jsonl("train.jsonl"), load_jsonl("val.jsonl")
if SMOKE:
    train_rows, val_rows = train_rows[:48], val_rows[:16]
train_ds = Dataset.from_list(featurize(train_rows)).shuffle(seed=42)
val_ds = Dataset.from_list([{"input_ids": enc(r["input"], r["src_lang"]),
                             "labels": enc(r["output"], r["tgt_lang"])} for r in val_rows])
print(len(train_rows), "unique ->", len(train_ds), "weighted train examples;", len(val_ds), "val")
print(tok.convert_ids_to_tokens(train_ds[0]["input_ids"]))
print(tok.convert_ids_to_tokens(train_ds[0]["labels"]))""",

    # 5 train
    """from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

# ~87k weighted examples / effective batch 32 ≈ 2.7k steps per epoch.
# Eval/save every 500 steps; the best checkpoint by val loss is restored at the end,
# so over-training in epoch 2 costs nothing. If CUDA OOM: batch 8 x accumulation 4.
args = Seq2SeqTrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=16, gradient_accumulation_steps=2,
    per_device_eval_batch_size=32,
    learning_rate=1e-4, num_train_epochs=2, warmup_steps=150,
    lr_scheduler_type="linear", weight_decay=0.01, optim="adafactor",
    fp16=(DEVICE == "cuda"),
    eval_strategy="steps", eval_steps=500, save_strategy="steps", save_steps=500,
    save_total_limit=2, load_best_model_at_end=True,
    metric_for_best_model="eval_loss", greater_is_better=False,
    logging_steps=100, train_sampling_strategy="group_by_length", report_to="none", seed=42,
)
if SMOKE:
    args.max_steps, args.eval_steps, args.save_steps, args.logging_steps = 3, 3, 3, 1

trainer = Seq2SeqTrainer(
    model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
    data_collator=DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100),
    processing_class=tok,
)
trainer.train()
print("best checkpoint:", trainer.state.best_model_checkpoint, "val loss:", trainer.state.best_metric)""",

    # 6 save model first (survives a predict crash)
    """# Save the fine-tuned model in fp16 (~1.2 GB) before predicting.
model = trainer.model
model.eval()
model.half().save_pretrained("ido-epo-nllb")
tok.save_pretrained("ido-epo-nllb")
if not SMOKE:
    os.system("zip -r -0 -q ido-epo-nllb.zip ido-epo-nllb && echo saved ido-epo-nllb.zip")
model.float()   # generate in fp32 — 1,000 sentences, speed is not an issue""",

    # 7 predict
    """# Translate the held-out real test split -> preds.jsonl (same row order as test.jsonl)
test_rows = load_jsonl("test.jsonl")
if SMOKE:
    test_rows = test_rows[:4] + test_rows[-4:]
model.to(DEVICE)
preds = [None] * len(test_rows)
BATCH = 32
for tgt in ("epo", "ido"):
    idx = [i for i, r in enumerate(test_rows) if r["tgt_lang"] == tgt]
    for s in range(0, len(idx), BATCH):
        chunk = idx[s:s + BATCH]
        feats = [{"input_ids": enc(test_rows[i]["input"], test_rows[i]["src_lang"])} for i in chunk]
        batch = tok.pad(feats, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(**batch, forced_bos_token_id=LANG_ID[tgt],
                                 num_beams=4, max_new_tokens=MAX_LEN)
        for i, o in zip(chunk, tok.batch_decode(out, skip_special_tokens=True)):
            preds[i] = o.strip()

with open("preds.jsonl", "w", encoding="utf-8") as f:
    for r, p in zip(test_rows, preds):
        f.write(json.dumps({"input": r["input"], "output": p}, ensure_ascii=False) + "\\n")
for r, p in list(zip(test_rows, preds))[:4] + list(zip(test_rows, preds))[-4:]:
    print(f"[{r['src_lang']}->{r['tgt_lang']}] {r['input']}\\n  ref : {r['output']}\\n  pred: {p}\\n")
print("wrote preds.jsonl — download it and run: python3 eval/eval_chrf.py --dataset %s --apertium --pred preds.jsonl")""" % DATASET,
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
    out = ROOT / "finetune_nllb.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("wrote", out)


def smoke():
    """Run cells 3-7 on CPU against the local datasets/<DATASET>/ dir (skips pip + git clone)."""
    os.environ["SMOKE"] = "1"
    workdir = Path(os.environ.get("SMOKE_DIR", "/tmp/nllb_smoke"))
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)
    g = {"DATA_DIR": str(ROOT / "datasets" / DATASET)}
    for i, src in enumerate(CELLS[2:], start=3):
        print(f"===== cell {i} =====", flush=True)
        exec(compile(src, f"cell{i}", "exec"), g)


if __name__ == "__main__":
    smoke() if "--smoke" in sys.argv else build()
