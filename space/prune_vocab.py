#!/usr/bin/env python3
"""Shrink the NLLB vocabulary (256k tokens for 200 languages) to the tokens that
text in the chosen languages actually uses, so the browser (transformers.js) build is small.

Keeps: every token seen when tokenizing the corpora below (published datasets,
the cached Tatoeba sentences of each --langs language, the io.wiki word list), every
single-character token outside CJK (so unseen words still tokenize, with no
<unk>), the special and language tokens, and the closure under BPE merges.
That closure makes the tokenization identical to the original's for any word
whose final tokens were all kept.

Usage: python prune_vocab.py SRC_MODEL_DIR OUT_DIR [--langs ido epo eng ...]
The kept token set depends only on the tokenizer and the corpus, so models that share a
tokenizer (e.g. the two halves of a split-decoder build) get identical new ids.
"""
import argparse
import bz2
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForSeq2SeqLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mtlib.langs import LANGS  # noqa: E402
from mtlib.paths import CACHE, DATASETS, WORKSPACE  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("src", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--langs", nargs="+", default=["ido", "epo"])
args = ap.parse_args()
SRC, OUT = args.src, args.out
# eng_Latn stays in for the io/eo build too (it was in the original keep list)
KEEP_SPECIAL = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"] + sorted(
    {LANGS[l].nllb for l in args.langs} | {"eng_Latn"})


def corpus():
    for p in sorted(DATASETS.glob("*/*.jsonl")):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            yield r.get("input", ""); yield r.get("output", "")
    for lang in args.langs:
        with bz2.open(CACHE / f"{lang}_sentences.tsv.bz2", "rt", encoding="utf-8") as f:
            for line in f:
                yield line.rstrip("\n").split("\t")[-1]
    freq = json.load(open(WORKSPACE / "extractor" / "work" / "io_wiki_frequency.json"))
    for it in freq["items"]:
        w = it["token"]
        yield w; yield w[:1].upper() + w[1:]


tj = json.load(open(SRC / "tokenizer.json", encoding="utf-8"))
vocab = tj["model"]["vocab"]                       # token -> id
id2tok = {i: t for t, i in vocab.items()}
for a in tj["added_tokens"]:
    id2tok[a["id"]] = a["content"]

tok = Tokenizer.from_file(str(SRC / "tokenizer.json"))
seen, batch, n_txt = set(), [], 0
for text in corpus():
    if text:
        batch.append(text)
    if len(batch) == 4096:
        for e in tok.encode_batch(batch, add_special_tokens=False):
            seen.update(e.ids)
        n_txt += len(batch); batch = []
for e in tok.encode_batch(batch, add_special_tokens=False):
    seen.update(e.ids)
n_txt += len(batch)
print(f"corpus: {n_txt} texts -> {len(seen)} distinct tokens")

keep = {id2tok[i] for i in seen if i in id2tok}
for t in vocab:
    c = t.lstrip("▁")
    if len(c) == 1 and ord(c) < 0x2E80:
        keep.add(t)
keep.update(KEEP_SPECIAL)
merges = tj["model"]["merges"]
changed = True
while changed:  # closure: the parts of every kept merge result are kept too
    changed = False
    for a, b in merges:
        if a + b in keep and (a not in keep or b not in keep):
            keep.update((a, b)); changed = True

old_ids = sorted(i for i, t in id2tok.items() if t in keep)
assert old_ids[:4] == [0, 1, 2, 3], old_ids[:4]
new_id = {o: n for n, o in enumerate(old_ids)}
print(f"vocab {len(id2tok)} -> {len(old_ids)}")

tj["model"]["vocab"] = {t: new_id[i] for t, i in vocab.items() if i in new_id}
tj["model"]["merges"] = [[a, b] for a, b in merges if a + b in keep]
tj["added_tokens"] = [dict(a, id=new_id[a["id"]]) for a in tj["added_tokens"] if a["id"] in new_id]
for name, st in tj["post_processor"]["special_tokens"].items():
    st["ids"] = [new_id[i] for i in st["ids"]]
OUT.mkdir(parents=True, exist_ok=True)
json.dump(tj, open(OUT / "tokenizer.json", "w", encoding="utf-8"), ensure_ascii=False)

cfg = json.load(open(SRC / "tokenizer_config.json"))
cfg["extra_special_tokens"] = [t for t in KEEP_SPECIAL if t.endswith("_Latn")]
cfg["additional_special_tokens"] = cfg["extra_special_tokens"]
json.dump(cfg, open(OUT / "tokenizer_config.json", "w"), indent=2, ensure_ascii=False)

model = AutoModelForSeq2SeqLM.from_pretrained(SRC, dtype=torch.float32)
w = model.get_input_embeddings().weight
w.data[: len(old_ids)] = w.data[torch.tensor(old_ids)].clone()
model.resize_token_embeddings(len(old_ids))
model.config.use_cache = True
model.save_pretrained(OUT)
print("saved", OUT, "params", sum(p.numel() for p in model.parameters()))
