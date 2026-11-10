"""Body-blend test (base embeddings + ours ido_Latn row, layers base+a*(ours-base)); variant of the embedding surgery test: ours body + base embedding rows for tokens unseen in v1 train."""
import json, sys, glob, os, torch
sys.path.insert(0, ".")
sys.path.insert(0, "eval")
torch.set_num_threads(8)
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.utils import logging; logging.set_verbosity_error()
from mtlib.metrics import corpus_chrf
from mtlib.langs import LANGS
OURS="models/ido-epo-nllb"; BASE="facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(OURS)
seen = set(tok.all_special_ids) | {tok.convert_tokens_to_ids(l.nllb) for l in LANGS.values()}
for line in open("datasets/v1-io-eo/train.jsonl"):
    r = json.loads(line)
    for t in (r["input"], r["output"]): seen.update(tok(t, add_special_tokens=False)["input_ids"])
print("seen tokens", len(seen))
ours = AutoModelForSeq2SeqLM.from_pretrained(OURS, dtype=torch.float32).eval()
base = AutoModelForSeq2SeqLM.from_pretrained(BASE, dtype=torch.float32).eval()
Eo = ours.get_input_embeddings().weight; Eb = base.get_input_embeddings().weight
n = min(Eo.shape[0], Eb.shape[0])
seen_mask = torch.zeros(Eo.shape[0], dtype=torch.bool); seen_mask[list(s for s in seen if s < Eo.shape[0])] = True
print("rel change seen", ((Eo[:n][seen_mask[:n]]-Eb[:n][seen_mask[:n]]).norm()/Eb[:n][seen_mask[:n]].norm()).item(),
      "unseen", ((Eo[:n][~seen_mask[:n]]-Eb[:n][~seen_mask[:n]]).norm()/Eb[:n][~seen_mask[:n]].norm()).item())
print("tied:", ours.get_output_embeddings().weight.data_ptr()==Eo.data_ptr())
orig = Eo.detach().clone()
def set_variant(v):
    with torch.no_grad():
        Eo.copy_(orig)
        if v == "restore-unseen":
            m = ~seen_mask[:n]; Eo[:n][m] = Eb[:n][m]
        elif v == "restore-all-but-ido":
            ido = tok.convert_tokens_to_ids("ido_Latn"); keep = Eo[ido].clone(); Eo[:n] = Eb[:n]; Eo[ido] = keep
        ours.lm_head.weight = ours.get_input_embeddings().weight
def tr(texts, src, tgt):
    out=[]
    for i in range(0,len(texts),16):
        ch=texts[i:i+16]
        ids=[[tok.convert_tokens_to_ids(LANGS[src].nllb)]+tok(t,add_special_tokens=False)["input_ids"][:200]+[tok.eos_token_id] for t in ch]
        w=max(map(len,ids)); inp=torch.tensor([x+[tok.pad_token_id]*(w-len(x)) for x in ids]); m=(inp!=tok.pad_token_id).long()
        with torch.inference_mode():
            g=ours.generate(input_ids=inp,attention_mask=m,forced_bos_token_id=tok.convert_tokens_to_ids(LANGS[tgt].nllb),num_beams=1,do_sample=False,max_length=None,max_new_tokens=160)
        out+= [o.strip() for o in tok.batch_decode(g,skip_special_tokens=True)]
    return out
def load(p,k=60): return [json.loads(l) for l in open(p)][:k]
X="eval/explore/2026-11-10/sets"
tests = {"ido-spa": load(f"{X}/ido-spa.jsonl"), "epo-spa": load(f"{X}/epo-spa.jsonl"), "ido-deu": load(f"{X}/ido-deu.jsonl")}
v1 = [json.loads(l) for l in open("datasets/v1-io-eo/test.jsonl")]
tests["v1 ido-epo"] = [{"src_lang":"ido","tgt_lang":"epo","input":r["input"],"ref":r["output"]} for r in v1 if r["src_lang"]=="ido"][:100]
tests["v1 epo-ido"] = [{"src_lang":"epo","tgt_lang":"ido","input":r["input"],"ref":r["output"]} for r in v1 if r["src_lang"]=="epo"][:100]
body_ours = {k: v.detach().clone() for k, v in ours.state_dict().items() if "embed" not in k and "shared" not in k and "lm_head" not in k}
body_base = dict(base.state_dict())
def blend(a, enc=None):
    set_variant("restore-all-but-ido")
    with torch.no_grad():
        sd = ours.state_dict()
        for k, w in body_ours.items():
            if k not in body_base: continue
            aa = enc if (enc is not None and ".encoder." in "." + k) else a
            sd[k].copy_(body_base[k] + aa * (w - body_base[k]))
import os
VARIANTS = os.environ.get("VARIANTS", "blend-0.5,blend-0.75,blend-0.9").split(",")
for v in VARIANTS:
    if v.startswith("enc"):   # e.g. enc1-dec0.5: encoder alpha 1, decoder alpha 0.5
        e, d = v[3:].split("-dec"); blend(float(d), enc=float(e))
    else:
        blend(float(v.split("-")[1]))
    res = {}
    for name, rows in tests.items():
        h = tr([r["input"] for r in rows], rows[0]["src_lang"], rows[0]["tgt_lang"])
        res[name] = round(corpus_chrf(h, [r["ref"] for r in rows]),1)
        if name=="ido-spa": print("   e.g.", h[:3])
    print(v, res, flush=True)
