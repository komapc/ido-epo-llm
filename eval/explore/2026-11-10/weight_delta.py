import torch, collections, glob, os
from safetensors.torch import load_file
ours = load_file("models/ido-epo-nllb/model.safetensors")
bp = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--facebook--nllb-200-distilled-600M/snapshots/*/"))[0]
fs = glob.glob(bp+"*.safetensors")
if fs: base = load_file(fs[0])
else: base = torch.load(bp+"pytorch_model.bin", map_location="cpu", weights_only=True)
print("base keys", len(base), "ours keys", len(ours))
grp = collections.defaultdict(lambda: [0.0, 0.0, 0])
lowrank = []
for k, w in ours.items():
    kb = k if k in base else ("model."+k if "model."+k in base else k.replace("model.","",1))
    if kb not in base: print("missing", k); continue
    b = base[kb].float(); w = w.float()
    if b.shape != w.shape:
        print("shape", k, tuple(b.shape), tuple(w.shape)); n=min(b.shape[0], w.shape[0]); b=b[:n]; w=w[:n]
    d = w - b
    g = "embed" if "embed" in k or "lm_head" in k or "shared" in k else ("encoder" if "encoder" in k else ("decoder" if "decoder" in k else "other"))
    grp[g][0] += d.pow(2).sum().item(); grp[g][1] += b.pow(2).sum().item(); grp[g][2] += d.numel()
    if d.ndim == 2 and min(d.shape) >= 512 and "embed" not in k and "shared" not in k and "lm_head" not in k:
        s = torch.linalg.svdvals(d); e = (s**2).cumsum(0)/(s**2).sum()
        lowrank.append((k, int((e < 0.9).sum())+1, min(d.shape)))
for g,(dn,bn,n) in grp.items(): print(f"{g:8s} params={n/1e6:7.1f}M  rel|Δ|={ (dn/bn)**.5:.4f}  share of Δ²={dn/sum(v[0] for v in grp.values()):.3f}")
import statistics
for side in ("encoder","decoder"):
    r=[x[1]/x[2] for x in lowrank if side in x[0]]
    print(side, "rank for 90% of Δ energy / full: median", round(statistics.median(r),3), "min", round(min(r),3), "max", round(max(r),3), "n", len(r))
