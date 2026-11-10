#!/usr/bin/env python3
"""Exploration (step A of "Ido for NLLB"): how far is the current model from
Ido <-> other languages, without any new training?

Usage:
  python3 eval/explore_zero_shot.py --out eval/explore/2026-11-10 \
      [--ours models/ido-epo-nllb] [--base facebook/nllb-200-distilled-600M] \
      [--langs eng spa fra deu] [--max-per-lang 300] [--forget-n 200]

For each language X it scores, on the same sentences and with the same greedy
decoding (what the browser build uses):
  ido->X  zero-shot   ours, source ido_Latn, forced target X
  ido->X  pivot-ours  ours ido->epo, then ours epo->X   (one model)
  ido->X  pivot-base  ours ido->epo, then base NLLB epo->X (two models)
  X->ido  zero-shot   ours, source X, forced target ido_Latn
  X->ido  pivot-ours  ours X->epo, then ours epo->ido
  X->ido  pivot-base  base X->epo, then ours epo->ido
and a forgetting check on Tatoeba epo<->X pairs: ours vs base, both directions.

EXPLORATION SET, NOT A BENCHMARK. The Ido sentences are the v1-io-eo val+test
Ido texts (never in v1 train), paired with their direct Tatoeba translations
into X. They overlap v1 val, are small (Spanish ~100), and Tatoeba sentences
are short. Use the numbers to decide what to train, not to report quality.

Writes <out>/sets/*.jsonl, <out>/preds/*.jsonl and <out>/results.md.
Needs torch + transformers (CPU is fine) and the cached Tatoeba dumps.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus.tatoeba import load_links, load_sentences  # noqa: E402
from mtlib.langs import LANGS  # noqa: E402
from mtlib.metrics import bleu, corpus_chrf  # noqa: E402
from mtlib.paths import ROOT, dataset_dir  # noqa: E402

SEED = 13
MAX_CHARS = 120


# ---------------------------------------------------------------- data sets
def heldout_ido_texts() -> set[str]:
    out: set[str] = set()
    for split in ("val", "test"):
        for line in (dataset_dir() / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            out.add(r["input"] if r["src_lang"] == "ido" else r["output"])
    return out


def build_sets(langs: list[str], max_per_lang: int, forget_n: int, out: Path) -> dict[str, list[dict]]:
    """Return {name: [{"src_lang","tgt_lang","input","ref"}]} and write them to out/sets/."""
    rng = random.Random(SEED)
    ido = load_sentences("ido")
    epo = load_sentences("epo")
    xs = {x: load_sentences(x) for x in langs}
    keep = set(ido) | set(epo)
    for s in xs.values():
        keep |= s.keys()
    adj = load_links(keep)

    held = heldout_ido_texts()
    held_ids = sorted(i for i, t in ido.items() if t in held)
    sets: dict[str, list[dict]] = {}
    for x in langs:
        by_text: dict[str, str] = {}
        for i in held_ids:                      # sorted ids: the smallest X id wins
            cands = sorted(t for t in adj.get(i, ()) if t in xs[x])
            if cands and ido[i] not in by_text:
                by_text[ido[i]] = xs[x][cands[0]]
        pairs = sorted(by_text.items())
        rng.shuffle(pairs)
        pairs = sorted(pairs[:max_per_lang])
        sets[f"ido-{x}"] = [{"src_lang": "ido", "tgt_lang": x, "input": a, "ref": b} for a, b in pairs]
        sets[f"{x}-ido"] = [{"src_lang": x, "tgt_lang": "ido", "input": b, "ref": a} for a, b in pairs]

        # forgetting check: direct epo<->X Tatoeba pairs (no Ido involved)
        # keep it Tatoeba-short (<= MAX_CHARS) so a few long Bible verses don't dominate CPU time
        ep = sorted((e, min(t for t in adj[e] if t in xs[x])) for e in epo
                    if len(epo[e]) <= MAX_CHARS and any(t in xs[x] for t in adj.get(e, ())))
        rng.shuffle(ep)
        ep = sorted(ep[:forget_n])
        sets[f"epo-{x}"] = [{"src_lang": "epo", "tgt_lang": x, "input": epo[e], "ref": xs[x][t]} for e, t in ep]
        sets[f"{x}-epo"] = [{"src_lang": x, "tgt_lang": "epo", "input": xs[x][t], "ref": epo[e]} for e, t in ep]

    (out / "sets").mkdir(parents=True, exist_ok=True)
    for name, rows in sets.items():
        with open(out / "sets" / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  set {name}: {len(rows)}")
    return sets


# ---------------------------------------------------------------- models
class NLLB:
    """Greedy NLLB translation with hand-built inputs ([src_lang] tokens </s>),
    as in the training notebook and the browser worker: the tokenizer does not
    know ido_Latn as a src_lang."""

    def __init__(self, name: str, path: str, batch: int = 16):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        self.torch, self.name, self.batch = torch, name, batch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path, dtype=torch.float32).eval()
        self.cache: dict[tuple[str, str, str], str] = {}

    def lang_id(self, code: str) -> int:
        tid = self.tok.convert_tokens_to_ids(LANGS[code].nllb)
        if tid is None or tid == self.tok.unk_token_id:
            raise ValueError(f"{self.name} has no token for {code}")
        return tid

    def translate(self, texts: list[str], src: str, tgt: str) -> list[str]:
        todo = sorted({t for t in texts if (src, tgt, t) not in self.cache}, key=len)
        sid, tid = self.lang_id(src), self.lang_id(tgt)
        eos, pad = self.tok.eos_token_id, self.tok.pad_token_id
        for i in range(0, len(todo), self.batch):
            chunk = todo[i:i + self.batch]
            ids = [[sid] + self.tok(t, add_special_tokens=False)["input_ids"][:254] + [eos] for t in chunk]
            width = max(map(len, ids))
            inp = self.torch.tensor([x + [pad] * (width - len(x)) for x in ids])
            mask = self.torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in ids])
            with self.torch.inference_mode():
                gen = self.model.generate(input_ids=inp, attention_mask=mask, forced_bos_token_id=tid,
                                          num_beams=1, do_sample=False, max_length=None, max_new_tokens=256)
            for t, o in zip(chunk, self.tok.batch_decode(gen, skip_special_tokens=True)):
                self.cache[(src, tgt, t)] = o.strip()
        return [self.cache[(src, tgt, t)] for t in texts]


# ---------------------------------------------------------------- run
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "explore" / date.today().isoformat())
    ap.add_argument("--ours", default=str(ROOT / "models" / "ido-epo-nllb"))
    ap.add_argument("--base", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--langs", nargs="+", default=["eng", "spa", "fra", "deu"])
    ap.add_argument("--max-per-lang", type=int, default=300)
    ap.add_argument("--forget-n", type=int, default=200)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    args = ap.parse_args()
    t0 = time.time()

    print("building exploration sets…")
    sets = build_sets(args.langs, args.max_per_lang, args.forget_n, args.out)

    import torch
    if args.threads:
        torch.set_num_threads(args.threads)
    ours = NLLB("ours", args.ours)
    base = NLLB("base", args.base)

    results: list[tuple[str, str, str, float, float, int]] = []  # set, condition, note, chrF, BLEU, n
    (args.out / "preds").mkdir(parents=True, exist_ok=True)

    def score(set_name: str, cond: str, hyps: list[str], note: str = "") -> None:
        rows = sets[set_name]
        refs = [r["ref"] for r in rows]
        c, b = corpus_chrf(hyps, refs), bleu(hyps, refs)
        results.append((set_name, cond, note, c, b, len(rows)))
        with open(args.out / "preds" / f"{set_name}.{cond}.jsonl", "w", encoding="utf-8") as f:
            for r, h in zip(rows, hyps):
                f.write(json.dumps({"input": r["input"], "ref": r["ref"], "output": h}, ensure_ascii=False) + "\n")
        print(f"  {set_name:9s} {cond:11s} chrF={c:5.1f} BLEU={b:5.1f} n={len(rows)}  [{time.time() - t0:5.0f}s]", flush=True)

    for x in args.langs:
        fwd = [r["input"] for r in sets[f"ido-{x}"]]
        if fwd:
            score(f"ido-{x}", "zero-shot", ours.translate(fwd, "ido", x))
            mid = ours.translate(fwd, "ido", "epo")
            score(f"ido-{x}", "pivot-ours", ours.translate(mid, "epo", x))
            score(f"ido-{x}", "pivot-base", base.translate(mid, "epo", x))
            back = [r["input"] for r in sets[f"{x}-ido"]]
            score(f"{x}-ido", "zero-shot", ours.translate(back, x, "ido"))
            score(f"{x}-ido", "pivot-ours", ours.translate(ours.translate(back, x, "epo"), "epo", "ido"))
            score(f"{x}-ido", "pivot-base", ours.translate(base.translate(back, x, "epo"), "epo", "ido"))
        for name, (s, t) in ((f"epo-{x}", ("epo", x)), (f"{x}-epo", (x, "epo"))):
            src = [r["input"] for r in sets[name]]
            score(name, "ours", ours.translate(src, s, t))
            score(name, "base", base.translate(src, s, t))

    write_report(args, sets, results, time.time() - t0)


def write_report(args, sets, results, secs: float) -> None:
    by = {(s, c): (ch, bl, n) for s, c, _, ch, bl, n in results}
    lines = [
        f"# Zero-shot / pivot exploration ({date.today().isoformat()})",
        "",
        "Step A of \"Ido for NLLB\": what the current io↔eo model does for other languages, before any training.",
        "",
        "**Exploration set, not a benchmark.** Ido side = v1-io-eo val+test Ido sentences (never in v1 train) with",
        "their direct Tatoeba translation; it overlaps v1 val, is small and the sentences are short.",
        f"Greedy decoding everywhere (as in the browser build). Models: ours = `{Path(args.ours).name}`",
        f"(komapc/nllb-ido-epo-600m), base = `{args.base}`. Runtime {secs / 60:.0f} min on CPU.",
        "",
        "## Ido ↔ X (chrF / BLEU)",
        "",
        "| direction | n | zero-shot (ours) | pivot via epo, ours only | pivot via epo, ours + base |",
        "|---|---|---|---|---|",
    ]
    for x in args.langs:
        for name in (f"ido-{x}", f"{x}-ido"):
            if (name, "zero-shot") not in by:
                continue
            cells = [f"{by[(name, c)][0]:.1f} / {by[(name, c)][1]:.1f}" for c in ("zero-shot", "pivot-ours", "pivot-base")]
            lines.append(f"| {name.replace('-', '→')} | {by[(name, 'zero-shot')][2]} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Forgetting check: Esperanto ↔ X, fine-tuned vs base (chrF / BLEU)",
        "",
        "| direction | n | ours | base | Δ chrF |",
        "|---|---|---|---|---|",
    ]
    for x in args.langs:
        for name in (f"epo-{x}", f"{x}-epo"):
            o, b = by[(name, "ours")], by[(name, "base")]
            lines.append(f"| {name.replace('-', '→')} | {o[2]} | {o[0]:.1f} / {o[1]:.1f} | {b[0]:.1f} / {b[1]:.1f} | {o[0] - b[0]:+.1f} |")
    lines += [
        "",
        "Conditions: *zero-shot* = ours with the source/target language tokens set directly;",
        "*pivot, ours only* = ours ido→epo then ours epo→X (one model, what the browser could do today);",
        "*pivot, ours + base* = the epo↔X hop by base NLLB (a second ~600 MB model).",
        "Predictions are in `preds/`, inputs in `sets/`.",
        "",
    ]
    (args.out / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out / 'results.md'}")


if __name__ == "__main__":
    main()
