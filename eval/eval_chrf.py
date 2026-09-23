#!/usr/bin/env python3
"""Head-to-head chrF/BLEU eval on the held-out real test split.

Usage:
  python3 eval_chrf.py --apertium                 # baseline to beat
  python3 eval_chrf.py --pred preds.jsonl         # LLM, {"output": "..."} per line
  python3 eval_chrf.py --apertium --pred preds.jsonl   # both, side by side

The LLM ships only if it beats Apertium overall, or at least on the subset where
Apertium emits a failure marker (`*`/`#`/`@`) — those are the inputs the rule
system can't handle, which is the whole reason to add a neural engine.

Self-contained chrF (char n-gram F, n=6, beta=2) and BLEU-4; no required deps.
chrF is computed as the standard corpus-POOLED statistic (Popovic 2015, and
what sacrebleu's default corpus chrF reports): n-gram match/total counts are
summed over the whole corpus for each order 1..6, precision and recall are
each averaged across orders, and a single F-beta is taken from those two
averages. This is NOT the same as averaging a per-sentence chrF score, and
earlier revisions of this script did the latter (plus averaged per-order F
instead of averaging P/R first) — both are non-standard and were fixed.
This script does not import or cross-check against sacrebleu; it is not a
dependency of this project (no requirements file lists it, and it isn't
installed in the environments this has been run in). Numbers here are
expected to track sacrebleu's chrF/BLEU closely but have not been verified
against it — treat cross-repo/paper comparisons with that caveat in mind.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TEST = REPO / "llm" / "data" / "out" / "test.jsonl"
PAIR_DIR = REPO / "apertium-ido-epo"
MODE = {("ido", "epo"): "ido-epo", ("epo", "ido"): "epo-ido"}
FAIL = ("*", "#", "@")


def char_ngrams(s: str, n: int) -> Counter:
    s = s.replace(" ", "")
    return Counter(s[i:i + n] for i in range(len(s) - n + 1)) if len(s) >= n else Counter()


def _order_prec_rec(hyps, refs, n: int) -> list[tuple[int, int, int, int]]:
    """Per-order (match, hyp_total, match, ref_total) counts, pooled over all
    hyp/ref pairs given (a single pair for sentence-level, the whole corpus
    for corpus-level)."""
    totals = [[0, 0, 0, 0] for _ in range(n)]  # match, hyp_total, match, ref_total
    for hyp, ref in zip(hyps, refs):
        for k in range(1, n + 1):
            h, r = char_ngrams(hyp, k), char_ngrams(ref, k)
            match = sum((h & r).values())
            t = totals[k - 1]
            t[0] += match
            t[1] += sum(h.values())
            t[2] += match
            t[3] += sum(r.values())
    return totals


def _fbeta_from_totals(totals, beta: float) -> float:
    """Standard (Popovic 2015 / sacrebleu) chrF: average precision and recall
    across n-gram orders separately, THEN take a single F-beta — not an
    average of per-order F-beta scores (which over-weights orders that
    happen to have low precision*recall)."""
    precs = [m / d for m, d, _, _ in totals if d]
    recs = [m / d for _, _, m, d in totals if d]
    if not precs or not recs:
        return 0.0
    prec, rec = sum(precs) / len(precs), sum(recs) / len(recs)
    if prec + rec == 0:
        return 0.0
    b2 = beta * beta
    return 100 * (1 + b2) * prec * rec / (b2 * prec + rec)


def chrf(hyp: str, ref: str, n: int = 6, beta: float = 2.0) -> float:
    """Sentence-level chrF (see corpus_chrf for the corpus-level metric this
    script actually reports)."""
    if not hyp or not ref:
        return 0.0
    return _fbeta_from_totals(_order_prec_rec([hyp], [ref], n), beta)


def corpus_chrf(hyps, refs, n: int = 6, beta: float = 2.0) -> float:
    """Corpus-level chrF: n-gram counts are pooled over the WHOLE corpus per
    order before computing precision/recall/F (not a mean of per-sentence
    chrF scores). This is what makes it comparable to sacrebleu's chrF."""
    if not hyps:
        return 0.0
    return _fbeta_from_totals(_order_prec_rec(hyps, refs, n), beta)


def bleu(hyps, refs, n=4) -> float:
    """Corpus BLEU-4 with a brevity penalty; token = whitespace."""
    p_num = [0] * n
    p_den = [0] * n
    hyp_len = ref_len = 0
    for h, r in zip(hyps, refs):
        ht, rt = h.split(), r.split()
        hyp_len += len(ht)
        ref_len += len(rt)
        for k in range(n):
            hg = Counter(tuple(ht[i:i + k + 1]) for i in range(len(ht) - k))
            rg = Counter(tuple(rt[i:i + k + 1]) for i in range(len(rt) - k))
            p_num[k] += sum((hg & rg).values())
            p_den[k] += max(sum(hg.values()), 1)
    if min(p_num) == 0:
        return 0.0
    logp = sum(math.log(p_num[k] / p_den[k]) for k in range(n)) / n
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return 100 * bp * math.exp(logp)


def apertium_translate(rows) -> list[str]:
    out = []
    for (sl, tl), group in _by_dir(rows).items():
        joined = "\n".join(r["input"].replace("\n", " ") for r in group) + "\n"
        proc = subprocess.run(["apertium", "-d", str(PAIR_DIR), MODE[(sl, tl)]],
                              input=joined, capture_output=True, text=True)
        res = proc.stdout.split("\n")
        for r, t in zip(group, res):
            r["_apertium"] = t.strip()
    return [r["_apertium"] for r in rows]


def _by_dir(rows):
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["src_lang"], r["tgt_lang"]), []).append(r)
    return groups


def report(name, hyps, refs, mask=None):
    if mask is not None:
        hyps = [h for h, m in zip(hyps, mask) if m]
        refs = [r for r, m in zip(refs, mask) if m]
    print(f"  {name:24s} chrF={corpus_chrf(hyps, refs):5.2f}  "
          f"BLEU={bleu(hyps, refs):5.2f}  (n={len(hyps)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apertium", action="store_true")
    ap.add_argument("--pred", type=Path, help="LLM predictions jsonl with 'output'")
    ap.add_argument("--test", type=Path, default=TEST)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.test.read_text(encoding="utf-8").splitlines() if l.strip()]
    refs = [r["output"] for r in rows]
    print(f"test: {len(rows)} real pairs")

    apertium_hyps = None
    if args.apertium:
        apertium_hyps = apertium_translate(rows)
        fail_mask = [any(m in h for m in FAIL) for h in apertium_hyps]
        print("Apertium baseline:")
        report("overall", apertium_hyps, refs)
        report("where Apertium FAILS", apertium_hyps, refs, fail_mask)
        report("where Apertium OK", apertium_hyps, refs, [not m for m in fail_mask])
        print(f"  Apertium emits a failure marker on "
              f"{sum(fail_mask)}/{len(rows)} ({100*sum(fail_mask)/len(rows):.1f}%) inputs")

    if args.pred:
        preds = [json.loads(l)["output"] for l in args.pred.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(preds) == len(rows), f"{len(preds)} preds vs {len(rows)} test rows"
        print("LLM:")
        report("overall", preds, refs)
        if apertium_hyps is not None:
            fail_mask = [any(m in h for m in FAIL) for h in apertium_hyps]
            report("where Apertium FAILS", preds, refs, fail_mask)


if __name__ == "__main__":
    main()
