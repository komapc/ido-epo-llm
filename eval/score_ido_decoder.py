#!/usr/bin/env python3
"""Score predictions into Ido, with bootstrap confidence intervals.

Usage: python3 eval/score_ido_decoder.py PREDS.jsonl [PREDS2.jsonl ...] [--n 1000]

Each row: {"set", "src_lang", "model", "input", "ref", "output"}, as written by
finetune_ido_decoder.ipynb (models "base" and "v2") or by converting browser-build
predictions (space/eval_web.mjs output) into that shape. Prints corpus chrF per
(set, src_lang, model) with a 95% bootstrap interval and, for every model other than
"base", the paired-bootstrap difference against "base" on the same sentences.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mtlib.metrics import _fbeta_from_totals, _order_prec_rec  # noqa: E402


def stats(hyp: str, ref: str) -> list[tuple[int, int, int, int]]:
    """Per-sentence chrF n-gram statistics (n = 1..6); corpus chrF sums them."""
    return [tuple(t) for t in _order_prec_rec([hyp], [ref], 6)]


def corpus(rows: list[list[tuple[int, int, int, int]]]) -> float:
    totals = [tuple(sum(r[n][i] for r in rows) for i in range(4)) for n in range(6)]
    return _fbeta_from_totals(totals, 2.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("preds", nargs="+", type=Path)
    ap.add_argument("--n", type=int, default=1000, help="bootstrap samples")
    args = ap.parse_args()
    groups: dict[tuple[str, str], dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for p in args.preds:
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            groups[(r["set"], r["src_lang"])][r["model"]].append(stats(r["output"], r["ref"]))
    rng = random.Random(0)
    print(f"{'set':10} {'src':4} {'model':8} {'n':>4} {'chrF':>6}  95% CI         Δ vs base (95% CI)")
    for (s, lang), models in sorted(groups.items()):
        n = len(next(iter(models.values())))
        samples = [[rng.randrange(n) for _ in range(n)] for _ in range(args.n)]
        boots = {m: sorted(corpus([rows[i] for i in idx]) for idx in samples) for m, rows in models.items()}
        for m, rows in sorted(models.items(), key=lambda kv: kv[0] != "base"):
            assert len(rows) == n, f"{s} {lang}: {m} has {len(rows)} rows, expected {n}"
            b = boots[m]
            line = f"{s:10} {lang:4} {m:8} {n:4} {corpus(rows):6.1f}  [{b[len(b) // 40]:.1f}, {b[-len(b) // 40 - 1]:.1f}]"
            if m != "base" and "base" in models:
                base = models["base"]
                d = sorted(corpus([rows[i] for i in idx]) - corpus([base[i] for i in idx]) for idx in samples)
                line += f"  {corpus(rows) - corpus(base):+5.1f} [{d[len(d) // 40]:+.1f}, {d[-len(d) // 40 - 1]:+.1f}]"
            print(line)


if __name__ == "__main__":
    main()
