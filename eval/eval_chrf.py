#!/usr/bin/env python3
"""Head-to-head chrF/BLEU eval on the held-out real test split.

Usage:
  python3 eval/eval_chrf.py --apertium                 # baseline to beat
  python3 eval/eval_chrf.py --pred preds.jsonl         # LLM, {"output": "..."} per line
  python3 eval/eval_chrf.py --apertium --pred preds.jsonl   # both, side by side
  python3 eval/eval_chrf.py --dataset NAME ...         # another split under datasets/

The LLM ships only if it beats Apertium overall, or at least on the subset where
Apertium emits a failure marker (`*`/`#`/`@`) — those are the inputs the rule
system can't handle, which is the whole reason to add a neural engine.

Metrics live in mtlib/metrics.py (corpus-pooled chrF, n=6, beta=2; BLEU-4).
The Apertium baseline exists only for the directions in mtlib.langs.APERTIUM_MODES;
rows in other directions are reported by the LLM side only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mtlib.langs import APERTIUM_MODES, APERTIUM_PAIR_DIR  # noqa: E402
from mtlib.metrics import bleu, corpus_chrf  # noqa: E402
from mtlib.paths import DEFAULT_DATASET, dataset_dir  # noqa: E402

FAIL = ("*", "#", "@")


def apertium_translate(rows) -> list[str]:
    for (sl, tl), group in _by_dir(rows).items():
        if (sl, tl) not in APERTIUM_MODES:
            raise SystemExit(f"no Apertium baseline for {sl}->{tl}; drop --apertium")
        joined = "\n".join(r["input"].replace("\n", " ") for r in group) + "\n"
        proc = subprocess.run(["apertium", "-d", str(APERTIUM_PAIR_DIR), APERTIUM_MODES[(sl, tl)]],
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
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="split directory under datasets/")
    ap.add_argument("--test", type=Path, help="test jsonl (default: datasets/<dataset>/test.jsonl)")
    args = ap.parse_args()
    args.test = args.test or dataset_dir(args.dataset) / "test.jsonl"

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
        pred_rows = [json.loads(l) for l in args.pred.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(pred_rows) != len(rows):
            raise SystemExit(f"{len(pred_rows)} preds vs {len(rows)} test rows")
        # Same row count is not enough: two different 1000-row test files
        # pass that check. Preds carry their input; require it to line up.
        if not all("input" in p for p in pred_rows):
            raise SystemExit("preds lack 'input'; regenerate them with the current notebook")
        off = [i for i, (p, r) in enumerate(zip(pred_rows, rows)) if p["input"] != r["input"]]
        if off:
            raise SystemExit(f"{len(off)}/{len(rows)} preds don't match {args.test} by input "
                             f"(first at row {off[0]}) — scored against a different test set?")
        preds = [p["output"] for p in pred_rows]
        print("LLM:")
        report("overall", preds, refs)
        if apertium_hyps is not None:
            fail_mask = [any(m in h for m in FAIL) for h in apertium_hyps]
            report("where Apertium FAILS", preds, refs, fail_mask)


if __name__ == "__main__":
    main()
