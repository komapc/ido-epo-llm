#!/usr/bin/env python3
"""Fail if any eval sentence text leaks into training data.

Checks the published datasets under datasets/ (what the notebooks train and
predict on), not corpus/out/:
  * within a dataset: no val/dev/test text, input or output, in its train, and no
    test text in val/dev;
  * across datasets: no val/dev/test text of ANY dataset in the training data of
    any dataset (train.jsonl, or a monolingual mono.jsonl.gz pool of {"text"}).
Exact text match, as make_dataset.py's own self-check.

Usage: python3 corpus/check_split.py [NAME ...]   (default: all datasets)
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mtlib.paths import DATASETS  # noqa: E402

EVAL = ("val", "dev", "test")


def texts(path: Path) -> set[str]:
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return {json.loads(l)["text"] for l in f if l.strip()}
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["input"] for r in rows} | {r["output"] for r in rows}


def splits(d: Path) -> dict[str, set[str]]:
    out = {s: texts(d / f"{s}.jsonl") for s in ("train", *EVAL) if (d / f"{s}.jsonl").exists()}
    if (d / "mono.jsonl.gz").exists():
        out["mono"] = texts(d / "mono.jsonl.gz")
    return out


def main() -> int:
    all_names = sorted(p.name for p in DATASETS.iterdir() if p.is_dir())
    names = sys.argv[1:] or all_names
    if not names:
        print(f"no datasets under {DATASETS}")
        return 1
    data = {n: splits(DATASETS / n) for n in all_names}
    training = {f"{n}/{s}": t for n, sp in data.items() for s, t in sp.items() if s in ("train", "mono")}
    ok = True
    for n in names:
        sp = data[n]
        bad = {}
        for e in EVAL:
            if e not in sp:
                continue
            for tname, t in training.items():
                bad[f"{e}∩{tname}"] = sp[e] & t
        if "test" in sp:
            for e in ("val", "dev"):
                if e in sp:
                    bad[f"test∩{e}"] = sp["test"] & sp[e]
        leaks = {k: len(v) for k, v in bad.items() if v}
        print(f"{n}: " + (", ".join(f"{k}: {v}" for k, v in leaks.items()) if leaks else
                          f"ok ({', '.join(f'{s} {len(t)}' for s, t in sp.items())})"))
        ok &= not leaks
    if not ok:
        print("LEAK: eval text appears in training data; rebuild into a new datasets/<name>/")
        return 1
    print("ok: no eval sentence text in any training data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
