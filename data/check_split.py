#!/usr/bin/env python3
"""Fail if any eval sentence text leaks across the published splits.

Checks data/{train,val,test}.jsonl (what the notebook trains and predicts on),
not data/out/: no val/test text, input or output, may appear anywhere in
train, and no test text in val. Same invariant as make_dataset.py's
self-check, but on the committed files, so a stale copy can't slip through.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent


def texts(name: str) -> set[str]:
    rows = [json.loads(l) for l in (DATA / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["input"] for r in rows} | {r["output"] for r in rows}


def main() -> int:
    train, val, test = texts("train"), texts("val"), texts("test")
    bad = {
        "val∩train": val & train,
        "test∩train": test & train,
        "test∩val": test & val,
    }
    for k, v in bad.items():
        print(f"{k}: {len(v)}")
    if any(bad.values()):
        print("LEAK: published splits share sentence texts; "
              "run data/make_dataset.py and copy data/out/{train,val,test}.jsonl to data/")
        return 1
    print("ok: no sentence text shared across train/val/test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
