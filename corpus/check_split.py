#!/usr/bin/env python3
"""Fail if any eval sentence text leaks across the published splits.

Checks every published dataset under datasets/ (what the notebooks train and
predict on), not corpus/out/: no val/test text, input or output, may appear
anywhere in train, and no test text in val. Same invariant as make_dataset.py's
self-check, but on the committed files, so a stale copy can't slip through.

Usage: python3 corpus/check_split.py [NAME ...]   (default: all datasets)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mtlib.paths import DATASETS  # noqa: E402


def texts(d: Path, split: str) -> set[str]:
    rows = [json.loads(l) for l in (d / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["input"] for r in rows} | {r["output"] for r in rows}


def check(d: Path) -> bool:
    train, val, test = texts(d, "train"), texts(d, "val"), texts(d, "test")
    bad = {
        "val∩train": val & train,
        "test∩train": test & train,
        "test∩val": test & val,
    }
    print(f"{d.name}: " + ", ".join(f"{k}: {len(v)}" for k, v in bad.items()))
    return not any(bad.values())


def main() -> int:
    names = sys.argv[1:] or sorted(p.name for p in DATASETS.iterdir() if p.is_dir())
    if not names:
        print(f"no datasets under {DATASETS}")
        return 1
    ok = all([check(DATASETS / n) for n in names])
    if not ok:
        print("LEAK: a published split shares sentence texts across train/val/test; "
              "rebuild with corpus/make_dataset.py into a new datasets/<name>/")
        return 1
    print("ok: no sentence text shared across train/val/test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
