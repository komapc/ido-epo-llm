#!/usr/bin/env python3
"""Build high-precision synthetic IO<->EO pairs with local Apertium.

Weak supervision: translate clean monolingual sentences through the Apertium
ido-epo pair and keep ONLY outputs with no failure markers (`*` unknown,
`#` generation fail, `@` analysis/transfer fail). These are cases where the
rule system is confident, so the pair is almost always correct. This gives bulk
volume + grammaticality, but caps quality at Apertium — it is mixed with, never
substituted for, the real Tatoeba pairs, and is excluded from eval downstream
(make_dataset.py holds out a real-only test split and dedups by source).

Source sentences are the Tatoeba monolingual dumps already cached by
build_tatoeba_pairs.py (clean, no wikitext). Re-run that first.

Output: data/out/apertium_synth.jsonl  (records like the Tatoeba builder).
"""
from __future__ import annotations

import bz2
import csv
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
OUT = HERE / "out"
REPO = HERE.parents[1]                      # apertium-dev/
PAIR_DIR = REPO / "apertium-ido-epo"        # has modes/ + compiled .bin

# Cap per direction so synthetic doesn't drown the real data. EO has 800k+
# sentences; we don't need them all.
MAX_PER_DIR = 40000
BATCH = 2000
FAIL_MARKERS = ("*", "#", "@")


def load_cached_sentences(lang: str, limit: int) -> list[str]:
    path = CACHE / f"{lang}_sentences.tsv.bz2"
    if not path.exists():
        raise SystemExit(f"missing {path} — run build_tatoeba_pairs.py first")
    out: list[str] = []
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 3 and row[2].strip():
                out.append(row[2].strip())
            if len(out) >= limit:
                break
    return out


def translate(sentences: list[str], mode: str) -> list[str]:
    """Run a batch through `apertium -d PAIR_DIR <mode>`, one line per sentence."""
    # Replace internal newlines so each sentence stays on its own line.
    joined = "\n".join(s.replace("\n", " ") for s in sentences) + "\n"
    proc = subprocess.run(
        ["apertium", "-d", str(PAIR_DIR), mode],
        input=joined, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(f"apertium {mode} failed: {proc.stderr[:500]}")
    return proc.stdout.split("\n")


def is_clean(out: str) -> bool:
    out = out.strip()
    if not out:
        return False
    return not any(m in out for m in FAIL_MARKERS)


def build_direction(src_lang: str, tgt_lang: str, mode: str, fh) -> int:
    print(f"== {mode}: loading up to {MAX_PER_DIR} {src_lang} sentences")
    srcs = load_cached_sentences(src_lang, MAX_PER_DIR)
    kept = 0
    for i in range(0, len(srcs), BATCH):
        chunk = srcs[i:i + BATCH]
        outs = translate(chunk, mode)
        for s, t in zip(chunk, outs):
            if is_clean(t) and t.strip() != s:
                fh.write(json.dumps({
                    "src_lang": src_lang, "tgt_lang": tgt_lang,
                    "src": s, "tgt": t.strip(), "provenance": "apertium_synth",
                }, ensure_ascii=False) + "\n")
                kept += 1
        print(f"   {i + len(chunk)}/{len(srcs)} processed, {kept} clean kept")
    return kept


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not (PAIR_DIR / "modes").exists():
        raise SystemExit(f"no Apertium pair at {PAIR_DIR}")
    total = 0
    with open(OUT / "apertium_synth.jsonl", "w", encoding="utf-8") as fh:
        total += build_direction("ido", "epo", "ido-epo", fh)
        total += build_direction("epo", "ido", "epo-ido", fh)
    print(f"wrote {total} synthetic records to {OUT / 'apertium_synth.jsonl'}")


if __name__ == "__main__":
    main()
