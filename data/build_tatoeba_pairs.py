#!/usr/bin/env python3
"""Build real IO<->EO sentence pairs from Tatoeba.

Tatoeba is the only source of genuine human Ido<->Esperanto sentence pairs.
We take direct IO<->EO links plus pivots through high-resource bridge
languages (English, French) where a single Ido sentence and a single Esperanto
sentence both translate the same bridge sentence.

Outputs newline-delimited JSON to data/out/tatoeba_io_eo.jsonl with records:
  {"src_lang","tgt_lang","src","tgt","provenance"}
Both directions are emitted so the bidirectional model sees each pair twice.

No third-party deps; uses urllib + bz2 from the stdlib. Re-runs are cached:
downloaded dumps land in data/cache/ and are reused.
"""
from __future__ import annotations

import bz2
import csv
import io
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
OUT = HERE / "out"
BASE = "https://downloads.tatoeba.org/exports"

# Tatoeba ISO 639-3 codes
IDO, EPO = "ido", "epo"
# Bridge languages for pivoting (Tatoeba code -> human name, for logging)
BRIDGES = {"eng": "English", "fra": "French", "spa": "Spanish", "deu": "German"}


def fetch(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached {dest.name}")
        return dest
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "ido-tradukilo-llm/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest


def load_sentences(lang: str) -> dict[int, str]:
    """sentence_id -> text for one language."""
    path = fetch(f"{BASE}/per_language/{lang}/{lang}_sentences.tsv.bz2",
                 CACHE / f"{lang}_sentences.tsv.bz2")
    out: dict[int, str] = {}
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            # id \t lang \t text
            if len(row) >= 3:
                out[int(row[0])] = row[2].strip()
    print(f"  {lang}: {len(out)} sentences")
    return out


def load_links() -> list[tuple[int, int]]:
    """All Tatoeba translation links (sentence_id, translation_id)."""
    path = fetch(f"{BASE}/links.tar.bz2", CACHE / "links.tar.bz2")
    import tarfile
    links: list[tuple[int, int]] = []
    with tarfile.open(path, "r:bz2") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("links.csv"))
        f = io.TextIOWrapper(tar.extractfile(member), encoding="utf-8")
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 2:
                links.append((int(row[0]), int(row[1])))
    print(f"  links: {len(links)} total")
    return links


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading sentences…")
    ido = load_sentences(IDO)
    epo = load_sentences(EPO)
    if not ido:
        print("No Ido sentences from Tatoeba — aborting.", file=sys.stderr)
        sys.exit(1)
    bridges = {b: load_sentences(b) for b in BRIDGES}

    print("Loading links…")
    links = load_links()
    # adjacency: sid -> set(translation ids), symmetric
    adj: dict[int, set[int]] = {}
    for a, b in links:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    pairs: set[tuple[str, str]] = set()  # (ido_text, epo_text)

    # 1) direct IO<->EO links
    for sid in ido:
        for t in adj.get(sid, ()):  # noqa: B007
            if t in epo:
                pairs.add((ido[sid], epo[t]))
    direct = len(pairs)
    print(f"direct IO<->EO pairs: {direct}")

    # 2) pivot: ido -> bridge -> epo (bridge sentence shared)
    for bcode, bsent in bridges.items():
        for sid, itext in ido.items():
            for b in adj.get(sid, ()):  # noqa: B007
                if b not in bsent:
                    continue
                for e in adj.get(b, ()):
                    if e in epo:
                        pairs.add((itext, epo[e]))
    print(f"total after pivots: {len(pairs)} (+{len(pairs) - direct})")

    n = 0
    with open(OUT / "tatoeba_io_eo.jsonl", "w", encoding="utf-8") as f:
        for itext, etext in sorted(pairs):
            if not itext or not etext:
                continue
            for src_lang, tgt_lang, src, tgt in (
                (IDO, EPO, itext, etext),
                (EPO, IDO, etext, itext),
            ):
                f.write(json.dumps({
                    "src_lang": src_lang, "tgt_lang": tgt_lang,
                    "src": src, "tgt": tgt, "provenance": "tatoeba",
                }, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} directed records to {OUT / 'tatoeba_io_eo.jsonl'}")


if __name__ == "__main__":
    main()
