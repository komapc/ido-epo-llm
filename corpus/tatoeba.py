"""Tatoeba dumps: download (cached in corpus/cache/), per-language sentences and
the translation-link graph. Shared by the pair builder and the experiments."""
from __future__ import annotations

import bz2
import csv
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mtlib.paths import CACHE  # noqa: E402

BASE = "https://downloads.tatoeba.org/exports"


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
    """sentence_id -> text for one language (Tatoeba ISO 639-3 code)."""
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


def load_links(keep: set[int] | None = None) -> dict[int, set[int]]:
    """Symmetric adjacency of Tatoeba translation links. With `keep`, only links
    whose both ends are in it are stored (the full graph is ~30M links)."""
    path = fetch(f"{BASE}/links.tar.bz2", CACHE / "links.tar.bz2")
    adj: dict[int, set[int]] = {}
    n = 0
    with tarfile.open(path, "r:bz2") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("links.csv"))
        f = io.TextIOWrapper(tar.extractfile(member), encoding="utf-8")
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 2:
                continue
            a, b = int(row[0]), int(row[1])
            n += 1
            if keep is not None and (a not in keep or b not in keep):
                continue
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    print(f"  links: {n} total")
    return adj
