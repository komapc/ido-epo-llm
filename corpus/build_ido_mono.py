#!/usr/bin/env python3
"""Build a clean monolingual Ido sentence pool for back-translation.

Usage: python3 corpus/build_ido_mono.py [--wiki PATH] [--fineweb PATH] [--per-template 2]

Sources:
  * Ido Wikipedia: the pages-articles dump the extractor already downloads
    (extractor/data/raw/iowiki-latest-pages-articles.xml.bz2), namespace 0, no redirects.
  * FineWeb-2 ido_Latn (HuggingFaceFW/fineweb-2, data/ido_Latn/train/000_00000.parquet,
    ODC-By), only documents NOT from wikipedia.org (those are covered by the dump).
    Reading the parquet needs pyarrow; without it, FineWeb is skipped.

Filters, in order:
  * sentence split on . ! ? followed by a capital; 4..40 words, <= 300 chars;
  * no markup left over ({ } [ ] | = < > http), at least 80% alphabetic words;
  * no Esperanto letters (ĉ ĝ ĥ ĵ ŝ ŭ) - FineWeb's language id confuses the two;
  * "Ido-ness": at least 35% of the words are among the 1,000 most frequent
    lower-case words of Tatoeba's Ido sentences (derived, not a hand-made list);
  * bot stubs: the template of a sentence (digits -> 0, non-initial capitalised
    words -> X) may occur at most --per-template times, and at most --per-prefix
    sentences may start with the same three words;
  * dedup by a punctuation-free lowercase key, and drop every sentence whose key
    is in a published dataset (v1-io-eo train/val/test, eval-ido-x-v1 dev/test)
    or in the reserved Tatoeba pairs.
Writes corpus/out/ido_mono.jsonl.gz: {"text", "source"} (source = iowiki | fineweb2:<host>).
"""
from __future__ import annotations

import argparse
import bz2
import collections
import gzip
import json
import re
import sys
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus.build_eval_ido_x import key  # noqa: E402
from corpus.tatoeba import load_sentences  # noqa: E402
from mtlib.paths import CACHE, DATASETS, OUT, WORKSPACE  # noqa: E402

WIKI = WORKSPACE / "extractor/data/raw/iowiki-latest-pages-articles.xml.bz2"
FINEWEB = CACHE / "fineweb2/ido_Latn_train_000_00000.parquet"
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"«(]?[A-ZÀ-Ý])")
WORD = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*")
BAD = re.compile(r"[{}\[\]|=<>]|https?:|www\.")
EO_LETTERS = re.compile(r"[ĉĝĥĵŝŭĈĜĤĴŜŬ]")


def wiki_pages(path: Path):
    """(title, wikitext) of every article (ns 0, not a redirect)."""
    with bz2.open(path, "rb") as f:
        title = ns = text = None
        redirect = False
        for _, el in ET.iterparse(f, events=("end",)):
            tag = el.tag.rsplit("}", 1)[-1]
            if tag == "title":
                title = el.text
            elif tag == "ns":
                ns = el.text
            elif tag == "redirect":
                redirect = True
            elif tag == "text":
                text = el.text or ""
            elif tag == "page":
                if ns == "0" and not redirect:
                    yield title, text
                title = ns = text = None
                redirect = False
                el.clear()


def strip_wiki(text: str) -> str:
    import mwparserfromhell
    code = mwparserfromhell.parse(text)
    for t in code.filter_templates(recursive=False):
        try:
            code.remove(t)
        except ValueError:
            pass
    plain = code.strip_code(normalize=True, collapse=True)
    # drop list items, headings and table/file leftovers line by line
    lines = [l.strip() for l in plain.splitlines()]
    return "\n".join(l for l in lines if l and not l.startswith(("*", "#", ":", ";", "!", "Kategorio:", "Arkivo:", "File:")))


def sentences(text: str):
    for para in text.split("\n"):
        for s in SENT_SPLIT.split(para.strip()):
            s = re.sub(r"\s+", " ", s).strip()
            if s:
                yield s


def template(s: str) -> str:
    words = s.split()
    out = [words[0]] + ["X" if w[:1].isupper() else w for w in words[1:]]
    return re.sub(r"\d+", "0", " ".join(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--wiki", type=Path, default=WIKI)
    ap.add_argument("--fineweb", type=Path, default=FINEWEB)
    ap.add_argument("--per-template", type=int, default=2)
    ap.add_argument("--per-prefix", type=int, default=100)
    args = ap.parse_args()

    # Ido-ness vocabulary from Tatoeba Ido sentences
    freq = collections.Counter(w.lower() for s in load_sentences("ido").values() for w in WORD.findall(s))
    common = {w for w, _ in freq.most_common(1000)}

    # every text already in a published dataset or reserved for training
    used = set()
    for d in ("v1-io-eo", "eval-ido-x-v1"):
        for p in (DATASETS / d).glob("*.jsonl"):
            for line in p.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                used |= {key(r["input"]), key(r["output"])}
    for p in OUT.glob("*_reserved_train.jsonl"):
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            used |= {key(r["src"]), key(r["tgt"])}

    docs: list[tuple[str, str]] = []   # (source, plain text)
    pages = list(wiki_pages(args.wiki))
    print(f"iowiki: {len(pages)} articles; stripping markup…", flush=True)
    with Pool() as pool:
        for plain in pool.imap(strip_wiki, (t for _, t in pages), chunksize=200):
            docs.append(("iowiki", plain))
    try:
        import pyarrow.parquet as pq
        t = pq.read_table(args.fineweb, columns=["text", "url"]).to_pylist()
        fw = [r for r in t if "wikipedia.org" not in r["url"]]
        print(f"fineweb2: {len(t)} docs, {len(fw)} not from wikipedia")
        docs += [("fineweb2:" + r["url"].split("/")[2], r["text"]) for r in fw]
    except ImportError:
        print("pyarrow not installed: FineWeb-2 skipped")

    stats = collections.Counter()
    seen, per_tpl, per_prefix, out = set(), collections.Counter(), collections.Counter(), []
    for src, text in docs:
        for s in sentences(text):
            stats["sentences"] += 1
            words = s.split()
            if not 4 <= len(words) <= 40 or len(s) > 300 or BAD.search(s):
                stats["shape"] += 1
                continue
            alpha = WORD.findall(s)
            if len(alpha) < 0.8 * len(words):
                stats["non_alpha"] += 1
                continue
            if EO_LETTERS.search(s):
                stats["esperanto_letters"] += 1
                continue
            if sum(w.lower() in common for w in alpha) < 0.35 * len(alpha):
                stats["not_ido"] += 1
                continue
            k = key(s)
            if k in seen or k in used:
                stats["duplicate_or_used"] += 1
                continue
            tpl, prefix = template(s), " ".join(words[:3]).lower()
            if per_tpl[tpl] >= args.per_template or per_prefix[prefix] >= args.per_prefix:
                stats["template_cap"] += 1
                continue
            seen.add(k)
            per_tpl[tpl] += 1
            per_prefix[prefix] += 1
            out.append({"text": s, "source": src})
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "ido_mono.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as f:
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    print(json.dumps(stats, indent=1))
    print(collections.Counter(r["source"].split(":")[0] for r in out))
    print(f"kept {len(out)} sentences -> {path}")


if __name__ == "__main__":
    main()
