#!/usr/bin/env python3
"""Build frozen dev/test sets for Ido <-> English, Spanish, French, German from Tatoeba.

Usage: python3 corpus/build_eval_ido_x.py [--name eval-ido-x-v1] [--reserve-frac 0.5]

The v1 dataset (datasets/v1-io-eo) used almost every Tatoeba Ido sentence that
has an Esperanto route, so a clean Ido<->X set has to avoid it. Candidates are Ido
sentences with a DIRECT Tatoeba link to an X sentence. One is kept only if no
text in its link neighbourhood (the Ido sentence, its direct translations in
ido/epo/eng/spa/fra/deu, and their direct translations) is in v1 train, compared
after lowercasing and stripping punctuation. That also catches an X sentence that
translates a different, trained-on Ido sentence.

Where a kept sentence goes:
  * Ido side in v1 val  -> dev
  * Ido side in v1 test -> test
  * Ido side not in v1 at all ("new") -> test, except that for languages with
    more than 500 new sentences (English, German) a --reserve-frac share is held
    back as future training data (written to corpus/out/, not published).
Rows with either side over 200 characters are dropped (a few Tatoeba entries are
whole paragraphs); this happens after the split, so it only removes rows. Reserved
pairs that share a Tatoeba id or a text with any dev/test row are dropped too.

One reference per Ido sentence: the X sentence with the smallest Tatoeba id.
Writes datasets/<name>/{dev,test}.jsonl (both directions, instruction format,
provenance "tatoeba") and SHA256SUMS. The directory must not exist yet: a
published eval set is frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus.tatoeba import load_links, load_sentences  # noqa: E402
from mtlib.langs import instruction  # noqa: E402
from mtlib.paths import DATASETS, OUT, dataset_dir  # noqa: E402

TARGETS = ["eng", "spa", "fra", "deu"]
NEIGHBOUR_LANGS = ["ido", "epo", *TARGETS]
SEED = 42
RESERVE_MIN = 500   # only hold back training data where the language has this many new sentences
MAX_CHARS = 200     # longer rows are paragraphs, not sentences


def key(s: str) -> str:
    """Comparison key: lowercase, no punctuation, single spaces."""
    s = unicodedata.normalize("NFC", s).lower()
    s = "".join(c if not unicodedata.category(c).startswith("P") else " " for c in s)
    return re.sub(r"\s+", " ", s).strip()


def v1_keys(split: str) -> set[str]:
    out = set()
    for line in (dataset_dir("v1-io-eo") / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out |= {key(r["input"]), key(r["output"])}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", default="eval-ido-x-v1")
    ap.add_argument("--reserve-frac", type=float, default=0.5)
    args = ap.parse_args()
    out_dir = DATASETS / args.name
    if out_dir.exists():
        sys.exit(f"{out_dir} exists; published eval sets are frozen")

    sents = {l: load_sentences(l) for l in NEIGHBOUR_LANGS}
    lang_of = {sid: l for l, d in sents.items() for sid in d}
    adj = load_links(set(lang_of))
    text = lambda sid: sents[lang_of[sid]][sid]  # noqa: E731

    train, val, test = v1_keys("train"), v1_keys("val"), v1_keys("test")

    def neighbourhood(sid: int) -> set[int]:
        first = {t for t in adj.get(sid, ()) if t in lang_of}
        second = {u for t in first for u in adj.get(t, ()) if u in lang_of}
        return {sid} | first | second

    rng = random.Random(SEED)
    stats, records = {}, {"dev": [], "test": []}
    reserve = []
    for x in TARGETS:
        cand = {"dev": [], "test": [], "new": []}
        dropped = 0
        for sid, s in sorted(sents["ido"].items()):
            refs = sorted(t for t in adj.get(sid, ()) if t in sents[x])
            if not refs or not s.strip() or not text(refs[0]).strip():
                continue
            if any(key(text(n)) in train for n in neighbourhood(sid)):
                dropped += 1
                continue
            k = key(s)
            bucket = "dev" if k in val else "test" if k in test else "new"
            cand[bucket].append((sid, refs[0]))
        new = cand["new"]
        rng.shuffle(new)
        n_res = int(len(new) * args.reserve_frac) if len(new) > RESERVE_MIN else 0
        res, new_test = new[:n_res], new[n_res:]
        split = {"dev": cand["dev"], "test": cand["test"] + sorted(new_test)}
        stats[x] = {"leak_dropped": dropped, "dev": len(split["dev"]), "test": len(split["test"]),
                    "test_from_v1_test": len(cand["test"]), "test_new": len(new_test), "reserved": len(res)}
        for name, pairs in split.items():
            seen_x = set()
            for sid, tid in pairs:
                s, t = text(sid).strip(), text(tid).strip()
                if len(s) > MAX_CHARS or len(t) > MAX_CHARS:
                    stats[x]["too_long"] = stats[x].get("too_long", 0) + 1
                    continue
                records[name].append({"instruction": instruction("ido", x), "input": s, "output": t,
                                      "src_lang": "ido", "tgt_lang": x, "provenance": "tatoeba",
                                      "src_id": sid, "tgt_id": tid})
                if key(t) not in seen_x:   # X->Ido: one row per X sentence
                    seen_x.add(key(t))
                    records[name].append({"instruction": instruction(x, "ido"), "input": t, "output": s,
                                          "src_lang": x, "tgt_lang": "ido", "provenance": "tatoeba",
                                          "src_id": tid, "tgt_id": sid})
        reserve += [{"src_lang": "ido", "tgt_lang": x, "src": text(sid), "tgt": text(tid),
                     "provenance": "tatoeba", "src_id": sid, "tgt_id": tid} for sid, tid in res]

    # The dev and test sets must not share any text either.
    dev_k = {key(r["input"]) for r in records["dev"]} | {key(r["output"]) for r in records["dev"]}
    clash = [r for r in records["test"] if key(r["input"]) in dev_k or key(r["output"]) in dev_k]
    if clash:
        print(f"dropping {len(clash)} test rows that share text with dev")
        records["test"] = [r for r in records["test"] if r not in clash]

    # Reserved training pairs must not touch any eval row, by Tatoeba id or by text.
    ev = records["dev"] + records["test"]
    ev_ids = {r["src_id"] for r in ev} | {r["tgt_id"] for r in ev}
    ev_keys = {key(r["input"]) for r in ev} | {key(r["output"]) for r in ev}
    n_res = len(reserve)
    reserve = [r for r in reserve if not ({r["src_id"], r["tgt_id"]} & ev_ids)
               and key(r["src"]) not in ev_keys and key(r["tgt"]) not in ev_keys]
    print(f"reserved pairs: {n_res}, {n_res - len(reserve)} dropped for touching dev/test")

    out_dir.mkdir(parents=True)
    sums = []
    for name, rows in records.items():
        p = out_dir / f"{name}.jsonl"
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (out_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.name}_reserved_train.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reserve), encoding="utf-8")

    by_dir = defaultdict(int)
    for name, rows in records.items():
        for r in rows:
            by_dir[(name, r["src_lang"], r["tgt_lang"])] += 1
    print(json.dumps(stats, indent=1))
    for k in sorted(by_dir):
        print(*k, by_dir[k])
    print(f"wrote {out_dir}; {len(reserve)} reserved pairs -> {OUT / (args.name + '_reserved_train.jsonl')}")


if __name__ == "__main__":
    main()
