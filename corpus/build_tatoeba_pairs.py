#!/usr/bin/env python3
"""Build real sentence pairs between two languages from Tatoeba.

Tatoeba is the only source of genuine human Ido sentence pairs. We take direct
SRC<->TGT links plus pivots through bridge languages, where a SRC sentence and a
TGT sentence both translate the same bridge sentence.

Usage: python3 corpus/build_tatoeba_pairs.py [--src ido] [--tgt epo] [--bridges eng fra spa deu]

Writes corpus/out/tatoeba_<src>_<tgt>.jsonl with records
  {"src_lang","tgt_lang","src","tgt","provenance","src_id","tgt_id","via"}
in both directions, so a bidirectional model sees each pair twice. src_id and
tgt_id are Tatoeba sentence ids (the smallest ones, when a text occurs more than
once); "via" is the bridge language of a pivot pair, null for a direct link.
The ids let a split keep every sentence linked through Tatoeba on one side.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus.tatoeba import load_links, load_sentences  # noqa: E402
from mtlib.langs import LANGS  # noqa: E402
from mtlib.paths import OUT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", default="ido", choices=LANGS)
    ap.add_argument("--tgt", default="epo", choices=LANGS)
    ap.add_argument("--bridges", nargs="*", default=["eng", "fra", "spa", "deu"], choices=LANGS)
    args = ap.parse_args()
    bridges = [b for b in args.bridges if b not in (args.src, args.tgt)]

    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading sentences…")
    src = load_sentences(args.src)
    tgt = load_sentences(args.tgt)
    if not src:
        print(f"No {args.src} sentences from Tatoeba — aborting.", file=sys.stderr)
        sys.exit(1)
    bridge_sents = {b: load_sentences(b) for b in bridges}

    print("Loading links…")
    keep = set(src) | set(tgt)
    for s in bridge_sents.values():
        keep |= s.keys()
    adj = load_links(keep)

    # (src_text, tgt_text) -> (src_id, tgt_id, via); direct links win over pivots
    pairs: dict[tuple[str, str], tuple[int, int, str | None]] = {}

    def add(sid: int, tid: int, via: str | None) -> None:
        key = (src[sid], tgt[tid])
        old = pairs.get(key)
        new = (sid, tid, via)
        if old is None or (old[2] is not None and via is None) or (
                (old[2] is None) == (via is None) and new[:2] < old[:2]):
            pairs[key] = new

    # 1) direct links
    for sid in src:
        for t in adj.get(sid, ()):
            if t in tgt:
                add(sid, t, None)
    direct = len(pairs)
    print(f"direct {args.src}<->{args.tgt} pairs: {direct}")

    # 2) pivots: src -> bridge -> tgt (bridge sentence shared)
    for bcode in bridges:
        bsent = bridge_sents[bcode]
        for sid in src:
            for b in adj.get(sid, ()):
                if b not in bsent:
                    continue
                for t in adj.get(b, ()):
                    if t in tgt:
                        add(sid, t, bcode)
    print(f"total after pivots: {len(pairs)} (+{len(pairs) - direct})")

    path = OUT / f"tatoeba_{args.src}_{args.tgt}.jsonl"
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for (stext, ttext), (sid, tid, via) in sorted(pairs.items()):
            if not stext or not ttext:
                continue
            for sl, tl, s, t, si, ti in ((args.src, args.tgt, stext, ttext, sid, tid),
                                         (args.tgt, args.src, ttext, stext, tid, sid)):
                f.write(json.dumps({
                    "src_lang": sl, "tgt_lang": tl, "src": s, "tgt": t, "provenance": "tatoeba",
                    "src_id": si, "tgt_id": ti, "via": via,
                }, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} directed records to {path}")


if __name__ == "__main__":
    main()
