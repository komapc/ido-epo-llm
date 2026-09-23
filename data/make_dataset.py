#!/usr/bin/env python3
"""Merge sources into train/val/test jsonl for instruction fine-tuning.

Inputs (data/out/):
  tatoeba_io_eo.jsonl   real human pairs (provenance "tatoeba")
  apertium_synth.jsonl  weak-supervision pairs (provenance "apertium_synth")

Rules:
  * Normalize whitespace; drop empties, near-duplicates (by normalized src+tgt),
    and pairs with a leading "*" Tatoeba marker or wild length ratios.
  * Eval (val/test) is drawn ONLY from real DIRECT+pivot pairs — never
    synthetic — so the metric reflects true generalization vs Apertium.
  * Output instruction format consumed by train/finetune_colab.ipynb:
      {"instruction","input","output","src_lang","tgt_lang","provenance","weight"}
    Real pairs get weight 3, synthetic 1 (so the trainer can upweight real
    data by repetition or a weighted loss).

Split strategy — leakage-safe "units" via union-find over raw sentence text:
  Tatoeba pairs are emitted in both directions (io->eo AND eo->io for the
  same underlying sentence pair), and it's also possible for two distinct
  pairs to share one side (e.g. the same short Esperanto rendering reached
  through two different pivot bridges) — and, since Ido and Esperanto are
  typologically close, a short sentence can be the *identical string* in
  both languages. Any of these cases means the same text can resurface
  under a different (src_lang, src) key, so filtering eval-leakage by
  "(src_lang, src) already used in eval" (the previous approach) misses it.

  Instead: build a graph where every pair (Tatoeba, either direction, or
  synthetic) is an edge between its src text and tgt text, take connected
  components (union-find), and assign a whole component to exactly one of
  train/val/test. A component this way is the transitive closure of "these
  sentences co-occur in some pair somewhere in the corpus", so no sentence
  text can end up in two different splits: the io->eo/eo->io mirror of one
  pair is always kept together, and so is any other pair that happens to
  reuse one of its sides.

  This is checked empirically before relying on it (see dev notes / report):
  the component-size distribution here has no "hub" text connecting a large
  fraction of the corpus (largest component observed: 17 texts out of
  ~34k), so component-level splitting costs very little data. If a future
  data refresh changes that, main() prints a warning rather than silently
  producing a near-unsplittable graph.
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
SEED = 42
VAL_N = 1000
TEST_N = 1000
MAX_LEN = 400          # chars; drop runaway sentences
LEN_RATIO = 3.0        # drop pairs where one side is >3x the other

INSTR = {
    ("ido", "epo"): "Translate Ido to Esperanto:",
    ("epo", "ido"): "Translate Esperanto to Ido:",
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def read(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  (skip) {path.name} not found")
        return []
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"  {path.name}: {len(rows)} records")
    return rows


def acceptable(src: str, tgt: str) -> bool:
    if not src or not tgt:
        return False
    if len(src) > MAX_LEN or len(tgt) > MAX_LEN:
        return False
    if src.startswith("*") or tgt.startswith("*"):
        return False
    a, b = len(src), len(tgt)
    if a and b and (a / b > LEN_RATIO or b / a > LEN_RATIO):
        return False
    return True


def clean(rows: list[dict]) -> list[dict]:
    """Normalize + dedup a record list, keyed on (src_lang, norm(src), norm(tgt))."""
    seen, out = set(), []
    for r in rows:
        s, t = norm(r["src"]), norm(r["tgt"])
        if not acceptable(s, t):
            continue
        key = (r["src_lang"], s, t)
        if key in seen:
            continue
        seen.add(key)
        out.append({**r, "src": s, "tgt": t})
    return out


class UnionFind:
    """Union-find over raw sentence text (not (lang, text)): Ido and Esperanto
    are close enough that short sentences are sometimes identical strings in
    both languages, so keying on text alone is what actually prevents leakage.
    """

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def main() -> None:
    rng = random.Random(SEED)
    tat = read(OUT / "tatoeba_io_eo.jsonl")
    syn = read(OUT / "apertium_synth.jsonl")

    tat, syn = clean(tat), clean(syn)
    print(f"after clean/dedup: tatoeba={len(tat)} synth={len(syn)}")

    # --- build leakage-safe "units" -------------------------------------
    uf = UnionFind()
    for r in tat + syn:
        uf.union(r["src"], r["tgt"])

    comp_size: dict[str, int] = defaultdict(int)
    for x in uf.parent:
        comp_size[uf.find(x)] += 1
    n_texts = len(uf.parent)
    biggest = max(comp_size.values(), default=0)
    print(f"text graph: {n_texts} distinct texts, {len(comp_size)} components, "
          f"largest={biggest} ({100 * biggest / max(n_texts, 1):.2f}%)")
    if n_texts and biggest / n_texts > 0.05:
        print(f"WARNING: largest text component is {100 * biggest / n_texts:.1f}% "
              "of all sentence texts — a 'hub' sentence may be linking most of "
              "the corpus into one component, which would make component-level "
              "splitting too coarse (val/test could balloon or shrink sharply). "
              "Investigate before trusting the split sizes below.")

    # Only Tatoeba (real) rows are eval-eligible; group them by component so
    # a whole component (mirrors + any transitively-linked pairs) moves together.
    comp_tat: dict[str, list[dict]] = defaultdict(list)
    for r in tat:
        comp_tat[uf.find(r["src"])].append(r)

    # Candidate eval components: shorter/cleaner third, as before, but chosen
    # at component granularity so a component never gets split across sets.
    def comp_len(rows: list[dict]) -> int:
        return min(len(r["src"]) + len(r["tgt"]) for r in rows)

    candidates = sorted(comp_tat.items(), key=lambda kv: comp_len(kv[1]))
    pool_n = max(1, max(VAL_N + TEST_N, len(candidates) // 3))
    pool = candidates[:pool_n]
    rng.shuffle(pool)

    val_comps: set[str] = set()
    test_comps: set[str] = set()
    val_rows: list[dict] = []
    test_rows: list[dict] = []
    for comp_id, rows in pool:
        if len(val_rows) < VAL_N:
            val_comps.add(comp_id)
            val_rows.extend(rows)
        elif len(test_rows) < TEST_N:
            test_comps.add(comp_id)
            test_rows.extend(rows)
        else:
            break
    eval_comps = val_comps | test_comps
    print(f"eval components: val={len(val_comps)} test={len(test_comps)} "
          f"-> val_rows={len(val_rows)} test_rows={len(test_rows)}")

    # Train = every tat/synth row whose component was NOT chosen for eval.
    # Because union() above ties together every text that ever co-occurs in
    # ANY pair, this is enough (not just "best effort") to guarantee no eval
    # sentence text -- source or target, either direction -- recurs in train.
    train = []
    dropped_syn = 0
    for r in tat:
        if uf.find(r["src"]) in eval_comps:
            continue
        train.append({**r, "weight": 3})
    for r in syn:
        if uf.find(r["src"]) in eval_comps:
            dropped_syn += 1
            continue
        train.append({**r, "weight": 1})
    rng.shuffle(train)
    print(f"synthetic rows dropped for sharing a component with eval: {dropped_syn}")

    def to_instruct(r):
        return {
            "instruction": INSTR[(r["src_lang"], r["tgt_lang"])],
            "input": r["src"], "output": r["tgt"],
            "src_lang": r["src_lang"], "tgt_lang": r["tgt_lang"],
            "provenance": r["provenance"], "weight": r.get("weight", 3),
        }

    for name, rows in (("train", train), ("val", val_rows), ("test", test_rows)):
        path = OUT / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(to_instruct(r), ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} -> {path}")

    real_train = sum(1 for r in train if r["provenance"] == "tatoeba")
    print(f"train: {real_train} real + {len(train) - real_train} synthetic "
          f"(eval is real-only, {len(val_rows) + len(test_rows)} pairs)")

    # --- self-check: the invariant this whole scheme exists to guarantee ---
    train_texts = {r["src"] for r in train} | {r["tgt"] for r in train}
    for split_name, rows in (("val", val_rows), ("test", test_rows)):
        leaked = [r for r in rows if r["src"] in train_texts or r["tgt"] in train_texts]
        if leaked:
            raise AssertionError(f"{len(leaked)} {split_name} rows leak into the train text set")
    print("leakage self-check passed: no val/test sentence text (either side, "
          "either direction) appears anywhere in train")


if __name__ == "__main__":
    main()
