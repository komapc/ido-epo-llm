"""chrF (char n-gram F, n=6, beta=2) and BLEU-4, self-contained, no deps.

chrF is the standard corpus-POOLED statistic (Popovic 2015, and what sacrebleu's
default corpus chrF reports): n-gram match/total counts are summed over the
whole corpus for each order 1..6, precision and recall are each averaged across
orders, and a single F-beta is taken from those two averages. Not verified
against sacrebleu (not a dependency); numbers are expected to track it closely.
"""
from __future__ import annotations

import math
from collections import Counter


def char_ngrams(s: str, n: int) -> Counter:
    s = s.replace(" ", "")
    return Counter(s[i:i + n] for i in range(len(s) - n + 1)) if len(s) >= n else Counter()


def _order_prec_rec(hyps, refs, n: int) -> list[tuple[int, int, int, int]]:
    """Per-order (match, hyp_total, match, ref_total) counts, pooled over all
    hyp/ref pairs given (a single pair for sentence-level, the whole corpus
    for corpus-level)."""
    totals = [[0, 0, 0, 0] for _ in range(n)]  # match, hyp_total, match, ref_total
    for hyp, ref in zip(hyps, refs):
        for k in range(1, n + 1):
            h, r = char_ngrams(hyp, k), char_ngrams(ref, k)
            match = sum((h & r).values())
            t = totals[k - 1]
            t[0] += match
            t[1] += sum(h.values())
            t[2] += match
            t[3] += sum(r.values())
    return totals


def _fbeta_from_totals(totals, beta: float) -> float:
    """Standard (Popovic 2015 / sacrebleu) chrF: average precision and recall
    across n-gram orders separately, THEN take a single F-beta — not an
    average of per-order F-beta scores (which over-weights orders that
    happen to have low precision*recall)."""
    precs = [m / d for m, d, _, _ in totals if d]
    recs = [m / d for _, _, m, d in totals if d]
    if not precs or not recs:
        return 0.0
    prec, rec = sum(precs) / len(precs), sum(recs) / len(recs)
    if prec + rec == 0:
        return 0.0
    b2 = beta * beta
    return 100 * (1 + b2) * prec * rec / (b2 * prec + rec)


def chrf(hyp: str, ref: str, n: int = 6, beta: float = 2.0) -> float:
    """Sentence-level chrF (see corpus_chrf for the corpus-level metric this
    script actually reports)."""
    if not hyp or not ref:
        return 0.0
    return _fbeta_from_totals(_order_prec_rec([hyp], [ref], n), beta)


def corpus_chrf(hyps, refs, n: int = 6, beta: float = 2.0) -> float:
    """Corpus-level chrF: n-gram counts are pooled over the WHOLE corpus per
    order before computing precision/recall/F (not a mean of per-sentence
    chrF scores). This is what makes it comparable to sacrebleu's chrF."""
    if not hyps:
        return 0.0
    return _fbeta_from_totals(_order_prec_rec(hyps, refs, n), beta)


def bleu(hyps, refs, n=4) -> float:
    """Corpus BLEU-4 with a brevity penalty; token = whitespace."""
    p_num = [0] * n
    p_den = [0] * n
    hyp_len = ref_len = 0
    for h, r in zip(hyps, refs):
        ht, rt = h.split(), r.split()
        hyp_len += len(ht)
        ref_len += len(rt)
        for k in range(n):
            hg = Counter(tuple(ht[i:i + k + 1]) for i in range(len(ht) - k))
            rg = Counter(tuple(rt[i:i + k + 1]) for i in range(len(rt) - k))
            p_num[k] += sum((hg & rg).values())
            # No max(…, 1): a sentence shorter than k+1 tokens has no k-grams,
            # and counting a phantom one capped identical output below 100.
            p_den[k] += sum(hg.values())
    if min(p_num) == 0:
        return 0.0
    logp = sum(math.log(p_num[k] / p_den[k]) for k in range(n)) / n
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return 100 * bp * math.exp(logp)
