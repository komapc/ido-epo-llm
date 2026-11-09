"""Language registry: one place that maps a language between the naming
schemes the pipeline touches (Tatoeba ISO 639-3 codes, NLLB language tokens,
Apertium modes). Add a language here before using it anywhere else."""
from __future__ import annotations

from dataclasses import dataclass

from .paths import WORKSPACE


@dataclass(frozen=True)
class Lang:
    code: str   # Tatoeba / ISO 639-3, also used in dataset records (src_lang, tgt_lang)
    nllb: str   # NLLB-200 language token
    name: str   # English name, used in instruction strings


LANGS: dict[str, Lang] = {l.code: l for l in (
    Lang("ido", "ido_Latn", "Ido"),        # not in NLLB-200: a token we add when fine-tuning
    Lang("epo", "epo_Latn", "Esperanto"),
    Lang("eng", "eng_Latn", "English"),
    Lang("spa", "spa_Latn", "Spanish"),
    Lang("fra", "fra_Latn", "French"),
    Lang("deu", "deu_Latn", "German"),
    Lang("ita", "ita_Latn", "Italian"),
)}

# Directions with a rule-based Apertium baseline (the sibling apertium-ido-epo pair).
APERTIUM_PAIR_DIR = WORKSPACE / "apertium-ido-epo"
APERTIUM_MODES: dict[tuple[str, str], str] = {("ido", "epo"): "ido-epo", ("epo", "ido"): "epo-ido"}


def instruction(src: str, tgt: str) -> str:
    """Instruction string stored in dataset records, e.g. 'Translate Ido to Esperanto:'."""
    return f"Translate {LANGS[src].name} to {LANGS[tgt].name}:"
