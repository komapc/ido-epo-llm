"""Ido <-> Esperanto demo for komapc/nllb-ido-epo-600m (Hugging Face Space, CPU)."""
import re

import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

REPO = "komapc/nllb-ido-epo-600m"
MAX_CHARS = 1500

torch.set_num_threads(max(1, torch.get_num_threads()))
tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForSeq2SeqLM.from_pretrained(REPO, dtype=torch.float32).eval()
LANG = {"ido": tok.convert_tokens_to_ids("ido_Latn"), "epo": tok.convert_tokens_to_ids("epo_Latn")}
DIRS = {"Ido → Esperanto": ("ido", "epo"), "Esperanto → Ido": ("epo", "ido")}

# The model was trained on sentence pairs, so long input is split into sentences.
_SENT = re.compile(r"(?<=[.!?…])\s+(?=[\"«„(]?[A-ZĈĜĤĴŜŬ0-9])")


def _sentences(text):
    for line in text.splitlines():
        yield [s for s in _SENT.split(line.strip()) if s] if line.strip() else []


@torch.inference_mode()
def _translate_batch(sents, src, tgt):
    if not sents:
        return []
    eos, pad = tok.eos_token_id, tok.pad_token_id
    ids = [[LANG[src]] + tok(s, add_special_tokens=False)["input_ids"][:254] + [eos] for s in sents]
    width = max(map(len, ids))
    input_ids = torch.tensor([x + [pad] * (width - len(x)) for x in ids])
    attn = (input_ids != pad).long()
    out = model.generate(input_ids=input_ids, attention_mask=attn, forced_bos_token_id=LANG[tgt],
                         num_beams=4, max_new_tokens=256)
    return tok.batch_decode(out, skip_special_tokens=True)


def translate(text, direction):
    text = (text or "").strip()[:MAX_CHARS]
    if not text:
        return ""
    src, tgt = DIRS[direction]
    lines = list(_sentences(text))
    flat = [s for line in lines for s in line]
    done = iter(_translate_batch(flat, src, tgt))
    return "\n".join(" ".join(next(done) for _ in line) for line in lines)


def swap(direction, src_text, out_text):
    other = [d for d in DIRS if d != direction][0]
    return other, out_text or src_text, ""


with gr.Blocks(title="Ido ↔ Esperanto (NLLB)") as demo:
    gr.Markdown(
        "# Ido ↔ Esperanto\n"
        "Neural translation with [komapc/nllb-ido-epo-600m](https://huggingface.co/komapc/nllb-ido-epo-600m), "
        "which is NLLB-200 600M fine-tuned for Ido. Compare it with the rule-based Apertium translator at "
        "[tradukilo.lipu.dev](https://tradukilo.lipu.dev/).\n\n"
        "Runs on a free CPU, so expect a few seconds per sentence. The model was trained on short sentences."
    )
    direction = gr.Radio(list(DIRS), value="Ido → Esperanto", label="Direction")
    with gr.Row():
        src = gr.Textbox(label="Source", lines=6, max_length=MAX_CHARS, placeholder="Skribez hike…")
        out = gr.Textbox(label="Translation", lines=6)
    with gr.Row():
        btn = gr.Button("Translate", variant="primary")
        sw = gr.Button("⇄ Swap")
    gr.Examples(
        examples=[
            ["Me volas lernar Ido nam ol esas facila.", "Ido → Esperanto"],
            ["La hundo kuras rapide en la parko.", "Ido → Esperanto"],
            ["Ka vu parolas Esperanto?", "Ido → Esperanto"],
            ["Mi ne scias, ĉu li venos morgaŭ.", "Esperanto → Ido"],
            ["La urbo havas multajn belajn domojn.", "Esperanto → Ido"],
        ],
        inputs=[src, direction],
    )
    gr.Markdown(
        "Model license: **CC-BY-NC 4.0** (non-commercial). Training data includes "
        "[Tatoeba](https://tatoeba.org) sentences (CC-BY 2.0 FR). "
        "Code: [komapc/ido-epo-llm](https://github.com/komapc/ido-epo-llm)."
    )
    btn.click(translate, [src, direction], out, api_name="translate")
    src.submit(translate, [src, direction], out)
    sw.click(swap, [direction, src, out], [direction, src, out])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
