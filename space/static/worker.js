// Runs the model off the main thread. Messages in:  {type:"load"} | {type:"translate", id, text, src, tgt}
// Messages out: {type:"progress", file, loaded, total} | {type:"ready"} | {type:"result", id, text} | {type:"error", message}
import { env, AutoTokenizer, AutoModelForSeq2SeqLM, Tensor }
  from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1/dist/transformers.min.js";

const MODEL = "komapc/nllb-ido-epo-600m-onnx";
env.allowLocalModels = false;

let tok, model, LANG;

async function load() {
  if (model) return;
  const progress_callback = (p) => {
    if (p.status === "progress") postMessage({ type: "progress", file: p.file, loaded: p.loaded, total: p.total });
  };
  tok = await AutoTokenizer.from_pretrained(MODEL, { progress_callback });
  model = await AutoModelForSeq2SeqLM.from_pretrained(MODEL, { dtype: "q8", device: "wasm", progress_callback });
  LANG = { ido: tok.convert_tokens_to_ids("ido_Latn"), epo: tok.convert_tokens_to_ids("epo_Latn") };
}

// The model was trained on sentence pairs, so long input is split into sentences.
const SENT = /(?<=[.!?…])\s+(?=["«„(]?[A-ZĈĜĤĴŜŬ0-9])/u;

async function translateSentence(s, src, tgt) {
  const ids = [LANG[src], ...tok.encode(s, { add_special_tokens: false }).slice(0, 254), tok.eos_token_id];
  const shape = [1, ids.length];
  const input_ids = new Tensor("int64", BigInt64Array.from(ids.map(BigInt)), shape);
  const attention_mask = new Tensor("int64", new BigInt64Array(ids.length).fill(1n), shape);
  const out = await model.generate({ inputs: input_ids, attention_mask, forced_bos_token_id: LANG[tgt], max_new_tokens: 256 });
  return tok.decode(out[0], { skip_special_tokens: true }).trim();
}

async function translate(text, src, tgt) {
  const lines = [];
  for (const line of text.split("\n")) {
    const parts = line.trim() ? line.trim().split(SENT).filter(Boolean) : [];
    const done = [];
    for (const s of parts) done.push(await translateSentence(s, src, tgt));
    lines.push(done.join(" "));
  }
  return lines.join("\n");
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      await load();
      postMessage({ type: "ready" });
    } else if (data.type === "translate") {
      await load();
      postMessage({ type: "result", id: data.id, text: await translate(data.text, data.src, data.tgt) });
    }
  } catch (e) {
    postMessage({ type: "error", id: data.id, message: String(e?.message || e) });
  }
};
