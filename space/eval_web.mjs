// Score a browser (transformers.js) build under Node: greedy, q8, as in the browser.
// npm i @huggingface/transformers@4.3.1, then:
// node space/eval_web.mjs MODELS_DIR VIEW OUT_DIR file1.jsonl[:src-tgt] ...
import { env, AutoTokenizer, AutoModelForSeq2SeqLM, Tensor } from '@huggingface/transformers';
import fs from 'node:fs'; import path from 'node:path';
const [,, modelsDir, view, outDir, ...files] = process.argv;
env.localModelPath = modelsDir; env.allowRemoteModels = false;
const tok = await AutoTokenizer.from_pretrained(view);
const model = await AutoModelForSeq2SeqLM.from_pretrained(view, { dtype: 'q8' });
const NLLB = { ido: 'ido_Latn', epo: 'epo_Latn', eng: 'eng_Latn', spa: 'spa_Latn', fra: 'fra_Latn', deu: 'deu_Latn' };
const id = (l) => tok.convert_tokens_to_ids(NLLB[l]);
fs.mkdirSync(outDir, { recursive: true });
for (const spec of files) {
  const [file, filt] = spec.split(':');               // filt = "src-tgt" to select rows
  let rows = fs.readFileSync(file, 'utf8').trim().split('\n').map(JSON.parse);
  if (filt) { const [s, t] = filt.split('-'); rows = rows.filter(r => r.src_lang === s && r.tgt_lang === t); }
  const t0 = Date.now(); const out = [];
  for (const r of rows) {
    const ids = [id(r.src_lang), ...tok.encode(r.input, { add_special_tokens: false }), 2];
    const input_ids = new Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, ids.length]);
    const attention_mask = new Tensor('int64', BigInt64Array.from(ids.map(() => 1n)), [1, ids.length]);
    const res = await model.generate({ inputs: input_ids, attention_mask, forced_bos_token_id: id(r.tgt_lang), max_new_tokens: 128, num_beams: 1 });
    out.push({ input: r.input, ref: r.ref ?? r.output, output: tok.decode(res[0], { skip_special_tokens: true }).trim() });
  }
  const name = path.basename(file, '.jsonl') + (filt ? '.' + filt : '');
  fs.writeFileSync(path.join(outDir, name + '.jsonl'), out.map(o => JSON.stringify(o)).join('\n') + '\n');
  console.log(name, rows.length, ((Date.now() - t0) / rows.length).toFixed(0) + ' ms/sent', 'e.g.', out[0].output);
}
