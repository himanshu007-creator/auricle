// Measures each stage of the response path against your own OpenAI-compatible
// endpoint, then prints the modelled gap from "user stops talking" to "first
// audio" with and without Auricle's eager modes. Same adapter code the browser runs.
//
//   OPENAI_BASE_URL=https://api.groq.com/openai/v1 OPENAI_API_KEY=gsk_... \
//   STT=whisper-large-v3-turbo LLM=openai/gpt-oss-20b TTS=canopylabs/orpheus-v1-english VOICE=autumn \
//   node bench/latency.mjs
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const { OpenAICompat } = await import(join(HERE, '../web/src/backend/openai.ts'));
const { encodeWav } = await import(join(HERE, '../web/src/wav.ts'));
const { takeSentences } = await import(join(HERE, '../web/src/heard.ts'));

const env = (k, d) => process.env[k] ?? d;
if (!process.env.OPENAI_API_KEY) {
  console.error('set OPENAI_API_KEY (and optionally OPENAI_BASE_URL, STT, LLM, TTS, VOICE)');
  process.exit(2);
}
const backend = new OpenAICompat({
  baseURL: env('OPENAI_BASE_URL', 'https://api.groq.com/openai/v1'),
  apiKey: process.env.OPENAI_API_KEY,
  models: { stt: env('STT', 'whisper-large-v3-turbo'), llm: env('LLM', 'openai/gpt-oss-20b'), tts: env('TTS', 'canopylabs/orpheus-v1-english') },
  voice: env('VOICE', 'autumn'),
  extraBody: process.env.EXTRA_BODY ? JSON.parse(process.env.EXTRA_BODY) : { reasoning_effort: 'low' },
});
const ROUNDS = Number(env('ROUNDS', 5));
const HANGOVER_MS = 240;

// A real 2 s speech segment: the bundled clip's first two seconds (16 kHz).
const wav = readFileSync(join(HERE, '../demo/public/pretend-agent.wav'));
const pcm = new Int16Array(wav.buffer, wav.byteOffset + 44, (wav.length - 44) / 2).subarray(0, 32000);
const segment = encodeWav(Float32Array.from(pcm, (v) => v / 32768), 16000);

const p50 = (a) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];
const timed = async (fn) => {
  const t = performance.now();
  const v = await fn();
  return { ms: performance.now() - t, v };
};

await backend.warmUp();
const stt = [];
const llmFirstToken = [];
const llmFirstSentence = [];
const llmFirstClause = [];
const ttsFirstByte = [];
const ttsTotal = [];
let transcript = '';
for (let i = 0; i < ROUNDS; i++) {
  const r = await timed(() => backend.transcribe(segment));
  stt.push(r.ms);
  transcript = r.v;

  const t0 = performance.now();
  let text = '';
  let first = null;
  let clause = null;
  let sentence = null;
  for await (const delta of backend.chat([{ role: 'system', content: 'You are a voice assistant. Keep answers short and conversational.' }, { role: 'user', content: 'What is a good name for a small grey cat, and why?' }])) {
    text += delta;
    first ??= performance.now() - t0;
    if (clause === null && takeSentences(text, 12, true).sentences.length) clause = performance.now() - t0;
    if (sentence === null && takeSentences(text, 20, false).sentences.length) sentence = performance.now() - t0;
  }
  llmFirstToken.push(first);
  llmFirstClause.push(clause ?? performance.now() - t0);
  llmFirstSentence.push(sentence ?? performance.now() - t0);

  const t1 = performance.now();
  let firstByte = null;
  let samples = 0;
  for await (const chunk of backend.speak('Sure thing, I can book that for you tomorrow at seven.')) {
    firstByte ??= performance.now() - t1;
    samples += chunk.samples.length;
  }
  ttsFirstByte.push(firstByte);
  ttsTotal.push(performance.now() - t1);
  process.stderr.write(`round ${i + 1}/${ROUNDS} done\n`);
}

console.log(`endpoint ${backend.config.baseURL}, ${ROUNDS} rounds, medians:`);
console.log(`  STT round trip (2 s segment -> "${transcript}")   ${p50(stt).toFixed(0)} ms`);
console.log(`  LLM first token                                   ${p50(llmFirstToken).toFixed(0)} ms`);
console.log(`  LLM first clause ready (eager chunk)              ${p50(llmFirstClause).toFixed(0)} ms`);
console.log(`  LLM first sentence ready                          ${p50(llmFirstSentence).toFixed(0)} ms`);
console.log(`  TTS first audio byte                              ${p50(ttsFirstByte).toFixed(0)} ms   (whole sentence ${p50(ttsTotal).toFixed(0)} ms)`);
const serial = HANGOVER_MS + p50(stt) + p50(llmFirstSentence) + p50(ttsFirstByte);
const eagerStt = Math.max(HANGOVER_MS, p50(stt)) - HANGOVER_MS; // STT started at the silence candidate; only the part beyond the hangover is waited for
const eager = HANGOVER_MS + eagerStt + p50(llmFirstClause) + p50(ttsFirstByte);
console.log(`\nmodelled gap from "user stops talking" to first audio:`);
console.log(`  serial (hangover + STT + first sentence + TTS first byte)       ${serial.toFixed(0)} ms`);
console.log(`  Auricle eager (STT during hangover, first clause to TTS)        ${eager.toFixed(0)} ms`);
console.log(`\nnetwork from this machine; the browser adds its own fetch overhead. Compare with the demo's agent bubble, which shows the measured number.`);
process.exit(0);
