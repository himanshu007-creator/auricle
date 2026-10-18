// Same clip, same Silero v6 weights, two runtimes:
//   - Auricle brain WASM (Rust, SIMD128), the module the browser worker runs
//   - onnxruntime-web WASM backend (SIMD, 1 thread), exactly what @ricky0123/vad-web runs
// Node's V8 executes the same WASM the browsers do; numbers are per 32 ms frame.
//
//   cd bench && npm install && npm run bench          (after ../scripts/build-wasm.sh)
import { readFileSync, statSync } from 'node:fs';
import { gzipSync } from 'node:zlib';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import * as ort from 'onnxruntime-web';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');
const BRAIN_DIR = join(ROOT, 'web/src/generated/brain');
const VAD_WEB = join(HERE, 'node_modules/@ricky0123/vad-web/dist');
const ORT_DIST = join(HERE, 'node_modules/onnxruntime-web/dist');
const MODEL = join(VAD_WEB, 'silero_vad_v6.onnx');
const PASSES = 5;

const wav = readFileSync(join(ROOT, 'demo/public/pretend-agent.wav'));
const pcm = new Int16Array(wav.buffer, wav.byteOffset + 44, (wav.length - 44) / 2);
const samples = Float32Array.from(pcm, (v) => v / 32768);
const FRAME = 512;
const frames = Array.from({ length: Math.floor(samples.length / FRAME) }, (_, i) => samples.subarray(i * FRAME, (i + 1) * FRAME));

const stats = (times) => {
  const s = [...times].sort((a, b) => a - b);
  const pct = (p) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
  return { mean: s.reduce((a, b) => a + b, 0) / s.length, p50: pct(0.5), p95: pct(0.95), max: s[s.length - 1] };
};
const fmt = (r) => `p50 ${r.p50.toFixed(3)} ms  p95 ${r.p95.toFixed(3)} ms  max ${r.max.toFixed(3)} ms  mean ${r.mean.toFixed(3)} ms`;
const gz = (file) => gzipSync(readFileSync(file)).length;
const kb = (n) => `${(n / 1024).toFixed(0)} KB`;

async function benchAuricle() {
  const t0 = performance.now();
  const mod = await import(join(BRAIN_DIR, 'auricle_brain.js'));
  await mod.default({ module_or_path: readFileSync(join(BRAIN_DIR, 'auricle_brain_bg.wasm')) });
  const brain = new mod.Brain();
  const startupMs = performance.now() - t0;
  const probs = [];
  const times = [];
  for (let pass = 0; pass < PASSES; pass++) {
    brain.reset();
    for (const f of frames) {
      const t = performance.now();
      const p = brain.process(f);
      brain.take_events();
      if (pass > 0) times.push(performance.now() - t);
      else probs.push(p);
    }
  }
  return { probs, startupMs, ...stats(times) };
}

async function benchOrt() {
  const t0 = performance.now();
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  ort.env.wasm.wasmPaths = ORT_DIST + '/';
  const session = await ort.InferenceSession.create(MODEL, { executionProviders: ['wasm'] });
  const startupMs = performance.now() - t0;
  const sr = new ort.Tensor('int64', [16000n]);
  const probs = [];
  const times = [];
  for (let pass = 0; pass < PASSES; pass++) {
    let state = new ort.Tensor('float32', new Float32Array(2 * 128), [2, 1, 128]);
    let context = new Float32Array(64);
    for (const f of frames) {
      const t = performance.now();
      // Mirrors vad-web's models/silero.js: 64 context samples + 512 new ones.
      const withContext = new Float32Array(64 + FRAME);
      withContext.set(context, 0);
      withContext.set(f, 64);
      context = f.slice(-64);
      const out = await session.run({ input: new ort.Tensor('float32', withContext, [1, withContext.length]), state, sr });
      state = out.stateN;
      if (pass > 0) times.push(performance.now() - t);
      else probs.push(out.output.data[0]);
    }
  }
  return { probs, startupMs, ...stats(times) };
}

const auricle = await benchAuricle();
const ortRes = await benchOrt();
let maxDiff = 0;
for (let i = 0; i < frames.length; i++) maxDiff = Math.max(maxDiff, Math.abs(auricle.probs[i] - ortRes.probs[i]));

console.log(`${frames.length} frames of 32 ms, ${PASSES - 1} timed passes after a warm-up pass, Node ${process.version} (V8 ${process.versions.v8}), ${process.arch}`);
console.log(`per frame   Auricle brain WASM (Rust, SIMD128):            ${fmt(auricle)}`);
console.log(`per frame   onnxruntime-web ${ort.env.versions?.web ?? '1.30'} wasm (SIMD, 1 thread): ${fmt(ortRes)}`);
console.log(`startup     Auricle ${auricle.startupMs.toFixed(0)} ms (instantiate + weights)   onnxruntime-web ${ortRes.startupMs.toFixed(0)} ms (runtime + session)`);
console.log(`parity      max |p_auricle - p_ort| over the clip: ${maxDiff.toExponential(2)}`);
console.log('\npayload (gzipped)');
const ours = gz(join(ROOT, 'web/src/generated/brain/auricle_brain_bg.wasm')) + gz(join(ROOT, 'target/wasm32-unknown-unknown/release/auricle_worklet.wasm.opt'));
console.log(`  Auricle brain + worklet WASM (weights inside):          ${kb(ours)}`);
const theirs = [
  ['silero_vad_v6.onnx', join(VAD_WEB, 'silero_vad_v6.onnx')],
  ['ort-wasm-simd-threaded.wasm', join(ORT_DIST, 'ort-wasm-simd-threaded.wasm')],
  ['ort.wasm.min.mjs', join(ORT_DIST, 'ort.wasm.min.mjs')],
  ['vad-web bundle.min.js', join(VAD_WEB, 'bundle.min.js')],
  ['vad.worklet.bundle.min.js', join(VAD_WEB, 'vad.worklet.bundle.min.js')],
];
let total = 0;
for (const [name, file] of theirs) {
  const g = gz(file);
  total += g;
  console.log(`  ${name.padEnd(55)}${kb(g).padStart(8)}  (${kb(statSync(file).size)} raw)`);
}
console.log(`  vad-web total:                                         ${kb(total)}`);
process.exit(0);
