// Post-processing experiment on TEN VAD's annotated set: does smoothing the
// per-frame probability raise frame-level F1 without adding latency?
// Variants are causal only (no lookahead), so each could run in the worker.
// Result (2026-10-17): none beats the raw output; see docs/COMPARISON.md.
//
//   node bench/smoothing.mjs /path/to/ten-vad/testset
import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');
const TESTSET = process.argv[2];

const mod = await import(join(ROOT, 'web/src/generated/brain/auricle_brain.js'));
await mod.default({ module_or_path: readFileSync(join(ROOT, 'web/src/generated/brain/auricle_brain_bg.wasm')) });
const brain = new mod.Brain();

const readWav = (file) => {
  const b = readFileSync(file);
  return Float32Array.from(new Int16Array(b.buffer, b.byteOffset + 44, (b.length - 44) / 2), (v) => v / 32768);
};
const readLabels = (file) => {
  const parts = readFileSync(file, 'utf8').trim().split(',');
  const spans = [];
  for (let i = 1; i + 2 < parts.length; i += 3) spans.push({ start: +parts[i], end: +parts[i + 1], speech: parts[i + 2] === '1' });
  return (t) => spans.find((s) => t >= s.start && t < s.end)?.speech ?? false;
};

// Causal smoothers: each maps the raw sequence to a smoothed one.
const smoothers = {
  raw: (p) => p,
  ema_0_5: (p) => ema(p, 0.5),
  ema_0_7: (p) => ema(p, 0.7),
  median3: (p) => p.map((_, i) => median([p[i], p[i - 1] ?? p[i], p[i - 2] ?? p[i]])),
  max2: (p) => p.map((v, i) => Math.max(v, p[i - 1] ?? v)),
  // Two-state hysteresis as a frame labeller: on above 0.5, off below 0.35.
  hysteresis: (p) => {
    let on = false;
    return p.map((v) => {
      on = on ? v >= 0.35 : v >= 0.5;
      return on ? 1 : 0;
    });
  },
};
function ema(p, alpha) {
  let s = p[0];
  return p.map((v) => (s = alpha * v + (1 - alpha) * s));
}
const median = (a) => [...a].sort((x, y) => x - y)[1];

const acc = {};
for (const name of readdirSync(TESTSET).filter((f) => f.endsWith('.wav')).sort()) {
  const scv = join(TESTSET, name.replace(/\.wav$/, '.scv'));
  if (!existsSync(scv)) continue;
  const truthAt = readLabels(scv);
  const samples = readWav(join(TESTSET, name));
  brain.reset();
  const probs = [];
  const truth = [];
  for (let i = 0; i + 512 <= samples.length; i += 512) {
    probs.push(brain.process(samples.subarray(i, i + 512)));
    brain.take_events();
    truth.push(truthAt((i + 256) / 16000));
  }
  for (const [label, fn] of Object.entries(smoothers)) {
    const s = fn(probs);
    const a = (acc[label] ??= { tp: 0, fp: 0, fn: 0, tn: 0 });
    for (let i = 0; i < s.length; i++) {
      const pred = label === 'hysteresis' ? s[i] === 1 : s[i] >= 0.5;
      if (pred && truth[i]) a.tp++;
      else if (pred) a.fp++;
      else if (truth[i]) a.fn++;
      else a.tn++;
    }
  }
}
console.log('32 ms frames, threshold 0.5 after smoothing');
for (const [label, a] of Object.entries(acc)) {
  const precision = a.tp / (a.tp + a.fp);
  const recall = a.tp / (a.tp + a.fn);
  const f1 = (2 * precision * recall) / (precision + recall);
  console.log(`  ${label.padEnd(12)} precision ${(100 * precision).toFixed(1)}%  recall ${(100 * recall).toFixed(1)}%  F1 ${(100 * f1).toFixed(2)}%`);
}
process.exit(0);
