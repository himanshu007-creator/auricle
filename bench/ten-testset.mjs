// Frame-level accuracy of the Auricle (Silero v6) probabilities on TEN VAD's
// human-annotated test set (30 real recordings, 16 kHz, labels in .scv files
// as "start,end,label" triplets). Optionally scores a TEN VAD output directory
// produced by its own Python demo (one probability per 16 ms hop per file).
//
//   node bench/ten-testset.mjs /path/to/ten-vad/testset [/path/to/ten-vad-outputs]
import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');
const TESTSET = process.argv[2];
const TEN_OUT = process.argv[3];
if (!TESTSET) {
  console.error('usage: node bench/ten-testset.mjs <testset dir> [ten-vad output dir]');
  process.exit(2);
}
const FRAME = 512;
const FRAME_MS = 32;

const mod = await import(join(ROOT, 'web/src/generated/brain/auricle_brain.js'));
await mod.default({ module_or_path: readFileSync(join(ROOT, 'web/src/generated/brain/auricle_brain_bg.wasm')) });
const brain = new mod.Brain();

function readWav(file) {
  const b = readFileSync(file);
  const pcm = new Int16Array(b.buffer, b.byteOffset + 44, (b.length - 44) / 2);
  return Float32Array.from(pcm, (v) => v / 32768);
}

/** "name,s,e,label,s,e,label,..." -> per-millisecond-resolution label lookup. */
function readLabels(file) {
  const parts = readFileSync(file, 'utf8').trim().split(',');
  const spans = [];
  for (let i = 1; i + 2 < parts.length; i += 3) spans.push({ start: +parts[i], end: +parts[i + 1], speech: parts[i + 2] === '1' });
  return (tSec) => spans.find((s) => tSec >= s.start && tSec < s.end)?.speech ?? false;
}

/** Counts hits per threshold for one sequence of (prob, truth) pairs. */
function tally(acc, pairs, thresholds) {
  for (const th of thresholds) {
    const a = (acc[th] ??= { tp: 0, fp: 0, fn: 0, tn: 0 });
    for (const [p, truth] of pairs) {
      const pred = p >= th;
      if (pred && truth) a.tp++;
      else if (pred && !truth) a.fp++;
      else if (!pred && truth) a.fn++;
      else a.tn++;
    }
  }
}

const thresholds = [0.3, 0.5, 0.7];
const auricle = {};
const ten = {};
let files = 0;
let frames = 0;
for (const name of readdirSync(TESTSET).filter((f) => f.endsWith('.wav')).sort()) {
  const base = name.replace(/\.wav$/, '');
  const scv = join(TESTSET, `${base}.scv`);
  if (!existsSync(scv)) continue;
  const truthAt = readLabels(scv);
  const samples = readWav(join(TESTSET, name));
  brain.reset();
  const pairs = [];
  for (let i = 0; i + FRAME <= samples.length; i += FRAME) {
    const p = brain.process(samples.subarray(i, i + FRAME));
    brain.take_events();
    // The frame covers [i, i+512); label it by its centre.
    pairs.push([p, truthAt((i + FRAME / 2) / 16000)]);
  }
  tally(auricle, pairs, thresholds);
  frames += pairs.length;
  files++;

  if (TEN_OUT && existsSync(join(TEN_OUT, `${base}.txt`))) {
    // TEN demo output: one line per 16 ms hop, "[index] probability flag".
    const lines = readFileSync(join(TEN_OUT, `${base}.txt`), 'utf8').trim().split('\n');
    const tenPairs = lines.map((line, k) => {
      const m = line.match(/([0-9.]+)\s*,?\s*(\d)\s*$/);
      const p = m ? +m[1] : NaN;
      return [p, truthAt((k * 256 + 128) / 16000)];
    }).filter(([p]) => !Number.isNaN(p));
    tally(ten, tenPairs, thresholds);
  }
}

const report = (name, acc) => {
  console.log(name);
  for (const th of thresholds) {
    const a = acc[th];
    if (!a) continue;
    const precision = a.tp / (a.tp + a.fp || 1);
    const recall = a.tp / (a.tp + a.fn || 1);
    const f1 = (2 * precision * recall) / (precision + recall || 1);
    const accuracy = (a.tp + a.tn) / (a.tp + a.tn + a.fp + a.fn);
    console.log(`  threshold ${th}: precision ${(100 * precision).toFixed(1)}%  recall ${(100 * recall).toFixed(1)}%  F1 ${(100 * f1).toFixed(1)}%  frame accuracy ${(100 * accuracy).toFixed(1)}%`);
  }
};
console.log(`${files} annotated files, ${frames} frames of 32 ms`);
report('Auricle / Silero v6 (32 ms frames)', auricle);
if (Object.keys(ten).length) report('TEN VAD (16 ms hops, its own ONNX + C++ features)', ten);
process.exit(0);
