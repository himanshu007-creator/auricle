// Endpointing accuracy on bench/data. The same Silero v6 probabilities feed
// Auricle's turn machine and vad-web's FrameProcessor (its shipped defaults);
// each is scored against the known speech intervals.
//
//   python3 bench/make_dataset.py bench/data && node bench/accuracy.mjs
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');
const DATA = join(HERE, 'data');
const require = createRequire(import.meta.url);
const { FrameProcessor, defaultFrameProcessorOptions } = require('@ricky0123/vad-web/dist/frame-processor.js');
const { Message } = require('@ricky0123/vad-web/dist/messages.js');
const FRAME = 512;
const FRAME_MS = 32;

const brainMod = await import(join(ROOT, 'web/src/generated/brain/auricle_brain.js'));
await brainMod.default({ module_or_path: readFileSync(join(ROOT, 'web/src/generated/brain/auricle_brain_bg.wasm')) });
const brain = new brainMod.Brain();
const MIN_SPEECH = Number(process.env.MIN_SPEECH_MS ?? 100);
brain.set_thresholds(0, 0.5, 0.35, 240, MIN_SPEECH);
const EVENT = ['', 'speech_candidate', 'speech_confirmed', 'silence_candidate', 'speech_end', 'misfire', 'duck', 'unduck', 'interrupt'];

function readWav(file) {
  const b = readFileSync(file);
  const pcm = new Int16Array(b.buffer, b.byteOffset + 44, (b.length - 44) / 2);
  return Float32Array.from(pcm, (v) => v / 32768);
}

/** Runs Auricle: returns probabilities and detected segments {start, end} in ms. */
function runAuricle(samples) {
  brain.reset();
  const probs = [];
  const segments = [];
  let open = null;
  for (let i = 0; i + FRAME <= samples.length; i += FRAME) {
    probs.push(brain.process(samples.subarray(i, i + FRAME)));
    const ev = brain.take_events();
    for (let k = 0; k < ev.length; k += 2) {
      const name = EVENT[ev[k]];
      const t = ev[k + 1];
      if (name === 'speech_candidate') open = { start: t, confirmed: false };
      else if (name === 'speech_confirmed' && open) open.confirmed = true;
      else if (name === 'speech_end' && open) {
        segments.push({ start: open.start, end: t });
        open = null;
      } else if (name === 'misfire') open = null;
    }
  }
  return { probs, segments };
}

/** Runs vad-web's FrameProcessor on precomputed probabilities (its model call is replaced by a lookup). */
async function runVadWeb(probs) {
  let idx = 0;
  const fp = new FrameProcessor(async () => ({ isSpeech: probs[idx], notSpeech: 1 - probs[idx] }), () => {}, { ...defaultFrameProcessorOptions }, FRAME_MS);
  fp.resume();
  const segments = [];
  let open = null;
  const dummy = new Float32Array(FRAME);
  for (idx = 0; idx < probs.length; idx++) {
    const t = idx * FRAME_MS;
    await fp.process(dummy, (e) => {
      if (e.msg === Message.SpeechStart) open = { start: t };
      else if (e.msg === Message.SpeechEnd && open) {
        segments.push({ start: open.start, end: t });
        open = null;
      } else if (e.msg === Message.VADMisfire) open = null;
    });
  }
  return { segments };
}

/** Scores detected segments against true intervals (seconds). */
function score(segments, truth) {
  const truthMs = truth.map(([s, e]) => ({ start: s * 1000, end: e * 1000 }));
  const overlap = (a, b) => Math.max(0, Math.min(a.end, b.end) - Math.max(a.start, b.start));
  // Match when the overlap covers half of the shorter of the two, so a split half of a long utterance counts as a split, not a false alarm.
  const matches = segments.map((seg) => truthMs.map((t, i) => (overlap(seg, t) >= 0.5 * Math.min(t.end - t.start, seg.end - seg.start) ? i : -1)).filter((i) => i >= 0));
  const detectedTrue = new Set(matches.flat());
  const perTrue = truthMs.map((_, i) => matches.filter((m) => m.includes(i)).length);
  // A detection that spans several utterances is timed against the first one
  // it starts on and the last one it ends on.
  const startErr = [];
  const endErr = [];
  segments.forEach((seg, k) => {
    if (matches[k].length === 0) return;
    startErr.push(seg.start - truthMs[Math.min(...matches[k])].start);
    endErr.push(seg.end - truthMs[Math.max(...matches[k])].end);
  });
  return {
    truth: truthMs.length,
    detected: segments.length,
    recall: detectedTrue.size / truthMs.length,
    falseAlarms: matches.filter((m) => m.length === 0).length,
    merges: matches.filter((m) => m.length > 1).length,
    mergedUtterances: matches.filter((m) => m.length > 1).reduce((a, m) => a + m.length, 0),
    splits: perTrue.filter((n) => n > 1).length,
    startErr,
    endErr,
  };
}

const median = (a) => (a.length ? [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)] : NaN);
const labels = JSON.parse(readFileSync(join(DATA, 'labels.json'), 'utf8'));
const totals = { auricle: {}, vadweb: {} };
const bySnr = {};
const add = (acc, s) => {
  acc.truth = (acc.truth ?? 0) + s.truth;
  acc.detected = (acc.detected ?? 0) + s.detected;
  acc.hit = (acc.hit ?? 0) + Math.round(s.recall * s.truth);
  acc.falseAlarms = (acc.falseAlarms ?? 0) + s.falseAlarms;
  acc.merges = (acc.merges ?? 0) + s.merges;
  acc.mergedUtterances = (acc.mergedUtterances ?? 0) + s.mergedUtterances;
  acc.splits = (acc.splits ?? 0) + s.splits;
  (acc.startErr ??= []).push(...s.startErr);
  (acc.endErr ??= []).push(...s.endErr);
};

for (const [file, info] of Object.entries(labels)) {
  const samples = readWav(join(DATA, file));
  const d = runAuricle(samples);
  const v = await runVadWeb(d.probs);
  const sd = score(d.segments, info.speech);
  const sv = score(v.segments, info.speech);
  const key = info.snr_db === null ? 'clean' : `${info.snr_db} dB`;
  bySnr[key] ??= { auricle: {}, vadweb: {} };
  add(bySnr[key].auricle, sd);
  add(bySnr[key].vadweb, sv);
  add(totals.auricle, sd);
  add(totals.vadweb, sv);
}

const row = (name, a) =>
  `${name.padEnd(9)} recall ${((100 * a.hit) / a.truth).toFixed(1).padStart(5)}%  false alarms ${String(a.falseAlarms).padStart(3)}  merged detections ${String(a.merges).padStart(3)} (${String(a.mergedUtterances).padStart(3)} utterances)  splits ${String(a.splits).padStart(3)}  start error median ${median(a.startErr).toFixed(0).padStart(5)} ms  end delay median ${median(a.endErr).toFixed(0).padStart(5)} ms`;
console.log(`${Object.keys(labels).length} files, ${totals.auricle.truth} labelled utterances. Same probabilities; only the endpointing policy differs.`);
console.log(`Auricle: candidate 3 frames, confirm 6/10, hangover 240 ms, min speech ${MIN_SPEECH} ms. vad-web defaults: thresholds 0.3/0.25, minSpeech 400 ms, redemption 1400 ms.\n`);
for (const [snr, r] of Object.entries(bySnr)) {
  console.log(`[${snr}]`);
  console.log('  ' + row('auricle', r.auricle));
  console.log('  ' + row('vad-web', r.vadweb));
}
console.log('[all]');
console.log('  ' + row('auricle', totals.auricle));
console.log('  ' + row('vad-web', totals.vadweb));
console.log('\nstart error: first start event minus true start (negative = fired early on pre-roll/noise). end delay: end event minus true end.');
process.exit(0);
