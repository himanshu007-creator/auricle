// Browser matrix with Playwright: Chromium, WebKit and Firefox.
//   1. bench/browser/index.html: per-frame cost of the Auricle brain WASM and of
//      onnxruntime-web in each browser, output parity, and whether the Rust
//      C-ABI worklets register and render.
//   2. The demo itself with a fake microphone where the browser offers one
//      (Chromium: the bundled clip as mic input; Firefox: its synthetic tone).
//
//   ../scripts/build-wasm.sh && (cd ../demo && npx vite --port 5174 &) && node browser-matrix.mjs
//   The parity row compares against onnxruntime running Silero v6, so it is only meaningful when the
//   brain was built with the Silero weights: MODELS="silero" ../scripts/build-wasm.sh first.
import { createServer } from 'node:http';
import { readFileSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, extname } from 'node:path';
import { chromium, webkit, firefox } from 'playwright';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');
const PORT = 4321;
const DEMO = process.env.DEMO_URL ?? 'http://localhost:5174/';
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.wasm': 'application/wasm', '.opt': 'application/wasm', '.wav': 'audio/wav', '.onnx': 'application/octet-stream' };

const server = createServer((req, res) => {
  const path = join(ROOT, decodeURIComponent(new URL(req.url, 'http://x').pathname));
  try {
    if (!statSync(path).isFile()) throw new Error("not a file");
    res.writeHead(200, { 'content-type': TYPES[extname(path)] ?? 'application/octet-stream' });
    res.end(readFileSync(path));
  } catch {
    res.writeHead(404);
    res.end();
  }
}).listen(PORT);

const fmt = (r) => `per frame p50 ${r.perFrameP50.toFixed(3)} ms (min ${r.perFrameMin.toFixed(3)}, max ${r.perFrameMax.toFixed(3)}), startup ${r.startupMs.toFixed(0)} ms`;

async function harness(name, browserType, launchOptions = {}) {
  const browser = await browserType.launch(launchOptions);
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.goto(`http://localhost:${PORT}/bench/browser/index.html`);
  await page.click('#run');
  await page.waitForFunction(() => window.__result, null, { timeout: 180000 });
  const r = await page.evaluate(() => window.__result);
  const version = browser.version();
  await browser.close();
  console.log(`\n== ${name} ${version}`);
  if (r.auricle) console.log(`  auricle brain WASM     ${fmt(r.auricle)}`);
  if (r.ort) console.log(`  onnxruntime-web WASM  ${fmt(r.ort)}`);
  if (r.parity !== undefined) console.log(`  parity max |diff|     ${r.parity.toExponential(2)}`);
  if (r.benchError) console.log(`  bench error: ${r.benchError}`);
  if (r.worklets) console.log(`  Rust worklets         registered, context ${r.worklets.contextRate} Hz, playback reported playing=${JSON.stringify(r.worklets.playingMessages)}`);
  if (r.workletError) console.log(`  worklet error: ${r.workletError}`);
  if (errors.length) console.log(`  page errors: ${errors.join(' | ')}`);
  return r;
}

async function demoAvailable() {
  try {
    return (await fetch(DEMO)).ok;
  } catch {
    return false;
  }
}

/** Runs the demo in VAD-only mode with a fake mic and reports what the pipeline saw. */
async function demo(name, browserType, launchOptions, contextOptions, seconds) {
  const browser = await browserType.launch(launchOptions);
  const context = await browser.newContext(contextOptions);
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.goto(DEMO);
  await page.click('#start');
  await page.waitForTimeout(seconds * 1000);
  const r = await page.evaluate(() => ({
    state: document.getElementById('state').textContent,
    status: document.getElementById('status').textContent,
    log: document.getElementById('log').textContent,
    clips: document.querySelectorAll('#chat audio').length,
  }));
  await browser.close();
  const count = (name) => (r.log.match(new RegExp(` ${name}$`, 'gm')) ?? []).length;
  console.log(`\n== demo in ${name}: state "${r.state}", ${r.status}`);
  console.log(`  events in ${seconds} s: candidate ${count('speech_candidate')}, confirmed ${count('speech_confirmed')}, speech_end ${count('speech_end')}, misfire ${count('misfire')}; recorded clips shown: ${r.clips}`);
  if (errors.length) console.log(`  page errors: ${errors.join(' | ')}`);
}

const attempt = async (label, fn) => {
  try {
    await fn();
  } catch (err) {
    console.log(`\n== ${label}: FAILED ${String(err).split('\n')[0]}`);
  }
};

await attempt('Chromium', () => harness('Chromium', chromium));
await attempt('WebKit', () => harness('WebKit', webkit));
await attempt('Firefox', () => harness('Firefox', firefox));

if (await demoAvailable()) {
  const clip = join(ROOT, 'demo/public/pretend-agent.wav');
  await attempt('demo in Chromium', () =>
    demo(
      'Chromium (fake mic playing the 19 s clip)',
      chromium,
      { args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', `--use-file-for-fake-audio-capture=${clip}%noloop`, '--autoplay-policy=no-user-gesture-required'] },
      { permissions: ['microphone'] },
      24,
    ),
  );
  await attempt('demo in Firefox', () => demo('Firefox (fake mic, synthetic tone)', firefox, { firefoxUserPrefs: { 'media.navigator.streams.fake': true, 'media.navigator.permission.disabled': true } }, {}, 8));
  await attempt('demo in WebKit', () => demo('WebKit (no fake mic available)', webkit, {}, {}, 4));
} else {
  console.log(`\ndemo not running at ${DEMO}; skipped the fake-mic runs`);
}
server.close();
