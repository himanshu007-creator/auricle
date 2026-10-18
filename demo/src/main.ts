// A small chat page on top of auricle. Left: key, start, live trace.
// Right: the conversation; an interrupted reply is cut to the words heard.
import { Auricle, OpenAICompat, encodeWav, formatStats, type CleanupMode, type AuricleState, type Profile, type Turn, type TurnEventName } from 'auricle';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const input = (id: string) => $<HTMLInputElement>(id);
const escapeHtml = (s: string) => s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]!);

// ---------- settings ----------
const STORAGE_KEY = 'auricle-demo-key';
try {
  const saved = sessionStorage.getItem(STORAGE_KEY);
  if (saved) {
    input('apiKey').value = saved;
    input('remember').checked = true;
  }
} catch {
  /* storage blocked */
}
const saveKey = () => {
  if (input('remember').checked) sessionStorage.setItem(STORAGE_KEY, input('apiKey').value);
  else sessionStorage.removeItem(STORAGE_KEY);
};
input('apiKey').addEventListener('change', saveKey);
input('remember').addEventListener('change', saveKey);

function makeBackend(): OpenAICompat | null {
  const apiKey = input('apiKey').value.trim();
  if (!apiKey) return null;
  let extraBody: Record<string, unknown> | undefined;
  try {
    extraBody = input('extraBody').value.trim() ? (JSON.parse(input('extraBody').value) as Record<string, unknown>) : undefined;
  } catch {
    setStatus('Extra chat body is not valid JSON; ignoring it.');
  }
  return new OpenAICompat({
    baseURL: input('baseURL').value.trim(),
    apiKey,
    models: { stt: input('sttModel').value.trim(), llm: input('llmModel').value.trim(), tts: input('ttsModel').value.trim() },
    voice: input('voice').value.trim() || undefined,
    extraBody,
  });
}

const setStatus = (s: string) => ($('status').textContent = s);
const log: string[] = [];
function addLog(line: string) {
  log.push(`${new Date().toISOString().slice(11, 23)} ${line}`);
  if (log.length > 500) log.shift();
  $('log').textContent = log.slice(-150).join('\n');
}

$('testConnection').onclick = async () => {
  const backend = makeBackend();
  const list = $('testResults');
  if (!backend) {
    list.innerHTML = '<li class="fail">Enter the API key first.</li>';
    return;
  }
  list.innerHTML = '<li>Testing three endpoints…</li>';
  const results = await backend.testConnection();
  list.innerHTML = '';
  for (const [role, r] of Object.entries(results)) {
    const li = document.createElement('li');
    li.className = r.ok ? 'ok' : 'fail';
    li.textContent = `${role}: ${r.ok ? 'ok, ' : 'failed: '}${r.message}`;
    list.appendChild(li);
  }
};

// ---------- chat ----------
const chat = $('chat');
let agentBubble: HTMLElement | null = null;

function addBubble(kind: 'user' | 'agent', html: string): HTMLElement {
  chat.querySelector('.empty')?.remove();
  const div = document.createElement('div');
  div.className = `msg ${kind}`;
  div.innerHTML = html;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

/** VAD-only mode: every detected segment becomes a playable clip, like the vad-web demo. */
function addRecording(seg: { samples: Float32Array; startMs: number; endMs: number }) {
  const url = URL.createObjectURL(encodeWav(seg.samples, 16000));
  const seconds = (seg.samples.length / 16000).toFixed(2);
  const bubble = addBubble('user', `<audio controls src="${url}"></audio><span class="meta">segment ${segments.length} · ${seconds} s incl. pre-roll · detected ${(seg.startMs / 1000).toFixed(1)}–${(seg.endMs / 1000).toFixed(1)} s · <a href="${url}" download="segment-${segments.length}.wav">download</a></span>`);
  bubble.classList.add('recording');
}

function userHtml(turn: Turn): string {
  const tokens = turn.verbatim.split(/\s+/).filter(Boolean);
  const removed = turn.mask.some((k) => k === false);
  const marked = tokens.map((t, i) => (turn.mask[i] === false ? `<del>${escapeHtml(t)}</del>` : escapeHtml(t))).join(' ');
  const timing = [turn.metrics.sttMs && `stt ${turn.metrics.sttMs} ms`].filter(Boolean).join(' · ');
  return `${removed ? marked : escapeHtml(turn.clean)}<span class="meta">you${timing ? ` · ${timing}` : ''}</span>`;
}

// ---------- trace ----------
const canvas = $<HTMLCanvasElement>('trace');
const ctx2d = canvas.getContext('2d')!;
const WINDOW_MS = 8000;
const frames: { p: number; tMs: number }[] = [];
const events: { name: TurnEventName; tMs: number }[] = [];
const agentSpans: { from: number; to: number | null }[] = [];
let lastFrameMs = 0;
const markColor: Partial<Record<TurnEventName, string>> = { speech_candidate: '#f4c95d', speech_confirmed: '#5ab0ff', interrupt: '#ff6b6b' };

function drawTrace() {
  const { width, height } = canvas;
  ctx2d.clearRect(0, 0, width, height);
  const t0 = Math.max(0, lastFrameMs - WINDOW_MS);
  const x = (tMs: number) => ((tMs - t0) / WINDOW_MS) * width;
  ctx2d.fillStyle = 'rgba(90,176,255,0.15)';
  for (const span of agentSpans) {
    const to = span.to ?? lastFrameMs;
    if (to >= t0) ctx2d.fillRect(x(span.from), 0, x(to) - x(span.from), height);
  }
  ctx2d.strokeStyle = '#5ad1a1';
  ctx2d.lineWidth = 2;
  ctx2d.beginPath();
  let started = false;
  for (const f of frames) {
    if (f.tMs < t0) continue;
    if (started) ctx2d.lineTo(x(f.tMs), height - f.p * height);
    else ctx2d.moveTo(x(f.tMs), height - f.p * height);
    started = true;
  }
  ctx2d.stroke();
  ctx2d.lineWidth = 1;
  for (const e of events) {
    const color = markColor[e.name];
    if (!color || e.tMs < t0) continue;
    ctx2d.strokeStyle = color;
    ctx2d.beginPath();
    ctx2d.moveTo(x(e.tMs), 0);
    ctx2d.lineTo(x(e.tMs), height);
    ctx2d.stroke();
  }
  requestAnimationFrame(drawTrace);
}
requestAnimationFrame(drawTrace);

// ---------- session ----------
let auricle: Auricle | null = null;
const segments: Float32Array[] = [];
const turns: Turn[] = [];

async function detectProfile(): Promise<Profile> {
  const selected = $<HTMLSelectElement>('profile').value;
  if (selected !== 'auto') return selected as Profile;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.some((d) => /head|airpod|buds|earphone|headset/i.test(d.label)) ? 'headphones' : 'speakers';
  } catch {
    return 'speakers';
  }
}

// Session readout: the library's own counters, one line.
function updateReadout() {
  if (auricle) $('readout').textContent = formatStats(auricle.stats);
}
setInterval(updateReadout, 1000);

function wire(d: Auricle) {
  d.on('state', (s: AuricleState) => {
    $('state').textContent = s.replace('_', ' ');
    $('dot').className = `dot ${s}`;
    addLog(`state ${s}`);
  });
  d.on('frame', (f) => {
    frames.push(f);
    lastFrameMs = f.tMs;
    if (frames.length > 20000) frames.splice(0, 5000);
  });
  d.on('event', (e) => {
    events.push(e);
    addLog(`${e.tMs} ms ${e.name}`);
  });
  d.on('agentPlaying', (playing) => {
    if (playing) agentSpans.push({ from: lastFrameMs, to: null });
    else if (agentSpans.length) agentSpans[agentSpans.length - 1].to = lastFrameMs;
  });
  d.on('speechEnd', (seg) => {
    segments.push(seg.samples);
    if (d.options.mode === 'vad') addRecording(seg);
  });
  d.on('turn', (turn) => {
    turns.push(turn);
    addBubble('user', userHtml(turn));
    agentBubble = addBubble('agent', '<span class="text">…</span><span class="meta">agent</span>');
  });
  d.on('agentText', ({ text }) => {
    if (agentBubble) agentBubble.querySelector('.text')!.textContent = text;
    chat.scrollTop = chat.scrollHeight;
  });
  d.on('metrics', (m) => {
    const meta = agentBubble?.querySelector('.meta');
    if (meta) meta.textContent = `agent · you stopped → first audio ${m.endToFirstAudioMs} ms (stt wait ${m.sttWaitMs}, first token ${m.firstTokenMs}, first audio ${m.firstAudioMs} after stt)`;
  });
  d.on('agentDone', () => {
    agentBubble = null;
  });
  d.on('interrupted', ({ heardText, generatedText, stopLatencyMs }) => {
    if (agentBubble) {
      const unheard = generatedText.startsWith(heardText) ? generatedText.slice(heardText.length) : '';
      agentBubble.innerHTML =
        `<span class="text">${escapeHtml(heardText) || '<em>(nothing heard yet)</em>'}</span>` +
        (unheard.trim() ? ` <span class="unheard">${escapeHtml(unheard.trim())}</span>` : '') +
        `<span class="meta">agent · <span class="cut">interrupted</span>, stopped in ${Math.round(stopLatencyMs)} ms; only the plain text stays in history</span>`;
      agentBubble = null;
    }
    addLog(`interrupted after ${Math.round(stopLatencyMs)} ms; heard "${heardText}"`);
  });
  d.on('error', (err) => {
    setStatus(`error: ${err.message}`);
    addLog(`error ${err.message}`);
  });
}

$('start').onclick = async () => {
  const backend = makeBackend();
  const profile = await detectProfile();
  try {
    auricle = await Auricle.create({
      backend: backend ?? undefined,
      mode: backend ? 'conversation' : 'vad',
      profile,
      loopback: $<HTMLSelectElement>('loopback').value as 'auto' | 'on' | 'off',
      cleanup: $<HTMLSelectElement>('cleanup').value as CleanupMode,
    });
    wire(auricle);
    await auricle.start();
  } catch (err) {
    setStatus(`could not start: ${(err as Error).message}`);
    return;
  }
  for (const id of ['stopBtn', 'interruptBtn', 'playClip']) $(id).toggleAttribute('disabled', false);
  $('start').toggleAttribute('disabled', true);
  setStatus(`${backend ? '' : 'VAD only (no key). '}${profile}, ${auricle.sampleRate} Hz, loopback ${auricle.loopbackActive ? 'on' : 'off'}`);
};

$('stopBtn').onclick = () => {
  auricle?.stop();
  auricle = null;
  for (const id of ['stopBtn', 'interruptBtn', 'playClip']) $(id).toggleAttribute('disabled', true);
  $('start').toggleAttribute('disabled', false);
};
$('interruptBtn').onclick = () => auricle?.interrupt();
$<HTMLSelectElement>('cleanup').onchange = (e) => auricle?.setCleanup((e.target as HTMLSelectElement).value as CleanupMode);

let clipSamples: Float32Array | null = null;
$('playClip').onclick = async () => {
  if (!clipSamples) {
    const bytes = new Uint8Array(await (await fetch(`${import.meta.env.BASE_URL}pretend-agent.wav`)).arrayBuffer());
    const view = new DataView(bytes.buffer, 44);
    clipSamples = new Float32Array((bytes.length - 44) / 2);
    for (let i = 0; i < clipSamples.length; i++) clipSamples[i] = view.getInt16(i * 2, true) / 32768;
  }
  auricle?.playClip(clipSamples, 16000);
};

$('export').onclick = async () => {
  const audio = await Promise.all(
    segments.map(async (s) => {
      const bytes = new Uint8Array(await encodeWav(s, 16000).arrayBuffer());
      let bin = '';
      bytes.forEach((b) => (bin += String.fromCharCode(b)));
      return btoa(bin);
    }),
  );
  const bundle = { exportedAt: new Date().toISOString(), userAgent: navigator.userAgent, frames, events, turns, messages: auricle?.messages ?? [], segmentsWavBase64: audio, log };
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(bundle)], { type: 'application/json' }));
  a.download = `auricle-session-${Date.now()}.json`;
  a.click();
};
