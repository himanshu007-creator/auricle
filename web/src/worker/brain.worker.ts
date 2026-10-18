// Brain worker: runs the VAD and turn machine on every 16 kHz frame from the
// capture worklet, drives the playback worklet directly over a MessagePort
// (duck / unduck / stop), assembles speech segments with pre-roll, and runs
// the text cleanup rules.
import init, { Brain, clean_text, keep_mask, ends_open, verify_subsequence } from '../generated/brain/auricle_brain.js';
import { EVENT_NAMES, type BrainToMain, type MainToBrain, type TurnEventName } from '../protocol.ts';

const FRAME_LEN = 512;
const FRAME_MS = 32;
const PRE_ROLL_FRAMES = 16; // 512 ms

let brain: Brain;
let playbackPort: MessagePort | null = null;
let frameIndex = 0;
const preRoll: Float32Array[] = [];
let segment: Float32Array[] | null = null;
let segmentStartMs = 0;
let segmentId = 0;

const post = (msg: BrainToMain, transfer: Transferable[] = []) => self.postMessage(msg, transfer);

function onFrame(frame: Float32Array) {
  const p = brain.process(frame);
  const tMs = frameIndex * FRAME_MS;
  frameIndex++;
  post({ type: 'frame', p, tMs });

  if (segment) segment.push(frame);
  preRoll.push(frame);
  if (preRoll.length > PRE_ROLL_FRAMES) preRoll.shift();

  const events = brain.take_events();
  for (let i = 0; i < events.length; i += 2) {
    const code = events[i];
    const atMs = events[i + 1];
    const name = EVENT_NAMES[code] as TurnEventName | '';
    if (!name) continue;
    switch (name) {
      case 'speech_candidate':
        if (!segment) {
          segment = preRoll.slice();
          segmentStartMs = Math.max(0, tMs - (preRoll.length - 1) * FRAME_MS);
          segmentId++;
        }
        break;
      case 'silence_candidate':
        if (segment) postSegment(true, tMs);
        break;
      case 'duck':
      case 'unduck':
        playbackPort?.postMessage({ type: name });
        break;
      case 'interrupt':
        playbackPort?.postMessage({ type: 'stop' });
        break;
      case 'misfire':
        segment = null;
        break;
      case 'speech_end':
        if (segment) {
          postSegment(false, tMs);
          segment = null;
        }
        break;
    }
    post({ type: 'event', name, tMs: atMs });
  }
}

function postSegment(provisional: boolean, endMs: number) {
  const samples = new Float32Array(segment!.length * FRAME_LEN);
  segment!.forEach((f, i) => samples.set(f, i * FRAME_LEN));
  post({ type: 'segment', id: segmentId, provisional, samples, startMs: segmentStartMs, endMs }, [samples.buffer]);
}

self.onmessage = async (e: MessageEvent<MainToBrain>) => {
  const msg = e.data;
  switch (msg.type) {
    case 'init':
      await init({ module_or_path: msg.wasm });
      brain = Brain.with_model(msg.model);
      post({ type: 'ready' });
      break;
    case 'ports':
      msg.capture.onmessage = (ev: MessageEvent<Float32Array>) => onFrame(ev.data);
      playbackPort = msg.playback;
      playbackPort.onmessage = (ev) => {
        if (ev.data.type === 'playing') brain.set_agent_playing(ev.data.value);
      };
      break;
    case 'profile':
      brain.set_profile(msg.value === 'headphones' ? 1 : 0);
      break;
    case 'thresholds':
      brain.set_thresholds(msg.which, msg.positive, msg.negative, msg.hangoverMs, msg.minSpeechMs);
      break;
    case 'agentPlaying':
      brain.set_agent_playing(msg.value);
      break;
    case 'clean':
      post({
        type: 'clean',
        id: msg.id,
        clean: clean_text(msg.text),
        mask: Array.from(keep_mask(msg.text), (b) => b === 1),
        endsOpen: ends_open(msg.text),
      });
      break;
    case 'verify':
      post({ type: 'verify', id: msg.id, ok: verify_subsequence(msg.verbatim, msg.candidate) });
      break;
    case 'reset':
      brain.reset();
      frameIndex = 0;
      preRoll.length = 0;
      segment = null;
      break;
  }
};
