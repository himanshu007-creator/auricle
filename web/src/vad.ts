// Level 1: just the detector, as plain callbacks. Everything else in Auricle
// (turn policy, ducking, backends) stays available through `.auricle`.
import { Auricle, type AuricleOptions } from './auricle.ts';
import type { Profile } from './protocol.ts';
import type { SessionStats } from './stats.ts';
import { encodeWav } from './wav.ts';

/** A finished stretch of the user's speech. */
export interface Segment {
  /** 16 kHz mono samples in -1..1, including 512 ms of audio from before the first voiced frame. */
  audio: Float32Array;
  /** Milliseconds since `start()` at which the segment began (pre-roll included). */
  startMs: number;
  /** Milliseconds since `start()` at which the hangover ended. */
  endMs: number;
  /** `endMs - startMs`. */
  durationMs: number;
  /** The segment as a 16-bit PCM WAV file, ready for `fetch` or `<audio>`. */
  wav(): Blob;
}

/** Thresholds for the turn policy while the agent is silent. */
export interface Thresholds {
  /** Probability a frame needs to count as voiced (default 0.5). */
  positive?: number;
  /** Below this a frame counts as silent; the gap to `positive` is hysteresis (default 0.35). */
  negative?: number;
  /** Silence after the last voiced frame before the segment ends (default 240; vad-web's default is 1,400). */
  hangoverMs?: number;
  /** Segments shorter than this are dropped as misfires (default 100). Raise it for keyboards and clicks. */
  minSpeechMs?: number;
}

export interface VadOptions {
  /** Fires on the speech candidate: three voiced frames, about 100 ms in. Reversible actions go here (show a meter, duck music). */
  onSpeechStart?: (at: { tMs: number }) => void;
  /** Fires on confirmation: six of the last ten frames voiced, 200 to 320 ms in. Irreversible actions go here (stop the agent). */
  onSpeechConfirmed?: (at: { tMs: number }) => void;
  /** The finished segment at 16 kHz with 512 ms of pre-roll, in the vad-web shape. */
  onSpeechEnd?: (audio: Float32Array, span: { startMs: number; endMs: number }) => void;
  /** The finished segment with its timing and a `wav()` helper; same moment as `onSpeechEnd`. */
  onSegment?: (segment: Segment) => void;
  /** A candidate that never confirmed. Undo whatever `onSpeechStart` did. */
  onMisfire?: (at: { tMs: number }) => void;
  /** Every 32 ms: the speech probability, 0 to 1, and the time since `start()`. */
  onFrame?: (p: number, tMs: number) => void;
  /** Anything that fails after `start()` resolved, for example the microphone being revoked. */
  onError?: (error: Error) => void;
  /** 'speakers' (default) guards against the agent's own echo; 'headphones' uses lower thresholds while the agent talks. */
  profile?: Profile;
  /** Overrides for the thresholds while the agent is silent. */
  thresholds?: Thresholds;
}

/** The detector. Create one with `createVad`, start it from a user gesture. */
export interface Vad {
  /** Asks for the microphone and starts scoring. Must be called from a user gesture (a click). */
  start(): Promise<void>;
  /** Stops scoring and releases the microphone. `start()` again for a new session. */
  stop(): void;
  /** Keeps the microphone open but ignores its audio until `resume()`. The current segment, if any, is dropped. */
  pause(): void;
  /** Resumes after `pause()`. */
  resume(): void;
  /** Same as `stop()`; the vad-web name. */
  destroy(): void;
  /** Changes the thresholds while running. */
  setThresholds(thresholds: Thresholds): void;
  /** True between `start()` and `stop()`, including while paused. */
  readonly listening: boolean;
  /** True while paused. */
  readonly paused: boolean;
  /** True from the speech candidate until the segment ends or misfires. */
  readonly speaking: boolean;
  /** The latest speech probability, 0 to 1. */
  readonly probability: number;
  /** Measured counters for this session: audio sent to STT against session length, segments, misfires. */
  readonly stats: SessionStats;
  /** The underlying instance, for everything the detector-only surface leaves out. */
  readonly auricle: Auricle;
}

const DEFAULTS: Required<Thresholds> = { positive: 0.5, negative: 0.35, hangoverMs: 240, minSpeechMs: 100 };

/**
 * Creates the detector. Downloads nothing: the model is inside the package.
 *
 * ```ts
 * const vad = await createVad({ onSegment: (s) => upload(s.wav()) });
 * button.onclick = () => vad.start();
 * ```
 */
export async function createVad(options: VadOptions = {}): Promise<Vad> {
  const auricleOptions: AuricleOptions = { mode: 'vad', profile: options.profile };
  const a = await Auricle.create(auricleOptions);
  let probability = 0;
  let speaking = false;
  let paused = false;
  let thresholds: Required<Thresholds> = { ...DEFAULTS, ...options.thresholds };
  const apply = () => a.setThresholds(0, thresholds.positive, thresholds.negative, thresholds.hangoverMs, thresholds.minSpeechMs);

  a.on('frame', ({ p, tMs }) => {
    if (paused) return;
    probability = p;
    options.onFrame?.(p, tMs);
  });
  a.on('event', ({ name, tMs }) => {
    if (paused) return;
    if (name === 'speech_candidate') {
      speaking = true;
      options.onSpeechStart?.({ tMs });
    } else if (name === 'speech_confirmed') options.onSpeechConfirmed?.({ tMs });
    else if (name === 'misfire') {
      speaking = false;
      options.onMisfire?.({ tMs });
    } else if (name === 'speech_end') speaking = false;
  });
  a.on('speechEnd', ({ samples, startMs, endMs }) => {
    if (paused) return;
    options.onSpeechEnd?.(samples, { startMs, endMs });
    options.onSegment?.({ audio: samples, startMs, endMs, durationMs: endMs - startMs, wav: () => encodeWav(samples, 16000) });
  });
  a.on('error', (err) => options.onError?.(err));

  const stop = () => {
    a.stop();
    speaking = false;
    paused = false;
    probability = 0;
  };

  return {
    async start() {
      await a.start();
      paused = false;
      apply();
    },
    stop,
    pause() {
      paused = true;
      speaking = false;
      // A threshold above 1 can never be reached, so the turn machine goes quiet without tearing down the graph.
      a.setThresholds(0, 1.01, 1.0, thresholds.hangoverMs, thresholds.minSpeechMs);
    },
    resume() {
      paused = false;
      apply();
    },
    destroy: stop,
    setThresholds(t) {
      thresholds = { ...thresholds, ...t };
      if (!paused) apply();
    },
    get listening() {
      return a.state !== 'idle';
    },
    get paused() {
      return paused;
    },
    get speaking() {
      return speaking;
    },
    get probability() {
      return probability;
    },
    get stats() {
      return a.stats;
    },
    auricle: a,
  };
}
