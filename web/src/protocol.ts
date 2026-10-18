// Messages between the main thread and the brain worker.

export const EVENT_NAMES = [
  '',
  'speech_candidate',
  'speech_confirmed',
  'silence_candidate',
  'speech_end',
  'misfire',
  'duck',
  'unduck',
  'interrupt',
] as const;

export type TurnEventName = Exclude<(typeof EVENT_NAMES)[number], ''>;

export type Profile = 'speakers' | 'headphones';

/** 0 = agent silent, 1 = agent speaking on speakers, 2 = agent speaking on headphones. */
export type ThresholdSlot = 0 | 1 | 2;

export type MainToBrain =
  /** The brain WASM bytes are decoded once on the main thread so the inlined worker stays small. */
  | { type: 'init'; wasm: ArrayBuffer; model: 0 | 1 }
  | { type: 'ports'; capture: MessagePort; playback: MessagePort }
  | { type: 'profile'; value: Profile }
  | { type: 'thresholds'; which: ThresholdSlot; positive: number; negative: number; hangoverMs: number; minSpeechMs: number }
  | { type: 'agentPlaying'; value: boolean }
  | { type: 'clean'; id: number; text: string }
  | { type: 'verify'; id: number; verbatim: string; candidate: string }
  | { type: 'reset' };

export type BrainToMain =
  | { type: 'ready' }
  | { type: 'frame'; p: number; tMs: number }
  | { type: 'event'; name: TurnEventName; tMs: number }
  /** A provisional segment is posted at each silence candidate so STT can start before the hangover ends; the final one shares its id. */
  | { type: 'segment'; id: number; provisional: boolean; samples: Float32Array; startMs: number; endMs: number }
  | { type: 'clean'; id: number; clean: string; mask: boolean[]; endsOpen: boolean }
  | { type: 'verify'; id: number; ok: boolean };
