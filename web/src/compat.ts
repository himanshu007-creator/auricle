// The @ricky0123/vad-web shaped API over Auricle, so an existing page switches
// by changing one import:
//
//   import { MicVAD, utils } from 'auricle/compat';
//   const vad = await MicVAD.new({ onSpeechEnd: (audio) => ... });
//   vad.start();
//
// What maps one to one: onSpeechStart (our candidate), onSpeechRealStart (our
// confirmation), onSpeechEnd with 16 kHz Float32Array audio, onVADMisfire,
// onFrameProcessed with {isSpeech, notSpeech}, positiveSpeechThreshold,
// negativeSpeechThreshold, redemptionMs (our hangover), minSpeechMs,
// startOnLoad, start(), pause(), destroy(), listening, utils.encodeWAV.
// What differs: pre-roll is fixed at 512 ms (preSpeechPadMs is accepted and
// ignored above that), onFrameProcessed receives no raw frame (the audio stays
// in the worklet), and there are no asset paths because there are no assets.
import { Auricle } from './auricle.ts';

/** The `@ricky0123/vad-web` option names, so a `MicVAD.new({...})` call moves over unchanged. */
export interface RealTimeVADOptions {
  /** Speech candidate, about 100 ms in (vad-web fires on the first frame above threshold). */
  onSpeechStart?: () => void;
  /** Speech confirmed, 200 to 320 ms in (vad-web waits `minSpeechMs`, 400 by default). */
  onSpeechRealStart?: () => void;
  /** The finished segment at 16 kHz with 512 ms of pre-roll. */
  onSpeechEnd?: (audio: Float32Array) => void;
  /** A candidate that never confirmed. */
  onVADMisfire?: () => void;
  /** Every 32 ms. The frame argument is empty: the samples stay inside the worklet. */
  onFrameProcessed?: (probabilities: { isSpeech: number; notSpeech: number }, frame: Float32Array) => void;
  /** Default 0.5. */
  positiveSpeechThreshold?: number;
  /** Default 0.35. */
  negativeSpeechThreshold?: number;
  /** Silence after speech before the segment ends; vad-web calls this redemption. Default 240, vad-web's 1,400. */
  redemptionMs?: number;
  /** Default 100; vad-web's 400. */
  minSpeechMs?: number;
  /** Accepted for compatibility; the pre-roll is a fixed 512 ms. */
  preSpeechPadMs?: number;
  /** Call `start()` inside `new()`. */
  startOnLoad?: boolean;
}

const DEFAULTS = { positiveSpeechThreshold: 0.5, negativeSpeechThreshold: 0.35, redemptionMs: 240, minSpeechMs: 100 };

/** The `@ricky0123/vad-web` class shape over the Auricle detector: `MicVAD.new`, `start`, `pause`, `destroy`. */
export class MicVAD {
  /** True between `start()` and `pause()`. */
  listening = false;

  private constructor(private readonly auricle: Auricle, private readonly options: RealTimeVADOptions) {}

  /** Builds the detector. Nothing is downloaded; there are no asset paths to set. */
  static async new(options: RealTimeVADOptions = {}): Promise<MicVAD> {
    const a = await Auricle.create({ mode: 'vad' });
    const vad = new MicVAD(a, options);
    a.on('frame', ({ p }) => options.onFrameProcessed?.({ isSpeech: p, notSpeech: 1 - p }, new Float32Array(0)));
    a.on('event', ({ name }) => {
      if (name === 'speech_candidate') options.onSpeechStart?.();
      else if (name === 'speech_confirmed') options.onSpeechRealStart?.();
      else if (name === 'misfire') options.onVADMisfire?.();
    });
    a.on('speechEnd', ({ samples }) => options.onSpeechEnd?.(samples));
    if (options.startOnLoad) await vad.start();
    return vad;
  }

  /** Asks for the microphone and starts scoring. Call from a user gesture. */
  async start(): Promise<void> {
    if (this.listening) return;
    await this.auricle.start();
    const o = { ...DEFAULTS, ...this.options };
    this.auricle.setThresholds(0, o.positiveSpeechThreshold, o.negativeSpeechThreshold, o.redemptionMs, o.minSpeechMs);
    this.listening = true;
  }

  /** Stops listening; start() resumes. */
  pause(): void {
    this.auricle.stop();
    this.listening = false;
  }

  destroy(): void {
    this.pause();
  }
}

/** 16-bit PCM WAV, like vad-web's utils.encodeWAV. */
function encodeWAV(samples: Float32Array, sampleRate = 16000): ArrayBuffer {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const str = (offset: number, s: string) => [...s].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
  str(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  str(8, 'WAVE');
  str(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  str(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buffer;
}

export const utils = { encodeWAV };
