// Session accounting: what the microphone session cost in things that are
// billed (audio sent to speech-to-text, characters sent to text-to-speech)
// against what it would have cost without the listener. Every field is a
// count taken from the events as they happen, so the numbers are measured
// for this session, not estimated.
import type { Emitter } from './emitter.ts';
import type { AuricleEvents } from './auricle.ts';

/**
 * Counters for the current session. Read `auricle.stats` (or `vad.stats`,
 * `useVad().stats`) at any time; the object is a fresh snapshot.
 *
 * What the numbers mean for cost:
 * - `speechMs` is the only audio that leaves the browser for speech-to-text.
 *   A streaming STT connection that is open for the whole session bills
 *   `sessionMs`; the ratio `sttShare` is how much of that you did not send.
 * - `ttsCharsAvoided` is text the model generated that was never synthesised
 *   or played because the user interrupted. With a per-character TTS price
 *   that is the saved spend; with a token-priced model the same text is also
 *   output the user never heard.
 */
export interface SessionStats {
  /** Milliseconds since `start()`; 0 before the session starts. */
  sessionMs: number;
  /** Milliseconds of audio inside finished speech segments: the audio that is sent to speech-to-text. */
  speechMs: number;
  /** `sessionMs - speechMs`: audio that stayed on the device. */
  silenceMs: number;
  /** `speechMs / sessionMs`, 0 to 1. An always-on streaming STT connection would bill 1. */
  sttShare: number;
  /** Finished speech segments. */
  segments: number;
  /** Speech candidates that never confirmed, so nothing was sent anywhere. */
  misfires: number;
  /** Completed user turns (transcribed and handed to the model), conversation mode only. */
  turns: number;
  /** Characters of agent text that were sent to text-to-speech and played. */
  ttsCharsGenerated: number;
  /** Characters the model produced that were never synthesised or never heard because the user interrupted. */
  ttsCharsAvoided: number;
  /** How many times the user interrupted the agent. */
  interruptions: number;
  /** Median milliseconds from the user's confirmed speech to the agent's audio stopping, over all interruptions; null before the first one. */
  stopLatencyMedianMs: number | null;
  /** Median of `endToFirstAudioMs` over the turns that got a spoken reply; null before the first one. */
  responseMedianMs: number | null;
}

/** Subscribes to an Auricle instance and keeps the counters. One per instance. */
export class SessionTracker {
  private startedAt = 0;
  private speechMs = 0;
  private segments = 0;
  private misfires = 0;
  private turns = 0;
  private generated = 0;
  private avoided = 0;
  private stops: number[] = [];
  private responses: number[] = [];

  constructor(source: Emitter<AuricleEvents>) {
    source.on('state', (s) => {
      if (s !== 'idle' && !this.startedAt) this.startedAt = performance.now();
    });
    source.on('speechEnd', ({ startMs, endMs }) => {
      this.speechMs += Math.max(0, endMs - startMs);
      this.segments++;
    });
    source.on('event', ({ name }) => name === 'misfire' && this.misfires++);
    source.on('turn', () => this.turns++);
    source.on('agentDone', ({ text }) => (this.generated += text.length));
    source.on('interrupted', ({ heardText, generatedText, stopLatencyMs }) => {
      this.generated += heardText.length;
      this.avoided += Math.max(0, generatedText.length - heardText.length);
      this.stops.push(stopLatencyMs);
    });
    source.on('metrics', (m) => m.endToFirstAudioMs !== undefined && this.responses.push(m.endToFirstAudioMs));
  }

  /** Resets every counter; called by `start()` so a second session starts clean. */
  reset(): void {
    this.startedAt = performance.now();
    this.speechMs = this.segments = this.misfires = this.turns = this.generated = this.avoided = 0;
    this.stops = [];
    this.responses = [];
  }

  snapshot(): SessionStats {
    const sessionMs = this.startedAt ? performance.now() - this.startedAt : 0;
    return {
      sessionMs,
      speechMs: this.speechMs,
      silenceMs: Math.max(0, sessionMs - this.speechMs),
      sttShare: sessionMs ? Math.min(1, this.speechMs / sessionMs) : 0,
      segments: this.segments,
      misfires: this.misfires,
      turns: this.turns,
      ttsCharsGenerated: this.generated,
      ttsCharsAvoided: this.avoided,
      interruptions: this.stops.length,
      stopLatencyMedianMs: median(this.stops),
      responseMedianMs: median(this.responses),
    };
  }
}

function median(xs: number[]): number | null {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

/** One line of the stats for a log or a status bar, e.g. `session 2:14 · sent to STT 0:31 (23%) · TTS chars 812, avoided 1,040 · 2 interruptions, median stop 41 ms`. */
export function formatStats(s: SessionStats): string {
  const mmss = (ms: number) => `${Math.floor(ms / 60000)}:${String(Math.floor((ms % 60000) / 1000)).padStart(2, '0')}`;
  const parts = [`session ${mmss(s.sessionMs)}`, `sent to STT ${mmss(s.speechMs)} (${Math.round(100 * s.sttShare)}%)`];
  if (s.ttsCharsGenerated || s.ttsCharsAvoided) parts.push(`TTS chars ${s.ttsCharsGenerated.toLocaleString()}, avoided ${s.ttsCharsAvoided.toLocaleString()}`);
  if (s.interruptions) parts.push(`${s.interruptions} interruption${s.interruptions === 1 ? '' : 's'}, median stop ${Math.round(s.stopLatencyMedianMs ?? 0)} ms`);
  if (s.misfires) parts.push(`${s.misfires} misfire${s.misfires === 1 ? '' : 's'}`);
  return parts.join(' · ');
}
