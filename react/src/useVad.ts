// Detector-only hooks. `useVad` is the Auricle shape; `useMicVAD` is the
// @ricky0123/vad-react shape, so a component switches by changing one import.
import { useCallback, useEffect, useRef, useState } from 'react';
import { createVad, type Vad, type VadOptions, type SessionStats, type Thresholds } from 'auricle';

/** `createVad` options plus `startOnLoad`. Options are read once, on the first `start()`. */
export interface UseVadOptions extends VadOptions {
  /** Start on mount (the page needs a prior user gesture for the microphone). */
  startOnLoad?: boolean;
  /** How often `stats` refreshes while listening, in ms (default 1000; 0 disables). */
  statsIntervalMs?: number;
}

/** What `useVad` returns. */
export interface UseVad {
  /** True between `start()` and `stop()`. */
  listening: boolean;
  /** True while paused: the microphone stays open but nothing is scored. */
  paused: boolean;
  /** True from the speech candidate until the segment ends or misfires. */
  speaking: boolean;
  /** Latest speech probability, 0 to 1. */
  probability: number;
  /** Finished segments at 16 kHz, oldest first. */
  segments: Float32Array[];
  /** Measured counters for this session, refreshed every `statsIntervalMs`. */
  stats: SessionStats | null;
  /** True while the detector is being created or the microphone requested. */
  loading: boolean;
  error: Error | null;
  /** Asks for the microphone and starts. Call from a click handler. */
  start: () => Promise<void>;
  /** Stops and releases the microphone. */
  stop: () => void;
  /** Keeps the microphone but stops scoring. */
  pause: () => void;
  /** Resumes after `pause`. */
  resume: () => void;
  /** Stops if listening, otherwise starts. */
  toggle: () => void;
  /** Changes thresholds while running. */
  setThresholds: (t: Thresholds) => void;
  /** The underlying detector, for anything the hook leaves out. */
  vad: Vad | null;
}

/**
 * The detector as React state.
 *
 * ```tsx
 * const { listening, speaking, probability, toggle } = useVad({ onSegment: (s) => upload(s.wav()) });
 * ```
 */

export function useVad(options: UseVadOptions = {}): UseVad {
  const ref = useRef<Vad | null>(null);
  const optionsRef = useRef(options);
  const [listening, setListening] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [probability, setProbability] = useState(0);
  const [segments, setSegments] = useState<Float32Array[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [paused, setPaused] = useState(false);
  const [stats, setStats] = useState<SessionStats | null>(null);

  const start = useCallback(async () => {
    if (ref.current) return;
    setLoading(true);
    const o = optionsRef.current;
    try {
      const vad = await createVad({
        ...o,
        onSpeechStart: (at) => {
          setSpeaking(true);
          o.onSpeechStart?.(at);
        },
        onMisfire: (at) => {
          setSpeaking(false);
          o.onMisfire?.(at);
        },
        onSpeechEnd: (audio, span) => {
          setSpeaking(false);
          setSegments((s) => [...s, audio]);
          o.onSpeechEnd?.(audio, span);
        },
        onFrame: (p, tMs) => {
          setProbability(p);
          o.onFrame?.(p, tMs);
        },
      });
      ref.current = vad;
      await vad.start();
      setListening(true);
      setPaused(false);
      setStats(vad.stats);
    } catch (err) {
      ref.current = null;
      setError(err as Error);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const stop = useCallback(() => {
    if (ref.current) setStats(ref.current.stats);
    ref.current?.stop();
    ref.current = null;
    setListening(false);
    setPaused(false);
    setSpeaking(false);
    setProbability(0);
  }, []);

  const pause = useCallback(() => {
    ref.current?.pause();
    setPaused(true);
    setSpeaking(false);
  }, []);
  const resume = useCallback(() => {
    ref.current?.resume();
    setPaused(false);
  }, []);
  const setThresholds = useCallback((t: Thresholds) => ref.current?.setThresholds(t), []);

  useEffect(() => {
    const every = optionsRef.current.statsIntervalMs ?? 1000;
    if (!listening || !every) return;
    const id = setInterval(() => ref.current && setStats(ref.current.stats), every);
    return () => clearInterval(id);
  }, [listening]);

  const toggle = useCallback(() => {
    if (ref.current) stop();
    else void start().catch(() => {});
  }, [start, stop]);

  useEffect(() => {
    if (options.startOnLoad) void start().catch(() => {});
    return stop;
    // Mount and unmount only: options are read once.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { listening, paused, speaking, probability, segments, stats, loading, error, start, stop, pause, resume, toggle, setThresholds, vad: ref.current };
}

/** The `@ricky0123/vad-react` option names. */
export interface UseMicVADOptions {
  onSpeechStart?: () => void;
  onSpeechRealStart?: () => void;
  onSpeechEnd?: (audio: Float32Array) => void;
  onVADMisfire?: () => void;
  onFrameProcessed?: (probabilities: { isSpeech: number; notSpeech: number }) => void;
  positiveSpeechThreshold?: number;
  negativeSpeechThreshold?: number;
  redemptionMs?: number;
  minSpeechMs?: number;
  startOnLoad?: boolean;
}

/** The `@ricky0123/vad-react` return shape (`listening`, `userSpeaking`, `loading`, `errored`, `start`, `pause`, `toggle`) over `useVad`, so a component switches by changing one import. */
export function useMicVAD(options: UseMicVADOptions = {}) {
  const v = useVad({
    startOnLoad: options.startOnLoad,
    onSpeechStart: () => options.onSpeechStart?.(),
    onSpeechConfirmed: () => options.onSpeechRealStart?.(),
    onSpeechEnd: (audio) => options.onSpeechEnd?.(audio),
    onMisfire: () => options.onVADMisfire?.(),
    onFrame: (p) => options.onFrameProcessed?.({ isSpeech: p, notSpeech: 1 - p }),
    thresholds: Object.fromEntries(Object.entries({ positive: options.positiveSpeechThreshold, negative: options.negativeSpeechThreshold, hangoverMs: options.redemptionMs, minSpeechMs: options.minSpeechMs }).filter(([, v]) => v !== undefined)),
  });
  return { listening: v.listening, errored: v.error ? { message: v.error.message } : false, loading: v.loading, userSpeaking: v.speaking, pause: v.stop, start: v.start, toggle: v.toggle };
}
