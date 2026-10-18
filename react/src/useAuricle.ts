// One hook over the Auricle class: the conversation as React state.
import { useCallback, useEffect, useRef, useState } from 'react';
import { Auricle, type AuricleOptions, type AuricleState, type SessionStats, type Turn, type TurnMetrics } from 'auricle';

/** `Auricle.create` options plus `autoStart`. Options are read once, on the first `start()`. */
export interface UseAuricleOptions extends AuricleOptions {
  /** Call start() as soon as the component mounts (needs a prior user gesture for the mic). */
  autoStart?: boolean;
  /** How often `stats` refreshes while running, in ms (default 1000; 0 disables). */
  statsIntervalMs?: number;
}

/** What `useAuricle` returns. */
export interface UseAuricle {
  /** idle | listening | user_speaking | thinking | agent_speaking. */
  state: AuricleState;
  /** Latest speech probability, 0 to 1, updated every 32 ms while running. */
  probability: number;
  /** Completed user turns, oldest first. */
  turns: Turn[];
  /** The reply being streamed, or the last complete one. */
  agentText: string;
  /** Set after an interruption: what the user actually heard. */
  heardText: string | null;
  /** Timings of the latest turn: STT wait, first token, first audio. */
  metrics: TurnMetrics | null;
  /** Measured counters for this session: audio sent to STT against session length, TTS characters avoided by interruption, stop latency. */
  stats: SessionStats | null;
  error: Error | null;
  /** Asks for the microphone and starts. Call from a click handler. */
  start: () => Promise<void>;
  /** Stops the agent and releases the microphone. */
  stop: () => void;
  /** Hard-stops whatever the agent is saying, as if the user had spoken. */
  interrupt: () => void;
  /** The underlying instance, for playClip, setThresholds and other calls. */
  auricle: Auricle | null;
}

export function useAuricle(options: UseAuricleOptions = {}): UseAuricle {
  const ref = useRef<Auricle | null>(null);
  const [state, setState] = useState<AuricleState>('idle');
  const [probability, setProbability] = useState(0);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [agentText, setAgentText] = useState('');
  const [heardText, setHeardText] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<TurnMetrics | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [stats, setStats] = useState<SessionStats | null>(null);
  // Options are read once; changing them after start needs stop() and start().
  const optionsRef = useRef(options);

  const start = useCallback(async () => {
    if (ref.current) return;
    const { autoStart: _ignored, statsIntervalMs: _ignored2, ...auricleOptions } = optionsRef.current;
    const d = await Auricle.create(auricleOptions);
    ref.current = d;
    d.on('state', setState);
    d.on('frame', ({ p }) => setProbability(p));
    d.on('turn', (turn) => {
      setTurns((t) => [...t, turn]);
      setAgentText('');
      setHeardText(null);
    });
    d.on('agentText', ({ text }) => setAgentText(text));
    d.on('interrupted', ({ heardText }) => setHeardText(heardText));
    d.on('metrics', setMetrics);
    d.on('error', setError);
    try {
      await d.start();
    } catch (err) {
      ref.current = null;
      setError(err as Error);
      throw err;
    }
  }, []);

  useEffect(() => {
    const every = optionsRef.current.statsIntervalMs ?? 1000;
    if (state === 'idle' || !every) return;
    const id = setInterval(() => ref.current && setStats(ref.current.stats), every);
    return () => clearInterval(id);
  }, [state]);

  const stop = useCallback(() => {
    ref.current?.stop();
    ref.current = null;
    setState('idle');
    setProbability(0);
  }, []);

  const interrupt = useCallback(() => ref.current?.interrupt(), []);

  useEffect(() => {
    if (options.autoStart) void start().catch(() => {});
    return stop;
    // Mount and unmount only: options are intentionally read once.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { state, probability, turns, agentText, heardText, metrics, stats, error, start, stop, interrupt, auricle: ref.current };
}
