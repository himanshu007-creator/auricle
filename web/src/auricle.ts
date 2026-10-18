// The orchestrator: owns the audio graph, the brain worker, the conversation
// state machine and the backend calls. The audio-side decisions (duck, stop)
// are made in the worker and reach the playback worklet without this file.
import { AudioGraph, type LoopbackMode } from './audio.ts';
import { Emitter } from './emitter.ts';
import { heardText, subsequenceMask, takeSentences, type SpokenSentence } from './heard.ts';
import type { ChatMessage, OpenAICompat } from './backend/openai.ts';
import type { BrainToMain, MainToBrain, Profile, ThresholdSlot, TurnEventName } from './protocol.ts';
import { encodeWav } from './wav.ts';
import BrainWorker from './worker/brain.worker.ts?worker&inline';
import { base64ToBytes } from './wasm.ts';
import { SessionTracker, type SessionStats } from './stats.ts';

/** The brain WASM (Auricle v1 weights inside), loaded on demand so `import 'auricle'` stays light. */
async function brainWasm(): Promise<Uint8Array<ArrayBuffer>> {
  const { BRAIN_WASM_B64 } = await import('./generated/brain_wasm_auricle.ts');
  return base64ToBytes(BRAIN_WASM_B64);
}

export type AuricleState = 'idle' | 'listening' | 'user_speaking' | 'thinking' | 'agent_speaking';
export type CleanupMode = 'verbatim' | 'rules' | 'rules+llm';

/** Options for `Auricle.create`. Everything is optional; with no `backend` the instance is a detector only. */
export interface AuricleOptions {
  /** Speech-to-text, chat and text-to-speech over one OpenAI-compatible base URL. Required for conversation mode. */
  backend?: OpenAICompat;
  /** 'vad' never calls the backend; 'conversation' needs one. Defaults from `backend`. */
  mode?: 'vad' | 'conversation';
  /** 'speakers' (default) raises the thresholds while the agent talks so its own echo cannot interrupt it; 'headphones' does not need that. */
  profile?: Profile;
  /** Whether to route the agent's audio back through the microphone's echo canceller: 'auto' (default) does it where the browser cannot cancel on its own. */
  loopback?: LoopbackMode;
  /** The system prompt for conversation mode. */
  systemPrompt?: string;
  /** Transcript cleanup before the model sees it: 'verbatim', 'rules' (default: fillers, repeats, stutters removed, words only ever deleted) or 'rules+llm'. */
  cleanup?: CleanupMode;
  /** How long to wait for more speech when the turn ends on a filler or conjunction. */
  openTurnWaitMs?: number;
  /** Start speech-to-text at the first silence candidate instead of after the hangover (default true). */
  eagerStt?: boolean;
  /** Send the model's first clause to TTS at a comma instead of waiting for a full sentence (default true). */
  eagerFirstChunk?: boolean;
}

export interface TurnMetrics {
  candidateToConfirmMs?: number;
  /** STT round trip. */
  sttMs?: number;
  /** Time the pipeline actually waited for STT after speech_end (0 when eager STT finished during the hangover). */
  sttWaitMs?: number;
  firstTokenMs?: number;
  firstAudioMs?: number;
  /** speech_end to first audio chunk queued. */
  endToFirstAudioMs?: number;
}

export interface Turn {
  verbatim: string;
  clean: string;
  /** One flag per whitespace token of `verbatim`: false means removed by cleanup. */
  mask: boolean[];
  cleanup: CleanupMode;
  metrics: TurnMetrics;
}

/** Everything an instance emits; subscribe with `auricle.on(name, fn)`, which returns an unsubscribe function. */
export interface AuricleEvents extends Record<string, unknown> {
  /** idle | listening | user_speaking | thinking | agent_speaking. */
  state: AuricleState;
  /** Every 32 ms: the speech probability and the time since start. */
  frame: { p: number; tMs: number };
  /** The turn machine's graded events: speech_candidate, speech_confirmed, silence_candidate, speech_end, misfire. */
  event: { name: TurnEventName; tMs: number };
  /** The speech candidate, about 100 ms into speech. */
  speechStart: { tMs: number };
  /** A finished segment: 16 kHz samples with 512 ms of pre-roll and its span. */
  speechEnd: { samples: Float32Array; startMs: number; endMs: number };
  /** A transcribed user turn, with verbatim and cleaned text and the timings. */
  turn: Turn;
  /** Streamed agent text: the new piece and the whole so far. */
  agentText: { delta: string; text: string };
  /** The reply finished playing in full. */
  agentDone: { text: string };
  /** The user interrupted: what they actually heard, what the model had generated, and how long the audio took to stop. */
  interrupted: { heardText: string; generatedText: string; stopLatencyMs: number };
  /** Emitted when the first audio of a reply is queued, with the turn's timings so far. */
  metrics: TurnMetrics;
  /** The playback worklet started or stopped producing sound. */
  agentPlaying: boolean;
  /** A failure after `start()`: backend errors, a revoked microphone, a worker crash. */
  error: Error;
}

interface Response {
  text: string;
  spoken: SpokenSentence[];
  abort: AbortController;
  assistantIndex: number;
  startedPlayed: number;
  llmDone: boolean;
  ttsDone: boolean;
  audioQueued: boolean;
}

const VAD_RATE = 16000;
/** Segments whose loudest 100 ms stays under this are not sent to STT: Whisper invents text on near-silence. */
const MIN_SEGMENT_PEAK_DBFS = -45;

/** Peak RMS over 100 ms windows, in dBFS. */
export function peakWindowDbfs(samples: Float32Array, windowSamples = VAD_RATE / 10): number {
  let peak = 0;
  for (let start = 0; start + windowSamples <= samples.length; start += windowSamples) {
    let sum = 0;
    for (let i = start; i < start + windowSamples; i++) sum += samples[i] * samples[i];
    peak = Math.max(peak, sum / windowSamples);
  }
  return 10 * Math.log10(peak + 1e-12);
}
const LLM_CLEANUP_PROMPT =
  'You clean up spoken transcripts. Delete fillers (um, uh), false starts, stutters and immediate repeats. ' +
  'Only delete words. Never add, reorder or rephrase anything. Reply with the cleaned text only.';

/**
 * The whole listener: microphone in, graded speech events, and in conversation
 * mode the speech-to-text, chat and text-to-speech round trip with ducking,
 * barge-in and a message history cut to what the user actually heard.
 *
 * ```ts
 * const auricle = await Auricle.create({ backend });
 * auricle.on('state', (s) => ui.set(s));
 * button.onclick = () => auricle.start();
 * ```
 */
export class Auricle extends Emitter<AuricleEvents> {
  readonly options: Required<Pick<AuricleOptions, 'mode' | 'profile' | 'loopback' | 'cleanup' | 'openTurnWaitMs' | 'systemPrompt' | 'eagerStt' | 'eagerFirstChunk'>> & AuricleOptions;
  state: AuricleState = 'idle';
  messages: ChatMessage[] = [];

  private worker: Worker | null = null;
  private graph: AudioGraph | null = null;
  private response: Response | null = null;
  private interruptedResponse: { response: Response; at: number } | null = null;
  private segmentQueue: Promise<void> = Promise.resolve();
  private pendingVerbatim = '';
  private waitingForSpeech: ((resumed: boolean) => void) | null = null;
  private lastPlayed = 0;
  private playing = false;
  private requestId = 0;
  private pendingWorkerReplies = new Map<number, (msg: BrainToMain) => void>();
  private candidateAtMs: number | null = null;
  private metrics: TurnMetrics = {};
  private eager: { id: number; promise: Promise<string>; abort: AbortController; startedAt: number } | null = null;
  private speechEndAt = 0;
  private readonly tracker: SessionTracker;

  private constructor(options: AuricleOptions) {
    super();
    this.tracker = new SessionTracker(this);
    this.options = {
      mode: options.backend ? 'conversation' : 'vad',
      profile: 'speakers',
      loopback: 'auto',
      cleanup: 'rules',
      openTurnWaitMs: 1500,
      eagerStt: true,
      eagerFirstChunk: true,
      systemPrompt:
        'You are a voice assistant. Keep answers short and conversational. The user speaks; transcripts may contain self-corrections, where later words replace earlier ones.',
      ...options,
    };
    if (this.options.mode === 'conversation' && !this.options.backend) throw new Error('conversation mode needs a backend');
    this.messages = [{ role: 'system', content: this.options.systemPrompt }];
  }

  /** Builds an instance. Nothing is downloaded and the microphone is not touched until `start()`. */
  static async create(options: AuricleOptions = {}): Promise<Auricle> {
    return new Auricle(options);
  }

  /** Whether the agent's audio is being routed through the microphone's echo canceller. */
  get loopbackActive(): boolean {
    return this.graph?.loopbackActive ?? false;
  }

  /** The audio context's sample rate, 0 before `start()`. */
  get sampleRate(): number {
    return this.graph?.context.sampleRate ?? 0;
  }

  /** Measured counters for this session: audio sent to STT against session length, TTS characters avoided by interruption, stop latency. A fresh snapshot each call. */
  get stats(): SessionStats {
    return this.tracker.snapshot();
  }

  /** Asks for the mic, builds the audio graph and starts listening. Call from a user gesture. */
  async start(): Promise<void> {
    if (this.state !== 'idle') return;
    this.tracker.reset();
    const worker = new BrainWorker();
    this.worker = worker;
    const wasm = (await brainWasm()).buffer;
    await new Promise<void>((resolve) => {
      worker.onmessage = (e: MessageEvent<BrainToMain>) => e.data.type === 'ready' && resolve();
      worker.postMessage({ type: 'init', wasm, model: 1 } satisfies MainToBrain, [wasm]);
    });
    worker.onmessage = (e: MessageEvent<BrainToMain>) => this.onWorker(e.data);

    this.graph = await AudioGraph.create({ loopback: this.options.loopback });
    const captureChannel = new MessageChannel();
    const playbackChannel = new MessageChannel();
    this.graph.capture.port.postMessage({ type: 'connect', port: captureChannel.port1 }, [captureChannel.port1]);
    this.graph.playback.port.postMessage({ type: 'connect', port: playbackChannel.port1 }, [playbackChannel.port1]);
    worker.postMessage({ type: 'ports', capture: captureChannel.port2, playback: playbackChannel.port2 }, [captureChannel.port2, playbackChannel.port2]);
    this.graph.playback.port.onmessage = (e) => this.onPlayback(e.data);
    this.send({ type: 'profile', value: this.options.profile });
    if (this.options.backend) {
      this.options.backend.config.decodeAudio ??= (data) => this.graph!.context.decodeAudioData(data);
      void this.options.backend.warmUp();
    }
    this.setState('listening');
  }

  /** Stops the agent, closes the audio graph, releases the microphone and ends the worker. `start()` again for a new session. */
  stop(): void {
    this.response?.abort.abort();
    this.response = null;
    this.graph?.close();
    this.graph = null;
    this.worker?.terminate();
    this.worker = null;
    this.setState('idle');
  }

  /** Switches between 'speakers' and 'headphones' while running. */
  setProfile(profile: Profile): void {
    this.options.profile = profile;
    this.send({ type: 'profile', value: profile });
  }

  /**
   * Sets the turn thresholds for one slot: 0 while the agent is silent, 1 while it speaks.
   * `minSpeechMs`: a confirmed segment with fewer voiced milliseconds ends as a misfire (default 100).
   */
  setThresholds(which: ThresholdSlot, positive: number, negative: number, hangoverMs: number, minSpeechMs = 100): void {
    this.send({ type: 'thresholds', which, positive, negative, hangoverMs, minSpeechMs });
  }

  /** Changes the transcript cleanup mode for the next turn. */
  setCleanup(mode: CleanupMode): void {
    this.options.cleanup = mode;
  }

  /** Plays audio as if the agent said it (the demo's pretend agent). */
  playClip(samples: Float32Array, rate: number): void {
    const copy = samples.slice();
    this.graph?.playback.port.postMessage({ type: 'chunk', samples: copy, rate }, [copy.buffer]);
    if (this.state === 'listening') this.setState('agent_speaking');
  }

  /** Manual hard stop of whatever the agent is saying. */
  interrupt(): void {
    this.graph?.playback.port.postMessage({ type: 'stop' });
    this.onInterrupt();
  }

  private send(msg: MainToBrain): void {
    this.worker?.postMessage(msg);
  }

  private ask<T extends BrainToMain>(msg: MainToBrain & { id: number }): Promise<T> {
    return new Promise((resolve) => {
      this.pendingWorkerReplies.set(msg.id, (reply) => resolve(reply as T));
      this.send(msg);
    });
  }

  private setState(state: AuricleState): void {
    if (this.state === state) return;
    this.state = state;
    this.emit('state', state);
  }

  private onWorker(msg: BrainToMain): void {
    switch (msg.type) {
      case 'frame':
        this.emit('frame', msg);
        break;
      case 'event':
        this.emit('event', msg);
        this.onTurnEvent(msg.name, msg.tMs);
        break;
      case 'segment':
        if (msg.provisional) {
          if (this.options.mode === 'conversation' && this.options.eagerStt) this.startEagerStt(msg.id, msg.samples);
          break;
        }
        this.emit('speechEnd', msg);
        if (this.options.mode === 'conversation') {
          const { samples, id } = msg;
          this.speechEndAt = performance.now();
          this.segmentQueue = this.segmentQueue.then(() => this.handleSegment(samples, id)).catch((err) => this.emit('error', err as Error));
        }
        break;
      case 'clean':
      case 'verify':
        this.pendingWorkerReplies.get(msg.id)?.(msg);
        this.pendingWorkerReplies.delete(msg.id);
        break;
    }
  }

  private onTurnEvent(name: TurnEventName, tMs: number): void {
    switch (name) {
      case 'speech_candidate':
        this.candidateAtMs = tMs;
        this.waitingForSpeech?.(true);
        break;
      case 'speech_confirmed':
        this.metrics.candidateToConfirmMs = this.candidateAtMs === null ? undefined : tMs - this.candidateAtMs;
        this.emit('speechStart', { tMs });
        if (this.options.mode === 'conversation' && this.state !== 'agent_speaking') this.setState('user_speaking');
        break;
      case 'interrupt':
        this.onInterrupt();
        break;
    }
  }

  private onPlayback(msg: { type: 'playing'; value: boolean; played: number } | { type: 'flushed'; played: number }): void {
    this.lastPlayed = msg.played;
    if (msg.type === 'playing') {
      this.playing = msg.value;
      this.emit('agentPlaying', msg.value);
      if (msg.value && this.state !== 'user_speaking') this.setState('agent_speaking');
      if (!msg.value) this.maybeFinishResponse();
      if (!msg.value && !this.response && this.state === 'agent_speaking') this.setState('listening');
    } else if (this.interruptedResponse) {
      const { response, at } = this.interruptedResponse;
      this.interruptedResponse = null;
      const rate = response.spoken[0]?.rate ?? 1;
      const heard = heardText(response.spoken, (msg.played - response.startedPlayed) / rate);
      const message = this.messages[response.assistantIndex];
      if (message) message.content = heard ? `${heard} [interrupted by user]` : '[interrupted by user before any audio was heard]';
      this.emit('interrupted', { heardText: heard, generatedText: response.text, stopLatencyMs: performance.now() - at });
    }
  }

  private onInterrupt(): void {
    const at = performance.now();
    if (this.response) {
      this.response.abort.abort();
      this.interruptedResponse = { response: this.response, at };
      this.response = null;
    } else if (this.state === 'agent_speaking') {
      this.emit('interrupted', { heardText: '', generatedText: '', stopLatencyMs: 0 });
    }
    if (this.options.mode === 'conversation') this.setState('user_speaking');
  }

  /** Transcribes the speech heard so far while the hangover is still running. A newer provisional replaces an older one. */
  private startEagerStt(id: number, samples: Float32Array): void {
    this.eager?.abort.abort();
    if (peakWindowDbfs(samples) < MIN_SEGMENT_PEAK_DBFS) {
      this.eager = null;
      return;
    }
    const abort = new AbortController();
    const promise = this.options.backend!.transcribe(encodeWav(samples, VAD_RATE), abort.signal);
    promise.catch(() => {});
    this.eager = { id, promise, abort, startedAt: performance.now() };
  }

  private async handleSegment(samples: Float32Array, id: number): Promise<void> {
    const backend = this.options.backend!;
    const eager = this.eager?.id === id ? this.eager : null;
    this.eager = null;
    if (!eager && peakWindowDbfs(samples) < MIN_SEGMENT_PEAK_DBFS) {
      if (!this.pendingVerbatim) this.setState('listening');
      return;
    }
    this.setState('thinking');
    const t0 = performance.now();
    let verbatim: string;
    try {
      // The final segment is the provisional one plus hangover silence, so the eager transcript is the transcript.
      verbatim = eager ? await eager.promise : await backend.transcribe(encodeWav(samples, VAD_RATE));
    } catch {
      verbatim = await backend.transcribe(encodeWav(samples, VAD_RATE));
    }
    this.metrics.sttMs = Math.round(performance.now() - (eager ? eager.startedAt : t0));
    this.metrics.sttWaitMs = Math.round(performance.now() - t0);
    if (!verbatim) {
      if (!this.pendingVerbatim) this.setState('listening');
      return;
    }
    this.pendingVerbatim = this.pendingVerbatim ? `${this.pendingVerbatim} ${verbatim}` : verbatim;

    const rules = await this.ask<Extract<BrainToMain, { type: 'clean' }>>({ type: 'clean', id: ++this.requestId, text: this.pendingVerbatim });
    if (rules.endsOpen && (await this.waitForMoreSpeech())) {
      return; // the next segment merges into this turn
    }
    const text = this.pendingVerbatim;
    this.pendingVerbatim = '';
    const turn: Turn = { verbatim: text, clean: rules.clean, mask: rules.mask, cleanup: this.options.cleanup, metrics: this.metrics };
    this.metrics = {};
    if (this.options.cleanup === 'verbatim') {
      turn.clean = text;
      turn.mask = turn.mask.map(() => true);
    } else if (this.options.cleanup === 'rules+llm') {
      await this.llmCleanup(turn);
    }
    this.emit('turn', turn);
    await this.respond(turn.clean);
  }

  /** Resolves true if the user starts speaking again within the open-turn window. */
  private waitForMoreSpeech(): Promise<boolean> {
    this.setState('listening');
    return new Promise((resolve) => {
      const timer = setTimeout(() => finish(false), this.options.openTurnWaitMs);
      const finish = (resumed: boolean) => {
        clearTimeout(timer);
        this.waitingForSpeech = null;
        resolve(resumed);
      };
      this.waitingForSpeech = finish;
    });
  }

  private async llmCleanup(turn: Turn): Promise<void> {
    try {
      let candidate = '';
      const prompt: ChatMessage[] = [
        { role: 'system', content: LLM_CLEANUP_PROMPT },
        { role: 'user', content: turn.verbatim },
      ];
      for await (const delta of this.options.backend!.chat(prompt, undefined, { maxTokens: 400 })) candidate += delta;
      candidate = candidate.trim();
      const verify = await this.ask<Extract<BrainToMain, { type: 'verify' }>>({ type: 'verify', id: ++this.requestId, verbatim: turn.verbatim, candidate });
      const mask = verify.ok ? subsequenceMask(turn.verbatim, candidate) : null;
      if (mask) {
        turn.clean = candidate;
        turn.mask = mask;
      }
      // A verifier failure keeps the rules output: cleanup can never invent words.
    } catch (err) {
      this.emit('error', err as Error);
    }
  }

  private async respond(userText: string): Promise<void> {
    const backend = this.options.backend!;
    this.messages.push({ role: 'user', content: userText });
    const response: Response = {
      text: '',
      spoken: [],
      abort: new AbortController(),
      assistantIndex: this.messages.push({ role: 'assistant', content: '' }) - 1,
      startedPlayed: this.lastPlayed,
      llmDone: false,
      ttsDone: false,
      audioQueued: false,
    };
    this.response = response;
    this.setState('thinking');
    const t0 = performance.now();
    let ttsChain: Promise<void> = Promise.resolve();
    const speak = (sentence: string) => {
      ttsChain = ttsChain.then(async () => {
        if (response.abort.signal.aborted) return;
        const entry: SpokenSentence = { text: sentence, samples: 0, rate: 0 };
        response.spoken.push(entry);
        for await (const chunk of backend.speak(sentence, response.abort.signal)) {
          if (response.abort.signal.aborted) return;
          if (!response.audioQueued) {
            response.audioQueued = true;
            this.metrics.firstAudioMs = Math.round(performance.now() - t0);
            this.metrics.endToFirstAudioMs = Math.round(performance.now() - this.speechEndAt);
            this.emit('metrics', { ...this.metrics });
          }
          entry.rate = chunk.rate;
          entry.samples += chunk.samples.length;
          this.graph?.playback.port.postMessage({ type: 'chunk', samples: chunk.samples, rate: chunk.rate }, [chunk.samples.buffer]);
        }
      });
    };

    try {
      let buffer = '';
      for await (const delta of backend.chat(this.messages.slice(0, response.assistantIndex), response.abort.signal)) {
        if (!response.text) this.metrics.firstTokenMs = Math.round(performance.now() - t0);
        response.text += delta;
        this.messages[response.assistantIndex].content = response.text;
        this.emit('agentText', { delta, text: response.text });
        buffer += delta;
        const { sentences, rest } = takeSentences(buffer, response.spoken.length === 0 ? 12 : 20, this.options.eagerFirstChunk && response.spoken.length === 0);
        buffer = rest;
        sentences.forEach(speak);
      }
      if (buffer.trim()) speak(buffer.trim());
      response.llmDone = true;
      await ttsChain;
      response.ttsDone = true;
      this.maybeFinishResponse();
    } catch (err) {
      if (!response.abort.signal.aborted) {
        this.emit('error', err as Error);
        if (this.response === response) {
          this.response = null;
          this.setState('listening');
        }
      }
    }
  }

  private maybeFinishResponse(): void {
    const r = this.response;
    if (!r || !r.ttsDone || (r.audioQueued && this.playing)) return;
    this.response = null;
    this.emit('agentDone', { text: r.text });
    this.setState('listening');
  }
}
