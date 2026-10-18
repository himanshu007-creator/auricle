// The only backend: three OpenAI-standard endpoints. Works with OpenAI and any
// server that copies its API, as long as it sends CORS headers.
import { encodeWav, parseWavHeader, pcmToFloat } from '../wav.ts';

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface OpenAICompatConfig {
  baseURL: string;
  apiKey?: string;
  /** Extra headers, for example a vendor key header. */
  headers?: Record<string, string>;
  models: { stt: string; llm: string; tts: string };
  voice?: string;
  /** 'auto' tries pcm, then wav, then mp3 and remembers what worked. */
  ttsFormat?: 'auto' | 'pcm' | 'wav' | 'mp3';
  /** Sample rate of raw pcm responses (OpenAI: 24000). */
  pcmSampleRate?: number;
  /** Decoder for compressed TTS formats; Auricle passes AudioContext.decodeAudioData. */
  decodeAudio?: (data: ArrayBuffer) => Promise<AudioBuffer>;
  temperature?: number;
  /** Extra fields merged into every chat/completions body, e.g. { reasoning_effort: 'low' }. */
  extraBody?: Record<string, unknown>;
}

export interface AudioChunk {
  samples: Float32Array;
  rate: number;
}

export type RoleCheck = { ok: boolean; message: string };

export class BackendError extends Error {
  status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.status = status;
  }
}

/** Trims to end in /v1 (OpenAI convention) unless the caller gave a deeper path. */
export function normalizeBaseURL(url: string): string {
  let base = url.trim().replace(/\/+$/, '');
  base = base.replace(/\/(chat\/completions|audio\/transcriptions|audio\/speech)$/, '');
  if (!/\/v\d+[a-z]*$/.test(base) && !/\/(openai|api)$/.test(base)) base += '/v1';
  return base;
}

export function explainHttpError(status: number): string {
  if (status === 401 || status === 403) return 'The server rejected the API key (401/403).';
  if (status === 404) return 'Not found (404): the base URL path is probably wrong, or this endpoint is not offered.';
  if (status === 429) return 'Rate limited (429): slow down or check your quota.';
  if (status >= 500) return `Server error (${status}).`;
  return `Request failed with status ${status}.`;
}

export function explainFetchError(err: unknown): string {
  if (err instanceof BackendError) return err.message;
  if (err instanceof TypeError) {
    return 'Network error: the server is unreachable, or it sends no CORS headers for browser calls. Self-hosted servers usually need a CORS option switched on; see the README for a proxy example.';
  }
  return err instanceof Error ? err.message : String(err);
}

export class OpenAICompat {
  readonly config: OpenAICompatConfig;
  private ttsFormat: 'pcm' | 'wav' | 'mp3' | null;

  constructor(config: OpenAICompatConfig) {
    this.config = { ...config, baseURL: normalizeBaseURL(config.baseURL) };
    this.ttsFormat = config.ttsFormat && config.ttsFormat !== 'auto' ? config.ttsFormat : null;
  }

  private headers(extra: Record<string, string> = {}): Record<string, string> {
    const h: Record<string, string> = { ...this.config.headers, ...extra };
    if (this.config.apiKey) h.Authorization = `Bearer ${this.config.apiKey}`;
    return h;
  }

  private async request(path: string, init: RequestInit): Promise<Response> {
    const res = await fetch(`${this.config.baseURL}${path}`, init);
    if (!res.ok) {
      let detail = '';
      try {
        detail = (await res.text()).slice(0, 300);
      } catch {
        /* body unreadable */
      }
      throw new BackendError(`${explainHttpError(res.status)}${detail ? ` ${detail}` : ''}`, res.status);
    }
    return res;
  }

  /** Opens the TLS connection ahead of the first turn. Failures are ignored. */
  async warmUp(): Promise<void> {
    try {
      await fetch(`${this.config.baseURL}/models`, { headers: this.headers() });
    } catch {
      /* the first real request will report the problem */
    }
  }

  async transcribe(wav: Blob, signal?: AbortSignal): Promise<string> {
    const form = new FormData();
    form.append('file', wav, 'segment.wav');
    form.append('model', this.config.models.stt);
    form.append('response_format', 'json');
    const res = await this.request('/audio/transcriptions', { method: 'POST', headers: this.headers(), body: form, signal });
    const data = (await res.json()) as { text?: string };
    return (data.text ?? '').trim();
  }

  /** Streams assistant text deltas from a chat completion. */
  async *chat(messages: ChatMessage[], signal?: AbortSignal, options: { maxTokens?: number } = {}): AsyncGenerator<string> {
    const res = await this.request('/chat/completions', {
      method: 'POST',
      headers: this.headers({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({
        ...this.config.extraBody,
        model: this.config.models.llm,
        messages,
        stream: true,
        temperature: this.config.temperature,
        max_tokens: options.maxTokens,
      }),
      signal,
    });
    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';
      for (const line of lines) {
        if (!line.startsWith('data:')) continue;
        const payload = line.slice(5).trim();
        if (payload === '[DONE]') return;
        try {
          const json = JSON.parse(payload) as { choices?: { delta?: { content?: string } }[] };
          const delta = json.choices?.[0]?.delta?.content;
          if (delta) yield delta;
        } catch {
          /* keep-alive or partial line */
        }
      }
    }
  }

  /** Streams decoded audio chunks for `text`. Format negotiation happens once per instance. */
  async *speak(text: string, signal?: AbortSignal): AsyncGenerator<AudioChunk> {
    const order: ('pcm' | 'wav' | 'mp3')[] = this.ttsFormat ? [this.ttsFormat] : ['pcm', 'wav', 'mp3'];
    let lastError: unknown;
    for (const format of order) {
      let res: Response;
      try {
        res = await this.request('/audio/speech', {
          method: 'POST',
          headers: this.headers({ 'Content-Type': 'application/json' }),
          body: JSON.stringify({ model: this.config.models.tts, voice: this.config.voice ?? 'alloy', input: text, response_format: format }),
          signal,
        });
      } catch (err) {
        lastError = err;
        if (err instanceof BackendError && err.status && err.status >= 400 && err.status < 500 && err.status !== 401 && err.status !== 403 && err.status !== 429) {
          continue; // this server does not offer the format; try the next one
        }
        throw err;
      }
      this.ttsFormat = format;
      if (format === 'mp3') {
        if (!this.config.decodeAudio) throw new BackendError('mp3 TTS needs a decodeAudio function');
        const decoded = await this.config.decodeAudio(await res.arrayBuffer());
        yield { samples: decoded.getChannelData(0), rate: decoded.sampleRate };
        return;
      }
      yield* streamPcm(res, format, this.config.pcmSampleRate ?? 24000);
      return;
    }
    throw lastError instanceof Error ? lastError : new BackendError('no TTS format worked');
  }

  /** One small request per role, with the failure explained in plain words. */
  async testConnection(): Promise<Record<'stt' | 'llm' | 'tts', RoleCheck>> {
    const check = async (fn: () => Promise<string>): Promise<RoleCheck> => {
      try {
        return { ok: true, message: await fn() };
      } catch (err) {
        return { ok: false, message: explainFetchError(err) };
      }
    };
    const silence = new Float32Array(16000 * 0.3);
    const [stt, llm, tts] = await Promise.all([
      check(async () => `transcribed silence as "${await this.transcribe(encodeWav(silence, 16000))}"`),
      check(async () => {
        let out = '';
        for await (const d of this.chat([{ role: 'user', content: 'Say hi.' }], undefined, { maxTokens: 5 })) out += d;
        return `replied "${out.trim()}"`;
      }),
      check(async () => {
        let samples = 0;
        let rate = 0;
        for await (const c of this.speak('Hi.')) {
          samples += c.samples.length;
          rate = c.rate;
        }
        return `got ${(samples / rate).toFixed(2)} s of audio as ${this.ttsFormat} at ${rate} Hz`;
      }),
    ]);
    return { stt, llm, tts };
  }
}

/** Streams raw pcm (16-bit mono LE at `pcmRate`) or a WAV body as float chunks. */
async function* streamPcm(res: Response, format: 'pcm' | 'wav', pcmRate: number): AsyncGenerator<AudioChunk> {
  const reader = res.body!.getReader();
  let pending = new Uint8Array(0);
  let header = format === 'pcm' ? { sampleRate: pcmRate, channels: 1, bitsPerSample: 16, format: 1, dataOffset: 0 } : null;
  while (true) {
    const { value, done } = await reader.read();
    if (value) {
      const merged = new Uint8Array(pending.length + value.length);
      merged.set(pending);
      merged.set(value, pending.length);
      pending = merged;
    }
    if (!header) {
      header = parseWavHeader(pending);
      if (!header) {
        if (done) return;
        continue;
      }
      pending = pending.subarray(header.dataOffset);
    }
    const { samples, consumed } = pcmToFloat(pending, header.bitsPerSample, header.channels, header.format === 3);
    pending = pending.subarray(consumed);
    if (samples.length) yield { samples, rate: header.sampleRate };
    if (done) return;
  }
}
