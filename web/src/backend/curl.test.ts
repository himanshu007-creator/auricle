import { describe, expect, it } from 'vitest';
import { parseCurl, tokenize } from './curl.ts';
import { normalizeBaseURL } from './openai.ts';

describe('curl parser', () => {
  it('tokenizes quotes and continuations', () => {
    expect(tokenize(`curl -H "A: b c" \\\n -d '{"x":"y z"}'`)).toEqual(['curl', '-H', 'A: b c', '-d', '{"x":"y z"}']);
  });

  it('reads base URL, key, model and role from one command', () => {
    const r = parseCurl(`curl https://api.example.com/v1/chat/completions \\
  -H "Authorization: Bearer sk-test" \\
  -H "Content-Type: application/json" \\
  -H "X-Vendor-Key: abc" \\
  -d '{"model":"my-llm","messages":[{"role":"user","content":"hi"}]}'`);
    expect(r.baseURL).toBe('https://api.example.com/v1');
    expect(r.apiKey).toBe('sk-test');
    expect(r.models.llm).toBe('my-llm');
    expect(r.headers).toEqual({ 'x-vendor-key': 'abc' });
    expect(r.warnings).toEqual([]);
  });

  it('fills every role from several commands', () => {
    const r = parseCurl(`curl https://h/v1/audio/transcriptions -F model=whisper-1 -F file=@a.wav
curl https://h/v1/audio/speech -d '{"model":"tts-1","voice":"nova","input":"hi"}'
curl https://h/v1/chat/completions -d '{"model":"gpt"}'`);
    expect(r.models).toEqual({ stt: 'whisper-1', tts: 'tts-1', llm: 'gpt' });
    expect(r.voice).toBe('nova');
  });

  it('warns on missing url', () => {
    expect(parseCurl('echo hi').warnings.length).toBe(1);
  });
});

describe('normalizeBaseURL', () => {
  it('ends in /v1 unless a versioned path is given', () => {
    expect(normalizeBaseURL('https://api.openai.com')).toBe('https://api.openai.com/v1');
    expect(normalizeBaseURL('https://api.openai.com/v1/')).toBe('https://api.openai.com/v1');
    expect(normalizeBaseURL('http://localhost:8000/v1/chat/completions')).toBe('http://localhost:8000/v1');
    expect(normalizeBaseURL('https://x.ai/v2beta')).toBe('https://x.ai/v2beta');
  });
});
