import { describe, expect, it } from 'vitest';
import { heardText, subsequenceMask, takeSentences } from './heard.ts';
import { encodeWav, parseWavHeader, pcmToFloat } from './wav.ts';

describe('heardText', () => {
  const spoken = [
    { text: 'First sentence here.', samples: 24000, rate: 24000 },
    { text: 'Second one is a bit longer.', samples: 48000, rate: 24000 },
  ];
  it('returns whole sentences that finished', () => {
    expect(heardText(spoken, 1.0)).toBe('First sentence here.');
  });
  it('cuts the partial sentence at a word boundary', () => {
    expect(heardText(spoken, 2.0)).toBe('First sentence here. Second one is');
  });
  it('returns everything when played past the end', () => {
    expect(heardText(spoken, 10)).toBe('First sentence here. Second one is a bit longer.');
  });
});

describe('takeSentences', () => {
  it('splits complete sentences and keeps the tail', () => {
    const { sentences, rest } = takeSentences('Hello there, how are you today? I am fine. And you');
    expect(sentences).toEqual(['Hello there, how are you today?']);
    expect(rest).toBe('I am fine. And you'); // short sentence merges into the next TTS call
  });
  it('eager mode cuts the first chunk at a clause boundary', () => {
    const { sentences, rest } = takeSentences('Sure thing, I can book that, and then confirm by text. ', 12, true);
    expect(sentences).toEqual(['Sure thing, I can book that,', 'and then confirm by text.']);
    expect(rest).toBe('');
    // A clause shorter than minChars waits for more text.
    expect(takeSentences('Sure thing, I can', 12, true).sentences).toEqual([]);
  });
  it('waits on a dot at the buffer edge', () => {
    expect(takeSentences('The value is 3.').sentences).toEqual([]);
  });
});

describe('subsequenceMask', () => {
  it('marks kept tokens', () => {
    expect(subsequenceMask('Um, I I want to go.', 'I want to go.')).toEqual([false, true, false, true, true, true]);
    expect(subsequenceMask('I want to go', 'I go now')).toBeNull();
  });
});

describe('wav', () => {
  it('round-trips through encode and parse', async () => {
    const samples = new Float32Array([0, 0.5, -0.5, 1]);
    const bytes = new Uint8Array(await encodeWav(samples, 16000).arrayBuffer());
    const header = parseWavHeader(bytes)!;
    expect(header.sampleRate).toBe(16000);
    expect(header.dataOffset).toBe(44);
    const { samples: back } = pcmToFloat(bytes.subarray(44), 16, 1);
    expect(Array.from(back).map((v) => +v.toFixed(3))).toEqual([0, 0.5, -0.5, 1]);
  });
});
