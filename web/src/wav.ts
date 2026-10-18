// Minimal WAV encode / decode for 16-bit PCM. Used for STT uploads and for
// streaming WAV responses from TTS endpoints.

export function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const ascii = (offset: number, s: string) => [...s].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
  ascii(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  ascii(8, 'WAVE');
  ascii(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  ascii(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

export interface WavHeader {
  sampleRate: number;
  channels: number;
  bitsPerSample: number;
  format: number; // 1 = PCM, 3 = float
  /** Byte offset where sample data starts. */
  dataOffset: number;
}

/** Parses a RIFF header from the first bytes of a WAV stream. Returns null if more bytes are needed. */
export function parseWavHeader(bytes: Uint8Array): WavHeader | null {
  if (bytes.length < 12) return null;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const tag = (o: number) => String.fromCharCode(bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]);
  if (tag(0) !== 'RIFF' || tag(8) !== 'WAVE') throw new Error('not a WAV stream');
  let offset = 12;
  let fmt: Omit<WavHeader, 'dataOffset'> | null = null;
  while (offset + 8 <= bytes.length) {
    const id = tag(offset);
    const size = view.getUint32(offset + 4, true);
    if (id === 'fmt ') {
      if (offset + 24 > bytes.length) return null;
      fmt = {
        format: view.getUint16(offset + 8, true),
        channels: view.getUint16(offset + 10, true),
        sampleRate: view.getUint32(offset + 12, true),
        bitsPerSample: view.getUint16(offset + 22, true),
      };
    } else if (id === 'data') {
      if (!fmt) throw new Error('WAV data chunk before fmt chunk');
      return { ...fmt, dataOffset: offset + 8 };
    }
    offset += 8 + size + (size & 1);
  }
  return null;
}

/** Converts interleaved PCM bytes to mono float samples; returns the samples and how many bytes were consumed. */
export function pcmToFloat(bytes: Uint8Array, bitsPerSample: number, channels: number, isFloat = false): { samples: Float32Array; consumed: number } {
  const bytesPerSample = bitsPerSample / 8;
  const frameBytes = bytesPerSample * channels;
  const frames = Math.floor(bytes.length / frameBytes);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const samples = new Float32Array(frames);
  for (let i = 0; i < frames; i++) {
    let sum = 0;
    for (let c = 0; c < channels; c++) {
      const o = i * frameBytes + c * bytesPerSample;
      if (isFloat) sum += view.getFloat32(o, true);
      else if (bitsPerSample === 16) sum += view.getInt16(o, true) / 32768;
      else if (bitsPerSample === 24) sum += ((view.getUint8(o) | (view.getUint8(o + 1) << 8) | (view.getInt8(o + 2) << 16)) / 8388608);
      else if (bitsPerSample === 32) sum += view.getInt32(o, true) / 2147483648;
      else if (bitsPerSample === 8) sum += (view.getUint8(o) - 128) / 128;
    }
    samples[i] = sum / channels;
  }
  return { samples, consumed: frames * frameBytes };
}
