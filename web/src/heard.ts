// Heard-text accounting: given what was sent to TTS and how much audio was
// actually played, work out the text the user really heard.

export interface SpokenSentence {
  text: string;
  /** Audio received for this sentence so far. */
  samples: number;
  rate: number;
}

/** Cuts at the last word boundary at or before `fraction` of the text. */
function cutAtWord(text: string, fraction: number): string {
  if (fraction >= 1) return text;
  const limit = Math.floor(text.length * fraction);
  const cut = text.lastIndexOf(' ', limit);
  return cut <= 0 ? '' : text.slice(0, cut);
}

/** `playedSeconds` is the agent audio played since this response started. */
export function heardText(spoken: SpokenSentence[], playedSeconds: number): string {
  const parts: string[] = [];
  let elapsed = 0;
  for (const s of spoken) {
    const duration = s.rate > 0 ? s.samples / s.rate : 0;
    if (playedSeconds >= elapsed + duration) {
      parts.push(s.text);
    } else {
      const partial = cutAtWord(s.text, (playedSeconds - elapsed) / Math.max(duration, 1e-6));
      if (partial) parts.push(partial);
      break;
    }
    elapsed += duration;
  }
  return parts.join(' ').trim();
}

/**
 * Splits streamed text into complete sentences for TTS; returns them and the
 * leftover. Sentences shorter than `minChars` are merged into the next one so
 * TTS is not called for "Yes." alone. A terminator at the very end of the
 * buffer is not trusted yet (it may be "3." of "3.5"); the caller flushes the
 * rest when the stream ends.
 */
export function takeSentences(buffer: string, minChars = 20, eager = false): { sentences: string[]; rest: string } {
  const sentences: string[] = [];
  // Eager: the first chunk may stop at a clause boundary so audio starts sooner.
  const boundary = eager ? /(?:[.!?]["')\]]*|[,;:])\s+/g : /[.!?]["')\]]*\s+/g;
  let match: RegExpExecArray | null;
  let start = 0;
  while ((match = boundary.exec(buffer)) !== null) {
    const end = match.index + match[0].length;
    const sentence = buffer.slice(start, end).trim();
    if (sentence.length >= minChars) {
      sentences.push(sentence);
      start = end;
    }
  }
  return { sentences, rest: buffer.slice(start) };
}

const normalize = (t: string) => t.toLowerCase().replace(/[^\p{L}\p{N}']/gu, '');

/** Keep-mask for `verbatim` tokens from a cleaned text, by greedy subsequence match. Null if not a subsequence. */
export function subsequenceMask(verbatim: string, clean: string): boolean[] | null {
  const source = verbatim.split(/\s+/).filter(Boolean);
  const target = clean.split(/\s+/).filter(Boolean).map(normalize).filter(Boolean);
  const mask = new Array<boolean>(source.length).fill(false);
  let j = 0;
  for (let i = 0; i < source.length && j < target.length; i++) {
    const n = normalize(source[i]);
    if (n === '' ) continue;
    if (n === target[j]) {
      mask[i] = true;
      j++;
    }
  }
  return j === target.length ? mask : null;
}
