import { describe, expect, it } from 'vitest';
import { mergeThresholds } from './vad.ts';

const DEFAULTS = { positive: 0.5, negative: 0.35, hangoverMs: 240, minSpeechMs: 100 };

describe('mergeThresholds', () => {
  it('keeps the defaults when fields are explicitly undefined (the useMicVAD shape)', () => {
    expect(mergeThresholds(DEFAULTS, { positive: undefined, negative: undefined, hangoverMs: undefined, minSpeechMs: undefined })).toEqual(DEFAULTS);
    expect(mergeThresholds(DEFAULTS, undefined)).toEqual(DEFAULTS);
  });
  it('applies only the fields that are set', () => {
    expect(mergeThresholds(DEFAULTS, { minSpeechMs: 400 })).toEqual({ ...DEFAULTS, minSpeechMs: 400 });
    expect(mergeThresholds({ ...DEFAULTS, minSpeechMs: 400 }, { hangoverMs: 500 })).toEqual({ ...DEFAULTS, minSpeechMs: 400, hangoverMs: 500 });
  });
});

import { defined } from './options.ts';
describe('defined', () => {
  it('drops undefined fields and keeps falsy ones', () => {
    expect(defined({ a: undefined, b: 0, c: false, d: '' })).toEqual({ b: 0, c: false, d: '' });
    expect(defined(undefined)).toEqual({});
  });
});
