import { describe, expect, it } from 'vitest';
import { Emitter } from './emitter.ts';
import { SessionTracker, formatStats } from './stats.ts';
import type { AuricleEvents } from './auricle.ts';

describe('SessionTracker', () => {
  it('counts what left the device and what an interruption avoided', () => {
    const src = new Emitter<AuricleEvents>();
    const t = new SessionTracker(src);
    src.emit('state', 'listening');
    src.emit('speechEnd', { samples: new Float32Array(0), startMs: 1000, endMs: 3500 });
    src.emit('speechEnd', { samples: new Float32Array(0), startMs: 9000, endMs: 10000 });
    src.emit('event', { name: 'misfire', tMs: 12000 });
    src.emit('turn', { verbatim: '', clean: '', mask: [], cleanup: 'rules', metrics: {} });
    src.emit('agentDone', { text: 'twelve chars' });
    src.emit('interrupted', { heardText: 'heard this', generatedText: 'heard this but not the rest', stopLatencyMs: 40 });
    src.emit('interrupted', { heardText: '', generatedText: 'nothing', stopLatencyMs: 20 });
    src.emit('metrics', { endToFirstAudioMs: 900 });
    const s = t.snapshot();
    expect(s.speechMs).toBe(3500);
    expect(s.segments).toBe(2);
    expect(s.misfires).toBe(1);
    expect(s.turns).toBe(1);
    expect(s.ttsCharsGenerated).toBe('twelve chars'.length + 'heard this'.length);
    expect(s.ttsCharsAvoided).toBe(' but not the rest'.length + 'nothing'.length);
    expect(s.interruptions).toBe(2);
    expect(s.stopLatencyMedianMs).toBe(40);
    expect(s.responseMedianMs).toBe(900);
    expect(s.sttShare).toBeGreaterThan(0);
    expect(s.sttShare).toBeLessThanOrEqual(1);
    expect(formatStats(s)).toMatch(/sent to STT 0:03/);
    t.reset();
    expect(t.snapshot().segments).toBe(0);
  });
});
