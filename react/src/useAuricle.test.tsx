import { act, renderHook } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { Emitter } from '../../web/src/emitter.ts';

// Replace the Auricle class with a fake emitter so the hook's wiring is tested without audio.
const fake = Object.assign(new Emitter<Record<string, unknown>>(), {
  state: 'idle',
  start: vi.fn(async () => undefined),
  stop: vi.fn(),
  interrupt: vi.fn(),
});
vi.mock('auricle', () => ({ Auricle: { create: vi.fn(async () => fake) } }));

const { useAuricle } = await import('./useAuricle.ts');

describe('useAuricle', () => {
  it('mirrors events into state', async () => {
    const { result } = renderHook(() => useAuricle());
    await act(() => result.current.start());
    act(() => {
      fake.emit('state', 'listening');
      fake.emit('frame', { p: 0.8, tMs: 32 });
      fake.emit('turn', { verbatim: 'um hi', clean: 'hi', mask: [false, true], cleanup: 'rules', metrics: {} });
      fake.emit('agentText', { delta: 'Hello', text: 'Hello' });
      fake.emit('interrupted', { heardText: 'Hel', generatedText: 'Hello there', stopLatencyMs: 90 });
    });
    expect(result.current.state).toBe('listening');
    expect(result.current.probability).toBe(0.8);
    expect(result.current.turns[0].clean).toBe('hi');
    expect(result.current.agentText).toBe('Hello');
    expect(result.current.heardText).toBe('Hel');
    act(() => result.current.stop());
    expect(fake.stop).toHaveBeenCalled();
    expect(result.current.state).toBe('idle');
  });
});
