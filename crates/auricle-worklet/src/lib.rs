//! C-ABI entry points for the two AudioWorklets. No wasm-bindgen: worklet
//! scopes lack the browser APIs its glue expects, so the JS side calls these
//! exports directly and reads/writes sample buffers through the exported memory.
//!
//! Capture: device-rate input -> 16 kHz -> 512-sample frames.
//! Playback: agent audio at any rate -> ring -> device rate, with ducking,
//! fade-out on hard stop and a played-sample counter for heard-text accounting.

use auricle_dsp::{FrameChunker, Resampler, RingBuffer};

const VAD_RATE: u32 = 16_000;
const FRAME_LEN: usize = 512;
const INPUT_CAPACITY: usize = 4096;
const MAX_READY_FRAMES: usize = 8;

pub struct Capture {
    resampler: Resampler,
    chunker: FrameChunker,
    input: Vec<f32>,
    resampled: Vec<f32>,
    frames: Vec<f32>,
    ready: usize,
}

#[no_mangle]
pub extern "C" fn capture_new(in_rate: u32) -> *mut Capture {
    let resampler = Resampler::new(in_rate, VAD_RATE, 24);
    let resampled = vec![0.0; resampler.max_output_len(INPUT_CAPACITY)];
    Box::into_raw(Box::new(Capture {
        resampler,
        chunker: FrameChunker::new(FRAME_LEN),
        input: vec![0.0; INPUT_CAPACITY],
        resampled,
        frames: vec![0.0; FRAME_LEN * MAX_READY_FRAMES],
        ready: 0,
    }))
}

#[no_mangle]
pub extern "C" fn capture_input_ptr(c: *mut Capture) -> *mut f32 {
    unsafe { (*c).input.as_mut_ptr() }
}

#[no_mangle]
pub extern "C" fn capture_input_capacity() -> u32 {
    INPUT_CAPACITY as u32
}

/// Consumes `n` samples from the input buffer; returns the number of complete
/// 16 kHz frames now waiting at `capture_frames_ptr` (each FRAME_LEN long).
#[no_mangle]
pub extern "C" fn capture_push(c: *mut Capture, n: u32) -> u32 {
    let c = unsafe { &mut *c };
    let n = (n as usize).min(INPUT_CAPACITY);
    c.ready = 0;
    let written = c.resampler.process(&c.input[..n], &mut c.resampled);
    let (frames, ready) = (&mut c.frames, &mut c.ready);
    c.chunker.push(&c.resampled[..written], |frame| {
        if *ready < MAX_READY_FRAMES {
            frames[*ready * FRAME_LEN..(*ready + 1) * FRAME_LEN].copy_from_slice(frame);
            *ready += 1;
        }
    });
    c.ready as u32
}

#[no_mangle]
pub extern "C" fn capture_frames_ptr(c: *mut Capture) -> *const f32 {
    unsafe { (*c).frames.as_ptr() }
}

#[no_mangle]
pub extern "C" fn capture_frame_len() -> u32 {
    FRAME_LEN as u32
}

pub struct Playback {
    out_rate: u32,
    in_rate: u32,
    resampler: Resampler,
    ring: RingBuffer,
    input: Vec<f32>,
    output: Vec<f32>,
    /// Resampler outputs that did not fit in the last render call.
    pending: Vec<f32>,
    gain: f32,
    target_gain: f32,
    gain_step: f32,
    flush_when_silent: bool,
    flushed: bool,
    played: u64,
}

#[no_mangle]
pub extern "C" fn playback_new(out_rate: u32, in_rate: u32, capacity_seconds: u32) -> *mut Playback {
    Box::into_raw(Box::new(Playback {
        out_rate,
        in_rate,
        resampler: Resampler::new(in_rate, out_rate, 24),
        ring: RingBuffer::new((in_rate * capacity_seconds) as usize),
        input: vec![0.0; INPUT_CAPACITY],
        output: vec![0.0; INPUT_CAPACITY],
        pending: Vec::with_capacity(16),
        gain: 1.0,
        target_gain: 1.0,
        gain_step: 1.0,
        flush_when_silent: false,
        flushed: false,
        played: 0,
    }))
}

/// Changes the rate of incoming audio. Only applied when nothing is queued.
#[no_mangle]
pub extern "C" fn playback_set_input_rate(p: *mut Playback, in_rate: u32) -> bool {
    let p = unsafe { &mut *p };
    if p.in_rate == in_rate {
        return true;
    }
    if !p.ring.is_empty() {
        return false;
    }
    p.in_rate = in_rate;
    p.resampler = Resampler::new(in_rate, p.out_rate, 24);
    p.ring = RingBuffer::new((in_rate * (p.ring.capacity() as u32 / p.in_rate.max(1)).max(1)) as usize);
    true
}

#[no_mangle]
pub extern "C" fn playback_input_ptr(p: *mut Playback) -> *mut f32 {
    unsafe { (*p).input.as_mut_ptr() }
}

#[no_mangle]
pub extern "C" fn playback_output_ptr(p: *mut Playback) -> *const f32 {
    unsafe { (*p).output.as_ptr() }
}

/// Queues `n` input-rate samples from the input buffer. Returns how many fit.
#[no_mangle]
pub extern "C" fn playback_write(p: *mut Playback, n: u32) -> u32 {
    let p = unsafe { &mut *p };
    let n = (n as usize).min(INPUT_CAPACITY);
    if p.flush_when_silent {
        return 0; // a hard stop is in progress; drop late chunks
    }
    p.ring.write(&p.input[..n]) as u32
}

/// Renders `n` output-rate samples into the output buffer with the current
/// gain ramp. Returns 1 when any queued audio was consumed, else 0.
#[no_mangle]
pub extern "C" fn playback_render(p: *mut Playback, n: u32) -> u32 {
    let p = unsafe { &mut *p };
    let n = (n as usize).min(INPUT_CAPACITY);
    let mut filled = 0;
    let mut consumed_any = false;

    // Leftovers from the previous call go first.
    while filled < n && !p.pending.is_empty() {
        p.output[filled] = p.pending.remove(0);
        filled += 1;
    }
    let mut sample = [0.0f32; 1];
    let mut scratch = [0.0f32; 16];
    while filled < n && !p.ring.is_empty() {
        p.ring.read(&mut sample);
        p.played += 1;
        consumed_any = true;
        let produced = p.resampler.process(&sample, &mut scratch);
        for &s in &scratch[..produced] {
            if filled < n {
                p.output[filled] = s;
                filled += 1;
            } else {
                p.pending.push(s);
            }
        }
    }
    p.output[filled..n].fill(0.0);

    for s in &mut p.output[..n] {
        if p.gain != p.target_gain {
            p.gain = if p.gain < p.target_gain {
                (p.gain + p.gain_step).min(p.target_gain)
            } else {
                (p.gain - p.gain_step).max(p.target_gain)
            };
        }
        *s *= p.gain;
    }
    if p.flush_when_silent && p.gain == 0.0 {
        p.ring.clear();
        p.pending.clear();
        p.flush_when_silent = false;
        p.flushed = true;
        p.target_gain = 1.0;
        p.gain_step = 1.0;
    }
    consumed_any as u32
}

/// Ramps linearly to `target` over `ramp_ms` (duck: 0.25 over 30 ms; unduck: 1.0).
#[no_mangle]
pub extern "C" fn playback_set_gain(p: *mut Playback, target: f32, ramp_ms: f32) {
    let p = unsafe { &mut *p };
    p.target_gain = target.clamp(0.0, 1.0);
    let ramp_samples = (ramp_ms * p.out_rate as f32 / 1000.0).max(1.0);
    p.gain_step = (p.gain - p.target_gain).abs() / ramp_samples;
}

/// Hard stop: fade over `fade_ms`, then drop everything queued.
#[no_mangle]
pub extern "C" fn playback_stop(p: *mut Playback, fade_ms: f32) {
    playback_set_gain(p, 0.0, fade_ms);
    unsafe { (*p).flush_when_silent = true };
}

/// Returns 1 once after a stop has finished flushing.
#[no_mangle]
pub extern "C" fn playback_take_flushed(p: *mut Playback) -> u32 {
    let p = unsafe { &mut *p };
    std::mem::replace(&mut p.flushed, false) as u32
}

/// Input-rate samples consumed by rendering so far.
#[no_mangle]
pub extern "C" fn playback_played(p: *mut Playback) -> f64 {
    unsafe { (*p).played as f64 }
}

#[no_mangle]
pub extern "C" fn playback_queued(p: *mut Playback) -> u32 {
    unsafe { (*p).ring.len() as u32 }
}

#[no_mangle]
pub extern "C" fn playback_gain(p: *mut Playback) -> f32 {
    unsafe { (*p).gain }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_yields_frames_at_16k() {
        let c = capture_new(48000);
        let input = unsafe { std::slice::from_raw_parts_mut(capture_input_ptr(c), 128) };
        let mut frames = 0;
        for _ in 0..(16000 / 128 * 3) {
            input.fill(0.1);
            frames += capture_push(c, 128);
        }
        assert_eq!(frames, 31); // 1 s of 16 kHz audio = 31 full frames

    }

    #[test]
    fn playback_ducks_stops_and_counts() {
        let p = playback_new(48000, 24000, 10);
        let input = unsafe { std::slice::from_raw_parts_mut(playback_input_ptr(p), 4000) };
        input.fill(0.5);
        for _ in 0..3 {
            assert_eq!(playback_write(p, 4000), 4000); // 0.5 s at 24 kHz
        }
        playback_set_gain(p, 0.25, 30.0);
        for _ in 0..40 {
            playback_render(p, 128); // 40 * 128 / 48000 = 107 ms > ramp
        }
        let out = unsafe { std::slice::from_raw_parts(playback_output_ptr(p), 128) };
        assert!((out[0] - 0.125).abs() < 0.01, "ducked sample {}", out[0]);
        assert_eq!(playback_played(p) as u32, 2560);
        playback_stop(p, 15.0);
        assert_eq!(playback_write(p, 100), 0);
        for _ in 0..10 {
            playback_render(p, 128);
        }
        assert_eq!(playback_take_flushed(p), 1);
        assert_eq!(playback_take_flushed(p), 0);
        assert_eq!(playback_queued(p), 0);
        assert_eq!(playback_render(p, 128), 0);
    }
}
