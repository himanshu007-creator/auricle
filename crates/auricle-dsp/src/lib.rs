//! Allocation-free audio plumbing shared by the worklets and the native harness:
//! a polyphase resampler, a ring buffer and a fixed-size frame chunker.
//! Every struct allocates once in `new` and never inside the hot path.

/// Polyphase windowed-sinc resampler between two integer sample rates.
/// The ratio is reduced to `up/down`; one FIR phase is stored per `up` step.
pub struct Resampler {
    up: usize,
    down: usize,
    taps_per_phase: usize,
    /// `up` phases, each `taps_per_phase` coefficients, stored contiguously.
    phases: Vec<f32>,
    /// Most recent input samples, oldest first; always `taps_per_phase` long.
    history: Vec<f32>,
    /// Position of the next output sample in units of 1/up input samples.
    phase_pos: usize,
}

impl Resampler {
    /// `taps` is the FIR length at the lower of the two rates (quality knob; 24 is clean for speech).
    pub fn new(in_rate: u32, out_rate: u32, taps: usize) -> Self {
        let g = gcd(in_rate, out_rate);
        let up = (out_rate / g) as usize;
        let down = (in_rate / g) as usize;
        // The FIR lives at the common rate in_rate*up. Cutoff is half the lower
        // rate; the kernel spans `taps` samples of the lower rate.
        let cutoff = 0.5 / up.max(down) as f64;
        let taps_per_phase = (taps * up.max(down)).div_ceil(up);
        let total = taps_per_phase * up;
        let center = (total as f64 - 1.0) / 2.0;
        // Hann-windowed lowpass, scaled by `up` to undo the zero-stuffing gain loss.
        let kernel: Vec<f32> = (0..total)
            .map(|i| {
                let n = i as f64 - center;
                let lowpass = if n == 0.0 { 2.0 * cutoff } else { (2.0 * std::f64::consts::PI * cutoff * n).sin() / (std::f64::consts::PI * n) };
                let window = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (total as f64 - 1.0)).cos();
                (up as f64 * lowpass * window) as f32
            })
            .collect();
        // Phase p uses kernel taps p, p+up, p+2up, ... reversed so the dot product runs over history oldest-first.
        let mut phases = vec![0.0; total];
        for p in 0..up {
            for t in 0..taps_per_phase {
                phases[p * taps_per_phase + t] = kernel[(taps_per_phase - 1 - t) * up + p];
            }
        }
        Self { up, down, taps_per_phase, phases, history: vec![0.0; taps_per_phase], phase_pos: 0 }
    }

    /// Upper bound on outputs produced for `n` inputs; size `out` with this.
    pub fn max_output_len(&self, n: usize) -> usize {
        (n * self.up) / self.down + 2
    }

    /// Feed `input`, write resampled samples to `out`, return how many were written.
    pub fn process(&mut self, input: &[f32], out: &mut [f32]) -> usize {
        let mut written = 0;
        for &x in input {
            self.history.copy_within(1.., 0);
            let last = self.taps_per_phase - 1;
            self.history[last] = x;
            // Each input advances `up` sub-steps; emit an output for every `down` sub-steps crossed.
            while self.phase_pos < self.up {
                let p = self.phase_pos;
                let coeffs = &self.phases[p * self.taps_per_phase..(p + 1) * self.taps_per_phase];
                let mut acc = 0.0f32;
                for (h, c) in self.history.iter().zip(coeffs) {
                    acc += h * c;
                }
                if written < out.len() {
                    out[written] = acc;
                    written += 1;
                }
                self.phase_pos += self.down;
            }
            self.phase_pos -= self.up;
        }
        written
    }
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Single-producer single-consumer ring of f32 samples. Overwrites nothing:
/// `write` returns how many samples fit.
pub struct RingBuffer {
    buf: Vec<f32>,
    read: usize,
    write: usize,
    len: usize,
}

impl RingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self { buf: vec![0.0; capacity], read: 0, write: 0, len: 0 }
    }
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    pub fn capacity(&self) -> usize { self.buf.len() }
    pub fn clear(&mut self) { self.read = 0; self.write = 0; self.len = 0; }

    pub fn write(&mut self, data: &[f32]) -> usize {
        let n = data.len().min(self.buf.len() - self.len);
        for &x in &data[..n] {
            self.buf[self.write] = x;
            self.write = (self.write + 1) % self.buf.len();
        }
        self.len += n;
        n
    }

    /// Pops up to `out.len()` samples; the rest of `out` is zero-filled. Returns samples popped.
    pub fn read(&mut self, out: &mut [f32]) -> usize {
        let n = out.len().min(self.len);
        for slot in out.iter_mut().take(n) {
            *slot = self.buf[self.read];
            self.read = (self.read + 1) % self.buf.len();
        }
        out[n..].fill(0.0);
        self.len -= n;
        n
    }
}

/// Collects arbitrary-length input into fixed-size frames.
pub struct FrameChunker {
    frame: Vec<f32>,
    filled: usize,
}

impl FrameChunker {
    pub fn new(frame_len: usize) -> Self {
        Self { frame: vec![0.0; frame_len], filled: 0 }
    }

    /// Pushes samples and calls `on_frame` once per completed frame.
    pub fn push(&mut self, input: &[f32], mut on_frame: impl FnMut(&[f32])) {
        for &x in input {
            self.frame[self.filled] = x;
            self.filled += 1;
            if self.filled == self.frame.len() {
                on_frame(&self.frame);
                self.filled = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resampler_preserves_a_tone() {
        // 48 kHz -> 16 kHz: a 440 Hz tone must come out at 440 Hz with near-unity amplitude.
        let mut rs = Resampler::new(48000, 16000, 24);
        let input: Vec<f32> = (0..48000).map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin()).collect();
        let mut out = vec![0.0; rs.max_output_len(input.len())];
        let n = rs.process(&input, &mut out);
        assert!((n as i64 - 16000).abs() <= 2, "got {n} samples");
        let peak = out[8000..n].iter().fold(0.0f32, |m, x| m.max(x.abs()));
        assert!((peak - 1.0).abs() < 0.05, "peak {peak}");
        let zero_crossings = out[8000..n].windows(2).filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0)).count();
        let hz = zero_crossings as f32 / 2.0 / ((n - 8000) as f32 / 16000.0);
        assert!((hz - 440.0).abs() < 5.0, "{hz} Hz");
    }

    #[test]
    fn resampler_handles_44100() {
        let mut rs = Resampler::new(44100, 16000, 24);
        let input = vec![0.5f32; 44100];
        let mut out = vec![0.0; rs.max_output_len(input.len())];
        let n = rs.process(&input, &mut out);
        assert!((n as i64 - 16000).abs() <= 2, "got {n}");
        assert!((out[n - 1] - 0.5).abs() < 0.01, "dc level {}", out[n - 1]);
    }

    #[test]
    fn ring_and_chunker() {
        let mut ring = RingBuffer::new(4);
        assert_eq!(ring.write(&[1.0, 2.0, 3.0, 4.0, 5.0]), 4);
        let mut out = [0.0; 3];
        assert_eq!(ring.read(&mut out), 3);
        assert_eq!(out, [1.0, 2.0, 3.0]);
        assert_eq!(ring.write(&[6.0, 7.0]), 2);
        let mut out = [0.0; 4];
        assert_eq!(ring.read(&mut out), 3);
        assert_eq!(out, [4.0, 6.0, 7.0, 0.0]);

        let mut chunker = FrameChunker::new(3);
        let mut frames = Vec::new();
        chunker.push(&[1.0, 2.0, 3.0, 4.0], |f| frames.push(f.to_vec()));
        chunker.push(&[5.0, 6.0], |f| frames.push(f.to_vec()));
        assert_eq!(frames, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    }
}
