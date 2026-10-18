//! Auricle Auricle v1: our own foreground-selective detector, trained from scratch in
//! the VAD-train repository (labels bootstrapped from Silero VAD v6 on clean
//! audio, competing-talker mixing). 83,393 parameters, one probability per
//! 16 ms hop.
//!
//! Graph, identical to vadtrain/features.py and model.py: periodic-Hann 512-point
//! power spectrum every 256 samples, 40 HTK mel bands (area-normalised), log,
//! per-band normalisation, two causal Conv1d(k=5)+ReLU over time, a two-layer
//! GRU of 64 units, one linear logit, sigmoid.

use crate::math::{bit_reverse, dot, f32_slice, fft, sigmoid, twiddles};
use crate::{Vad, FRAME_LEN};

/// WEIGHT_LAYOUT (see model/auricle_v1_layout.json): contiguous little-endian f32,
/// mean (40), std (40), conv1 weight (64,40,5) + bias, conv2 weight (64,64,5) + bias,
/// gru weight_ih l0, l1 (192,64), weight_hh l0, l1 (192,64), bias_ih l0, l1, bias_hh l0, l1 (192),
/// out weight (64), out bias (1). GRU gate order r, z, n (PyTorch).
static WEIGHTS: &[u8] = include_bytes!("../model/auricle_v1_16k.f32");

pub const WINDOW: usize = 512;
/// Samples per hop: one probability every 16 ms.
pub const HOP: usize = 256;
const BINS: usize = WINDOW / 2 + 1;
const N_MELS: usize = 40;
const CH: usize = 64;
const K: usize = 5;
const HIDDEN: usize = 64;
const LAYERS: usize = 2;
const F_MIN: f64 = 20.0;
const F_MAX: f64 = 8000.0;
const LOG_FLOOR: f32 = 1e-6;
const SR: f64 = 16_000.0;

pub struct Auricle {
    mean: &'static [f32],
    std: &'static [f32],
    /// (N_MELS, BINS) triangular filters, dense so each band is one dot product.
    filterbank: Vec<f32>,
    /// Conv weights re-laid out as (cout, k, cin) so the inner loop is contiguous.
    conv1_w: Vec<f32>,
    conv1_b: &'static [f32],
    conv2_w: Vec<f32>,
    conv2_b: &'static [f32],
    w_ih: [&'static [f32]; LAYERS],
    w_hh: [&'static [f32]; LAYERS],
    b_ih: [&'static [f32]; LAYERS],
    b_hh: [&'static [f32]; LAYERS],
    out_w: &'static [f32],
    out_b: f32,
    window: [f32; WINDOW],
    twiddle_re: Vec<f32>,
    twiddle_im: Vec<f32>,
    fft_re: [f32; WINDOW],
    fft_im: [f32; WINDOW],
    power: [f32; BINS],
    /// The previous hop's samples: each window is [previous hop, new hop].
    prev: [f32; HOP],
    primed: bool,
    /// Last K normalised mel frames (oldest first); conv1's causal receptive field.
    hist1: [[f32; N_MELS]; K],
    /// Last K conv1 outputs; conv2's receptive field.
    hist2: [[f32; CH]; K],
    h: [[f32; HIDDEN]; LAYERS],
}

fn hz_to_mel(f: f64) -> f64 {
    2595.0 * (1.0 + f / 700.0).log10()
}

fn mel_to_hz(m: f64) -> f64 {
    700.0 * (10f64.powf(m / 2595.0) - 1.0)
}

/// Same construction as vadtrain.features.mel_filterbank, including its bin rounding and area normalisation.
fn mel_filterbank() -> Vec<f32> {
    let (lo_mel, hi_mel) = (hz_to_mel(F_MIN), hz_to_mel(F_MAX));
    let bins: Vec<usize> = (0..N_MELS + 2)
        .map(|i| {
            let mel = lo_mel + (hi_mel - lo_mel) * i as f64 / (N_MELS + 1) as f64;
            ((WINDOW + 1) as f64 * mel_to_hz(mel) / SR).floor() as usize
        })
        .collect();
    let mut fb = vec![0.0f32; N_MELS * BINS];
    for m in 0..N_MELS {
        let lo = bins[m];
        let mut centre = bins[m + 1];
        let mut hi = bins[m + 2];
        hi = hi.max(centre + 1);
        centre = centre.max(lo + 1);
        let row = &mut fb[m * BINS..(m + 1) * BINS];
        for k in lo..centre {
            row[k] = (k - lo) as f32 / (centre - lo) as f32;
        }
        for k in centre..hi {
            row[k] = (hi - k) as f32 / (hi - centre) as f32;
        }
        let sum: f32 = row.iter().sum();
        let norm = sum.max(1e-6);
        row.iter_mut().for_each(|v| *v /= norm);
    }
    fb
}

/// (cout, cin, k) in the export -> (cout, k, cin) for contiguous dot products.
fn relayout(w: &[f32], cout: usize, cin: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; cout * K * cin];
    for o in 0..cout {
        for i in 0..cin {
            for k in 0..K {
                out[(o * K + k) * cin + i] = w[(o * cin + i) * K + k];
            }
        }
    }
    out
}

impl Default for Auricle {
    fn default() -> Self {
        Self::new()
    }
}

impl Auricle {
    pub fn new() -> Self {
        let all = f32_slice(WEIGHTS);
        let mut cursor = 0;
        let mut take = |n: usize| {
            let s = &all[cursor..cursor + n];
            cursor += n;
            s
        };
        let mean = take(N_MELS);
        let std = take(N_MELS);
        let conv1_w = relayout(take(CH * N_MELS * K), CH, N_MELS);
        let conv1_b = take(CH);
        let conv2_w = relayout(take(CH * CH * K), CH, CH);
        let conv2_b = take(CH);
        let w_ih = [take(3 * HIDDEN * CH), take(3 * HIDDEN * HIDDEN)];
        let w_hh = [take(3 * HIDDEN * HIDDEN), take(3 * HIDDEN * HIDDEN)];
        let b_ih = [take(3 * HIDDEN), take(3 * HIDDEN)];
        let b_hh = [take(3 * HIDDEN), take(3 * HIDDEN)];
        let out_w = take(HIDDEN);
        let out_b = take(1)[0];
        debug_assert_eq!(cursor, all.len());

        let mut window = [0.0f32; WINDOW];
        for (n, w) in window.iter_mut().enumerate() {
            *w = (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / WINDOW as f64).cos()) as f32;
        }
        let (twiddle_re, twiddle_im) = twiddles(WINDOW);
        Self {
            mean,
            std,
            filterbank: mel_filterbank(),
            conv1_w,
            conv1_b,
            conv2_w,
            conv2_b,
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            out_w,
            out_b,
            window,
            twiddle_re,
            twiddle_im,
            fft_re: [0.0; WINDOW],
            fft_im: [0.0; WINDOW],
            power: [0.0; BINS],
            prev: [0.0; HOP],
            primed: false,
            hist1: [[0.0; N_MELS]; K],
            hist2: [[0.0; CH]; K],
            h: [[0.0; HIDDEN]; LAYERS],
        }
    }

    /// Stages the first half-window without scoring it, so the first scored
    /// window is samples [0, 512) exactly as in training (no centre padding).
    pub fn prime(&mut self, samples: &[f32; HOP]) {
        self.prev = *samples;
        self.primed = true;
    }

    /// Scores the window [previous hop, `new`] and returns the speech probability.
    pub fn push_hop(&mut self, new: &[f32; HOP]) -> f32 {
        // Windowed 512-point FFT, bit-reversed load.
        for i in 0..WINDOW {
            let x = if i < HOP { self.prev[i] } else { new[i - HOP] };
            let j = bit_reverse(i, 9);
            self.fft_re[j] = x * self.window[i];
            self.fft_im[j] = 0.0;
        }
        self.prev = *new;
        fft::<WINDOW>(&mut self.fft_re, &mut self.fft_im, &self.twiddle_re, &self.twiddle_im);
        for k in 0..BINS {
            self.power[k] = self.fft_re[k] * self.fft_re[k] + self.fft_im[k] * self.fft_im[k];
        }
        // Log-mel, normalised, into the conv1 history.
        self.hist1.rotate_left(1);
        let feat = &mut self.hist1[K - 1];
        for m in 0..N_MELS {
            let mel = dot(&self.filterbank[m * BINS..(m + 1) * BINS], &self.power);
            feat[m] = ((mel + LOG_FLOOR).ln() - self.mean[m]) / self.std[m];
        }
        // Causal conv1 + ReLU over the last K frames, into the conv2 history.
        self.hist2.rotate_left(1);
        let mut out1 = [0.0f32; CH];
        for o in 0..CH {
            let mut acc = self.conv1_b[o];
            for k in 0..K {
                acc += dot(&self.conv1_w[(o * K + k) * N_MELS..(o * K + k + 1) * N_MELS], &self.hist1[k]);
            }
            out1[o] = acc.max(0.0);
        }
        self.hist2[K - 1] = out1;
        let mut x = [0.0f32; CH];
        for o in 0..CH {
            let mut acc = self.conv2_b[o];
            for k in 0..K {
                acc += dot(&self.conv2_w[(o * K + k) * CH..(o * K + k + 1) * CH], &self.hist2[k]);
            }
            x[o] = acc.max(0.0);
        }
        // Two GRU layers, gates r, z, n.
        for l in 0..LAYERS {
            let (w_ih, w_hh, b_ih, b_hh) = (self.w_ih[l], self.w_hh[l], self.b_ih[l], self.b_hh[l]);
            let h = &mut self.h[l];
            let mut gi = [0.0f32; 3 * HIDDEN];
            let mut gh = [0.0f32; 3 * HIDDEN];
            for j in 0..3 * HIDDEN {
                gi[j] = dot(&w_ih[j * CH..(j + 1) * CH], &x) + b_ih[j];
                gh[j] = dot(&w_hh[j * HIDDEN..(j + 1) * HIDDEN], h) + b_hh[j];
            }
            for n in 0..HIDDEN {
                let r = sigmoid(gi[n] + gh[n]);
                let z = sigmoid(gi[HIDDEN + n] + gh[HIDDEN + n]);
                let cand = (gi[2 * HIDDEN + n] + r * gh[2 * HIDDEN + n]).tanh();
                h[n] = (1.0 - z) * cand + z * h[n];
            }
            x = *h;
        }
        sigmoid(dot(self.out_w, &x) + self.out_b)
    }
}

impl Vad for Auricle {
    /// Scores both 16 ms hops of the 32 ms frame and returns the larger: the
    /// frame contains speech if either half does. The first frame primes the
    /// window with its first half so the hop grid matches training.
    fn process(&mut self, frame: &[f32; FRAME_LEN]) -> f32 {
        let (first, second) = frame.split_at(HOP);
        let (first, second): (&[f32; HOP], &[f32; HOP]) = (first.try_into().unwrap(), second.try_into().unwrap());
        if !self.primed {
            self.prime(first);
            return self.push_hop(second);
        }
        let a = self.push_hop(first);
        let b = self.push_hop(second);
        a.max(b)
    }

    fn reset(&mut self) {
        self.prev = [0.0; HOP];
        self.primed = false;
        self.hist1 = [[0.0; N_MELS]; K];
        self.hist2 = [[0.0; CH]; K];
        self.h = [[0.0; HIDDEN]; LAYERS];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_f32(bytes: &[u8]) -> Vec<f32> {
        bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }

    /// Hop by hop against the PyTorch model on the same clip (tools/auricle_fixture.py).
    #[test]
    fn matches_pytorch_per_hop() {
        let input = read_f32(include_bytes!("../model/parity_input.f32"));
        let expected = read_f32(include_bytes!("../model/parity_auricle_expected.f32"));
        let mut vad = Auricle::new();
        vad.prime(input[..HOP].try_into().unwrap());
        let mut max_err = 0.0f32;
        let mut hops = 0;
        for (i, chunk) in input[HOP..].chunks_exact(HOP).enumerate() {
            let p = vad.push_hop(chunk.try_into().unwrap());
            max_err = max_err.max((p - expected[i]).abs());
            hops += 1;
        }
        assert_eq!(hops, expected.len());
        assert!(max_err < 1e-4, "max abs diff vs PyTorch {max_err}");
    }

    /// The frame path yields the same hop sequence (minus the max) as the hop path.
    #[test]
    fn frame_path_matches_hop_path() {
        let input = read_f32(include_bytes!("../model/parity_input.f32"));
        let mut by_hop = Auricle::new();
        by_hop.prime(input[..HOP].try_into().unwrap());
        let hop_probs: Vec<f32> = input[HOP..].chunks_exact(HOP).map(|c| by_hop.push_hop(c.try_into().unwrap())).collect();
        let mut by_frame = Auricle::new();
        for (f, frame) in input.chunks_exact(FRAME_LEN).enumerate() {
            let p = by_frame.process(frame.try_into().unwrap());
            let want = if f == 0 { hop_probs[0] } else { hop_probs[2 * f - 1].max(hop_probs[2 * f]) };
            assert!((p - want).abs() < 1e-6, "frame {f}: {p} vs {want}");
        }
    }
}
