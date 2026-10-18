//! Silero VAD v6, 16 kHz branch, as plain Rust. The weights are exported from
//! the official ONNX file by tools/export_silero.py (MIT, see model/LICENSE.silero).
//!
//! Graph: 64 context samples + 512 new samples, reflect-padded by 64, Hann-windowed
//! 256-point FFT at hop 128 (4 frames), magnitude of 129 bins, four Conv1d+ReLU
//! blocks (strides 1,2,2,1), one LSTM cell, ReLU, 1x1 conv, sigmoid.

use crate::math::{dot, f32_slice, fft, sigmoid};
use crate::{Vad, FRAME_LEN};

/// WEIGHT_LAYOUT: contiguous little-endian f32 in this order and shape.
/// encoder.{0..3}: weight (cout, cin, 3) then bias (cout) with
///   (cout, cin) = (128,129), (64,128), (64,64), (128,64)
/// decoder.rnn: weight_ih (512,128), weight_hh (512,128), bias_ih (512), bias_hh (512)
///   gate order i, f, g, o (PyTorch LSTMCell)
/// decoder.decoder.2: weight (1,128,1), bias (1)
static WEIGHTS: &[u8] = include_bytes!("../model/silero_v6_16k.f32");

const CONTEXT: usize = 64;
const FFT: usize = 256;
const HOP: usize = 128;
const BINS: usize = 129;
const STFT_FRAMES: usize = 4; // (576 + 64 - 256) / 128 + 1
const HIDDEN: usize = 128;

struct Conv {
    cout: usize,
    cin: usize,
    stride: usize,
    /// Re-laid out as (cout, k, cin) so the inner loop runs over contiguous channels.
    weight: Vec<f32>,
    bias: &'static [f32],
}

pub struct Silero {
    convs: [Conv; 4],
    w_ih: &'static [f32],
    w_hh: &'static [f32],
    b_ih: &'static [f32],
    b_hh: &'static [f32],
    out_w: &'static [f32],
    out_b: f32,
    /// Periodic Hann window, 256 taps.
    window: [f32; FFT],
    /// Twiddle factors for the 256-point FFT: cos and -sin of 2*pi*k/256.
    twiddle_re: [f32; FFT / 2],
    twiddle_im: [f32; FFT / 2],
    /// Scratch for one FFT frame.
    fft_re: [f32; FFT],
    fft_im: [f32; FFT],
    context: [f32; CONTEXT],
    h: [f32; HIDDEN],
    c: [f32; HIDDEN],
    // Scratch buffers so process() never allocates.
    padded: [f32; CONTEXT + FRAME_LEN + CONTEXT],
    mag: Vec<f32>,
    act_a: Vec<f32>,
    act_b: Vec<f32>,
}

impl Default for Silero {
    fn default() -> Self {
        Self::new()
    }
}

impl Silero {
    pub fn new() -> Self {
        let all = f32_slice(WEIGHTS);
        let mut cursor = 0;
        let mut take = |n: usize| {
            let s = &all[cursor..cursor + n];
            cursor += n;
            s
        };
        let dims = [(128, 129, 1), (64, 128, 2), (64, 64, 2), (128, 64, 1)];
        let convs = dims.map(|(cout, cin, stride)| {
            let w = take(cout * cin * 3);
            let mut weight = vec![0.0f32; cout * 3 * cin];
            for o in 0..cout {
                for i in 0..cin {
                    for k in 0..3 {
                        weight[(o * 3 + k) * cin + i] = w[(o * cin + i) * 3 + k];
                    }
                }
            }
            Conv { cout, cin, stride, weight, bias: take(cout) }
        });
        let w_ih = take(4 * HIDDEN * HIDDEN);
        let w_hh = take(4 * HIDDEN * HIDDEN);
        let b_ih = take(4 * HIDDEN);
        let b_hh = take(4 * HIDDEN);
        let out_w = take(HIDDEN);
        let out_b = take(1)[0];
        debug_assert_eq!(cursor, all.len());

        let mut window = [0.0f32; FFT];
        for (n, w) in window.iter_mut().enumerate() {
            *w = (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / FFT as f64).cos()) as f32;
        }
        let mut twiddle_re = [0.0f32; FFT / 2];
        let mut twiddle_im = [0.0f32; FFT / 2];
        for k in 0..FFT / 2 {
            let angle = 2.0 * std::f64::consts::PI * k as f64 / FFT as f64;
            twiddle_re[k] = angle.cos() as f32;
            twiddle_im[k] = -(angle.sin() as f32);
        }
        Self {
            convs,
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            out_w,
            out_b,
            window,
            twiddle_re,
            twiddle_im,
            fft_re: [0.0; FFT],
            fft_im: [0.0; FFT],
            context: [0.0; CONTEXT],
            h: [0.0; HIDDEN],
            c: [0.0; HIDDEN],
            padded: [0.0; CONTEXT + FRAME_LEN + CONTEXT],
            mag: vec![0.0; BINS * STFT_FRAMES],
            act_a: vec![0.0; 128 * STFT_FRAMES],
            act_b: vec![0.0; 128 * STFT_FRAMES],
        }
    }

    /// Windowed 256-point FFT per hop; keeps the magnitude of bins 0..=128.
    /// Equivalent to the model's basis matmul (a Hann-windowed DFT) at a
    /// fraction of the multiplies.
    fn stft_magnitude(&mut self) {
        for t in 0..STFT_FRAMES {
            let frame = &self.padded[t * HOP..t * HOP + FFT];
            // Bit-reversed load with the window applied.
            for i in 0..FFT {
                let j = (i as u8).reverse_bits() as usize;
                self.fft_re[j] = frame[i] * self.window[i];
                self.fft_im[j] = 0.0;
            }
            fft::<FFT>(&mut self.fft_re, &mut self.fft_im, &self.twiddle_re, &self.twiddle_im);
            for k in 0..BINS {
                self.mag[t * BINS + k] = (self.fft_re[k] * self.fft_re[k] + self.fft_im[k] * self.fft_im[k]).sqrt();
            }
        }
    }
}

/// Conv1d with kernel 3, zero padding 1, followed by ReLU.
/// `input` is (t_in, cin) row-major; writes (t_out, cout) into `output` and returns t_out.
fn conv3_relu(conv: &Conv, input: &[f32], t_in: usize, output: &mut [f32]) -> usize {
    let t_out = (t_in + 2 - 3) / conv.stride + 1;
    let cin = conv.cin;
    for t in 0..t_out {
        for o in 0..conv.cout {
            let mut acc = conv.bias[o];
            for k in 0..3 {
                let Some(idx) = (t * conv.stride + k).checked_sub(1) else { continue };
                if idx >= t_in {
                    continue;
                }
                let w = &conv.weight[(o * 3 + k) * cin..(o * 3 + k + 1) * cin];
                acc += dot(w, &input[idx * cin..(idx + 1) * cin]);
            }
            output[t * conv.cout + o] = acc.max(0.0);
        }
    }
    t_out
}

impl Silero {
    /// Runs the network on the 64 context + 512 new samples already staged in
    /// `padded[..576]`, adding the reflect padding first.
    fn forward(&mut self) -> f32 {
        let end = CONTEXT + FRAME_LEN;
        for i in 0..CONTEXT {
            self.padded[end + i] = self.padded[end - 2 - i];
        }
        self.stft_magnitude();

        let mut t = STFT_FRAMES;
        t = conv3_relu(&self.convs[0], &self.mag, t, &mut self.act_a);
        t = conv3_relu(&self.convs[1], &self.act_a, t, &mut self.act_b);
        t = conv3_relu(&self.convs[2], &self.act_b, t, &mut self.act_a);
        t = conv3_relu(&self.convs[3], &self.act_a, t, &mut self.act_b);
        debug_assert_eq!(t, 1);
        let feat = &self.act_b[..HIDDEN];

        // LSTM cell, gates i, f, g, o.
        let mut gates = [0.0f32; 4 * HIDDEN];
        for (j, g) in gates.iter_mut().enumerate() {
            *g = dot(&self.w_ih[j * HIDDEN..(j + 1) * HIDDEN], feat)
                + dot(&self.w_hh[j * HIDDEN..(j + 1) * HIDDEN], &self.h)
                + self.b_ih[j]
                + self.b_hh[j];
        }
        for n in 0..HIDDEN {
            let i = sigmoid(gates[n]);
            let f = sigmoid(gates[HIDDEN + n]);
            let g = gates[2 * HIDDEN + n].tanh();
            let o = sigmoid(gates[3 * HIDDEN + n]);
            self.c[n] = f * self.c[n] + i * g;
            self.h[n] = o * self.c[n].tanh();
        }

        let logit = self.h.iter().zip(self.out_w).map(|(h, w)| h.max(0.0) * w).sum::<f32>() + self.out_b;
        sigmoid(logit)
    }

}

impl Vad for Silero {
    fn process(&mut self, frame: &[f32; FRAME_LEN]) -> f32 {
        // Context, then the new samples; `forward` adds the reflect padding.
        self.padded[..CONTEXT].copy_from_slice(&self.context);
        self.padded[CONTEXT..CONTEXT + FRAME_LEN].copy_from_slice(frame);
        self.context.copy_from_slice(&frame[FRAME_LEN - CONTEXT..]);
        self.forward()
    }

    fn reset(&mut self) {
        self.context = [0.0; CONTEXT];
        self.padded = [0.0; CONTEXT + FRAME_LEN + CONTEXT];
        self.h = [0.0; HIDDEN];
        self.c = [0.0; HIDDEN];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_f32(bytes: &[u8]) -> Vec<f32> {
        bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }

    #[test]
    fn matches_onnxruntime_on_speech() {
        let input = read_f32(include_bytes!("../model/parity_input.f32"));
        let expected = read_f32(include_bytes!("../model/parity_expected.f32"));
        let mut vad = Silero::new();
        let mut max_err = 0.0f32;
        for (i, frame) in input.chunks_exact(FRAME_LEN).enumerate() {
            let p = vad.process(frame.try_into().unwrap());
            max_err = max_err.max((p - expected[i]).abs());
        }
        assert!(max_err < 1e-4, "max abs diff vs onnxruntime {max_err}");
        assert!(expected.iter().any(|&p| p > 0.9) && expected.iter().any(|&p| p < 0.1));
    }
}
