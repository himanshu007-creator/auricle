"""Export Silero VAD v6 (16 kHz branch) weights to a flat little-endian f32 blob
and write a parity fixture so the Rust forward pass can be checked against
onnxruntime.

Usage:
    python tools/export_silero.py path/to/silero_vad.onnx crates/auricle-vad/model [fixture.wav]

The optional 16 kHz mono 16-bit WAV is used for the parity fixture (first 100
frames); without it a synthetic signal is used.

The blob layout is documented in crates/auricle-vad/src/silero.rs (WEIGHT_LAYOUT).
The STFT basis is not exported: it is a DFT basis under a periodic Hann window
that Rust computes at start-up (verified below against the tensor in the model).
"""
import struct
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

PREFIX = "If_0_then_branch__Inline_0__"
WEIGHT_ORDER = [
    "encoder.0.reparam_conv.weight", "encoder.0.reparam_conv.bias",
    "encoder.1.reparam_conv.weight", "encoder.1.reparam_conv.bias",
    "encoder.2.reparam_conv.weight", "encoder.2.reparam_conv.bias",
    "encoder.3.reparam_conv.weight", "encoder.3.reparam_conv.bias",
    "decoder.rnn.weight_ih", "decoder.rnn.weight_hh",
    "decoder.rnn.bias_ih", "decoder.rnn.bias_hh",
    "decoder.decoder.2.weight", "decoder.decoder.2.bias",
]
FRAME = 512
CONTEXT = 64


def load_weights(model):
    if_node = next(n for n in model.graph.node if n.op_type == "If")
    branch = next(helper.get_attribute_value(a) for a in if_node.attribute if a.name == "then_branch")
    out = {}
    for n in branch.node:
        if n.op_type == "Constant" and n.output[0].startswith(PREFIX):
            name = n.output[0][len(PREFIX):]
            if "/" not in name:
                out[name] = numpy_helper.to_array(helper.get_attribute_value(n.attribute[0]))
    return out


def dft_basis():
    basis = np.fft.fft(np.eye(256))[:129]
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(256) / 256)  # periodic Hann
    return (np.vstack([basis.real, basis.imag]) * window).astype(np.float32)


def conv1d(x, w, b, stride, pad):
    """x: (cin, t), w: (cout, cin, k). Zero padding on both sides."""
    cin, t = x.shape
    cout, _, k = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad)))
    tout = (t + 2 * pad - k) // stride + 1
    y = np.empty((cout, tout), np.float32)
    for i in range(tout):
        y[:, i] = np.tensordot(w, xp[:, i * stride:i * stride + k], axes=([1, 2], [0, 1])) + b
    return y


def reference_forward(w, chunk, h, c):
    """numpy mirror of the ONNX graph; chunk is 576 samples (64 context + 512 new)."""
    x = np.concatenate([chunk, chunk[-2:-66:-1]])  # reflect pad 64 on the right
    frames = np.stack([x[i * 128:i * 128 + 256] for i in range(4)])  # (4, 256)
    spec = dft_basis() @ frames.T  # (258, 4)
    mag = np.sqrt(spec[:129] ** 2 + spec[129:] ** 2)
    y = mag
    for i, stride in enumerate([1, 2, 2, 1]):
        y = np.maximum(conv1d(y, w[f"encoder.{i}.reparam_conv.weight"], w[f"encoder.{i}.reparam_conv.bias"], stride, 1), 0)
    feat = y[:, 0]
    gates = w["decoder.rnn.weight_ih"] @ feat + w["decoder.rnn.bias_ih"] + w["decoder.rnn.weight_hh"] @ h + w["decoder.rnn.bias_hh"]
    i, f, g, o = np.split(gates, 4)
    sig = lambda v: 1 / (1 + np.exp(-v))
    c2 = sig(f) * c + sig(i) * np.tanh(g)
    h2 = sig(o) * np.tanh(c2)
    logit = w["decoder.decoder.2.weight"][0, :, 0] @ np.maximum(h2, 0) + w["decoder.decoder.2.bias"][0]
    return float(sig(logit)), h2, c2


def test_signal(n_frames, seed=7):
    """Noise bursts and a vowel-like harmonic tone so probabilities span 0..1."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames * FRAME) / 16000
    tone = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 12)) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
    gate = (np.sin(2 * np.pi * 0.4 * t) > 0).astype(np.float32)
    sig = 0.2 * tone * gate + 0.01 * rng.standard_normal(t.size) + 0.3 * rng.standard_normal(t.size) * (np.sin(2 * np.pi * 0.13 * t) > 0.8)
    return np.clip(sig, -1, 1).astype(np.float32)


def read_wav_16k_mono(path):
    import wave
    with wave.open(str(path)) as f:
        assert f.getframerate() == 16000 and f.getnchannels() == 1 and f.getsampwidth() == 2
        pcm = np.frombuffer(f.readframes(f.getnframes()), np.int16)
    return pcm.astype(np.float32) / 32768


def main(onnx_path, out_dir, wav_path=None):
    model = onnx.load(onnx_path)
    w = load_weights(model)
    basis_in_model = w["stft.forward_basis_buffer"][:, 0, :]
    assert np.allclose(basis_in_model, dft_basis(), atol=1e-5), "STFT basis is not a Hann-windowed DFT basis"

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    signal = read_wav_16k_mono(wav_path)[: 100 * FRAME] if wav_path else test_signal(60)
    n_frames = signal.size // FRAME
    state = np.zeros((2, 1, 128), np.float32)
    context = np.zeros(CONTEXT, np.float32)
    h = np.zeros(128, np.float32)
    c = np.zeros(128, np.float32)
    expected, max_err = [], 0.0
    for i in range(n_frames):
        frame = signal[i * FRAME:(i + 1) * FRAME]
        chunk = np.concatenate([context, frame])
        p_ort, state = sess.run(None, {"input": chunk[None], "state": state, "sr": np.array(16000, np.int64)})
        p_ref, h, c = reference_forward(w, chunk, h, c)
        max_err = max(max_err, abs(float(p_ort[0, 0]) - p_ref))
        expected.append(float(p_ort[0, 0]))
        context = frame[-CONTEXT:]
    print(f"numpy reference vs onnxruntime: max abs diff {max_err:.2e} over {n_frames} frames")
    assert max_err < 1e-4

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    blob = b"".join(np.ascontiguousarray(w[k], dtype=np.float32).tobytes() for k in WEIGHT_ORDER)
    (out / "silero_v6_16k.f32").write_bytes(blob)
    (out / "parity_input.f32").write_bytes(signal.tobytes())
    (out / "parity_expected.f32").write_bytes(np.array(expected, np.float32).tobytes())
    print(f"wrote {len(blob)} bytes of weights, fixture of {n_frames} frames, probs min {min(expected):.3f} max {max(expected):.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
