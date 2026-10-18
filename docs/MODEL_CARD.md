<p align="center"><img src="https://raw.githubusercontent.com/himanshu007-creator/auricle/main/demo/public/brand/logo.svg" alt="Auricle" height="64" /></p>

---
license: mit
language: [en, es, de, ar, hi, zh, ja]
tags: [voice-activity-detection, vad, speech, audio, onnx, rust, webassembly, voice-agent]
library_name: auricle
pipeline_tag: voice-activity-detection
---

# Auricle v1

A voice activity detector trained from scratch to keep listening to the user when somebody else in the room is talking. 83,393 parameters, 334 KB of float32 weights, one decision every 16 ms, causal, streaming. Ships inside the `auricle` npm package as a hand-written Rust forward pass; this card holds the weights, the ONNX export and the numbers.

## Files

| File | What |
|---|---|
| `vad.f32` | the weights as one little-endian float32 blob, in the order given by `layout.json` |
| `layout.json` | tensor names and shapes, sample rate 16000, window 512, hop 256, 40 mel bands, GRU gate order r, z, n, recommended threshold |
| `vad.onnx` | the same network for onnxruntime; input `feats` (1, T, 40) log-mel, outputs `logits` (1, T) and the GRU state |

## Architecture

Periodic-Hann 512-point power spectrum every 256 samples at 16 kHz, 40 HTK mel bands (area-normalised), log, per-band normalisation, two causal Conv1d (kernel 5, 64 channels) with ReLU, a two-layer GRU of 64 units, one linear logit, sigmoid. The feature code is `vadtrain/features.py` in the training repository; the Rust port is `crates/auricle-vad/src/auricle.rs` in the Auricle repository, with a parity test against PyTorch on the same clip.

## Training data and recipe

961 hours of LibriSpeech (train-clean-100, train-clean-360, train-other-500) labelled once on the clean audio by Silero VAD v6 with a -60 dBFS energy floor. The model never sees the clean audio. Each 5-second training clip is one to three utterances of one speaker with realistic gaps, then: a room impulse response (RIRS_NOISES, 50%), MUSAN noise or music at -5 to 25 dB SNR (85%), a competing talker from a different speaker at 3 to 25 dB below the user (45%), MUSAN babble as the far background (25% of those), VocalSound laughs, coughs, sighs and sniffs dropped into gaps at the user's level with label 0 (20%), background-only clips with no foreground at all (12%), telephone bandlimiting (20%), clipping (10%), random gain -30 to 0 dB. Labels stay those of the clean foreground, so the only way to score well is to learn which voice is the user.

AdamW, lr 2e-3 with 500 warm-up steps and cosine decay, weight decay 1e-2, gradient clip 1, EMA 0.999 on the weights, binary cross-entropy with a 0.5 s burn-in mask and half weight within two frames of a label change. 20,000 steps of 128 clips on a six-core Ryzen 5 4600H CPU, 9.7 hours. Dev selection by frame F1 on held-out dev-clean speakers through the same mixer.

Code, config and logs: https://github.com/himanshu007-creator/VAD-train (`runs/v1`).

## Evaluation

Silero VAD v6 is scored on identical audio in every table.

### Human-labelled real audio

TEN VAD's 30 annotated recordings (LibriSpeech, GigaSpeech and DNS Challenge material), frame F1 at threshold 0.5:

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Auricle v1 | 92.0% | 93.4% | 92.7% |
| Silero v6 | 93.6% | 93.5% | 93.6% |

### Competing talker

500 generated clips from held-out LibriSpeech test-clean speakers: foreground at 0 dB, a different speaker 3 to 15 dB below, both through a room impulse response.

| Model | Foreground F1 | Foreground recall | Background-only frames called speech |
|---|---|---|---|
| Auricle v1 | 94.2% | 97.3% | 35.4% |
| Silero v6 | 89.4% | 97.4% | 81.7% |

### Languages

FLEURS test splits, two hours per language, labelled by Silero on the clean audio; the competing test uses a same-language background talker.

| Language | Clean agreement F1 | Competing F1, v1 vs Silero | Background false alarms, v1 vs Silero |
|---|---|---|---|
| English | 98.6% | 94.6% vs 89.7% | 35% vs 82% |
| Spanish | 98.6% | 95.9% vs 93.3% | 52% vs 87% |
| German | 98.4% | 94.5% vs 92.3% | 61% vs 83% |
| Arabic | 98.1% | 96.4% vs 94.5% | 59% vs 83% |
| Hindi | 97.4% | 95.3% vs 93.4% | 60% vs 80% |
| Mandarin | 95.0% | 89.5% vs 86.1% | 59% vs 84% |
| Japanese | 93.3% | 90.1% vs 87.3% | 63% vs 77% |

### Threshold sweep, competing set

| Threshold | Foreground recall | Background false alarms |
|---|---|---|
| 0.3 | 97.3% | 35.4% |
| 0.5 | 97.3% | 35.4% |
| 0.7 | 75.9% | 14.2% |

The ROC is steep: lowering false alarms by threshold costs recall quickly. The recommended operating point is 0.5.

## Negative results

Two further runs are documented in `docs/RESEARCH.md` of the Auricle repository. Harder competing-talker mixing with 4x loss weight on background-only frames (v1.5) shifted the operating point only: identical false-alarm rate at matched recall, 1.3 points lost on TEN. Adding ten hours each of six languages to the mix (v1.6) changed every number by under 0.3 points. Conclusion: with the mixed audio as the only input, a small causal model separates talkers by level, and level is ambiguous at 0 to 20 dB. The next model takes an enrolment embedding as a second input.

## Limitations

- Not more accurate than Silero v6 on single-speaker real audio (0.9 points below on 30 recordings, within noise); the labels came from Silero, so on that task it can only match.
- Competing-talker numbers are on generated clips until a public meeting-corpus benchmark is added.
- 35% background false alarms is 2.3x better than Silero, not "ignores background speech".
- Trained on English read speech; the six other languages are measured, not trained on. Spontaneous speech, children, shouting and far-field phone audio are unmeasured.
- No field numbers on laptop speakers, Bluetooth headsets or phones yet.

## Use

```ts
import { createVad } from 'auricle';
const vad = await createVad({ model: 'auricle', onSpeechEnd: (audio) => upload(audio) });
await vad.start();
```

Python, with the ONNX file: compute log-mel exactly as `vadtrain/features.py` does (periodic Hann 512, hop 256, 40 HTK mel bands area-normalised, `log(mel + 1e-6)`), run `vad.onnx`, apply sigmoid.

## Attribution

Labels were bootstrapped from [Silero VAD](https://github.com/snakers4/silero-vad) v6 (MIT) on clean audio. Data: LibriSpeech, MUSAN (CC BY 4.0), RIRS_NOISES (Apache 2.0), VocalSound (CC BY-SA 4.0), FLEURS (CC BY 4.0) for evaluation only. The TEN VAD test set was used for evaluation only.

## Citation

```
@software{auricle_v1_2026,
  title  = {Auricle v1: a foreground-selective voice activity detector for the browser},
  author = {Himanshu},
  year   = {2026},
  url    = {https://huggingface.co/auricle-voice/auricle-v1}
}
```
