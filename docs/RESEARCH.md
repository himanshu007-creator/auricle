# Research notes: where Auricle stands and whether to train our own model

Written 2026-10-17 after the benchmark and optimisation passes. Companion to `COMPARISON.md` (numbers) and `docs/ROADMAP.md` (roadmap).

## Are we done?

No. Two columns, kept apart on purpose.

**Measured.** On every axis a script can check, Auricle is the best option on the web for what it does:

| Axis | Auricle | The de facto standard (`@ricky0123/vad-web`) |
|---|---|---|
| Frame accuracy on human-labelled recordings | identical model, F1 93.7% | identical |
| Per 32 ms frame in Chromium and WebKit | 0.05 ms | 0.18 ms |
| Start-up to a ready model | 13 to 15 ms | 175 to 215 ms |
| Download, gzipped | 925 KB | 5,520 KB |
| Turn end after the user stops | 324 ms | 1,432 ms |
| Ducking, barge-in arbitration, heard-text history, verified transcript cleanup, backend adapter | yes | none, by design |

**Unmeasured.** No Firefox or mobile numbers, no field sessions on laptop speakers, no end-to-end latency against a live endpoint, zero users, zero releases. "Best on the web" is true on the measured column and unproven on the human column.

## What we tried to beat the detector with, and what happened

All scored on TEN VAD's 30 human-annotated recordings (frame F1 at threshold 0.5) or on the 240-utterance labelled turn set.

| Idea | Result | Decision |
|---|---|---|
| Causal smoothing of the probability: EMA 0.5 and 0.7, 3-frame median, 2-frame max, two-state hysteresis | every variant lowers F1 (raw 93.73%, best variant 93.67%) | not added; the recurrent state already smooths |
| 16 ms hop: score the last 512 samples every 256 new ones | F1 93.7% to 89.8%, recall 83% | rejected and removed; a 16 ms model must be trained at 16 ms |
| Minimum voiced duration before a segment counts | no effect up to 300 ms, costs one-word turns above | kept as a knob, default 100 ms |
| TEN VAD as the base model | lower F1 than Silero v6 at our thresholds on TEN's own set; licence forbids use inside a library | benchmark reference only |
| FFT STFT, explicit SIMD, time-major layout | 0.32 ms to 0.05 ms per frame, parity unchanged | shipped |
| Eager STT during the hangover, first clause to TTS, connection warm-up | mechanism verified, live figures pending | shipped |

Conclusion: the detector is at its ceiling for this model. Bolt-on tricks lose on the data.

## Can we create our own model?

Yes, and it is the only way past the ceiling. What "better" can and cannot mean:

- **Not better single-speaker accuracy.** Silero v6 scores 93.7% F1 on real labelled audio and was trained on far more data than we can gather. A model trained here will likely land slightly below that on the same test.
- **Better at the thing Silero cannot do: ignoring background speech.** A television, a colleague, a second person all count as speech today, and no threshold fixes that. A foreground-selective model, one that counts only the dominant, sustained speaker and needs no enrollment, is the barrier worth breaking for agents. The recipe is mostly data, not architecture: competing-speaker mixing with foreground-only labels, generated automatically.
- **Not faster.** 0.05 ms per frame is already nothing; speed is not a reason to train.

### What it takes

| Step | Effort |
|---|---|
| Data: a clean speech corpus (LibriSpeech, about 6 GB), a noise set, and a mixing pipeline that creates competing-speaker clips with foreground-only labels automatically | half a day, mostly download and disk |
| Model: a small recurrent network on log-mel features, around 100k parameters, trained on Apple Silicon | a few hours of training, a day of iteration |
| Evaluation: TEN's labelled set for plain accuracy, plus a generated competing-speaker set for the new metric, background false-alarm rate gated by foreground F1 | half a day |
| Port: weights into the existing Rust path behind the `Vad` trait, parity test against the PyTorch reference, the same bench scripts | half a day |

Roughly three to four days for a model that matches Silero on single speakers and beats it when someone else is talking. It is Phase C of `docs/ROADMAP.md`.

### Acceptance criteria before it ships

1. F1 on TEN's labelled set within 1 point of Silero v6.
2. Background false-alarm rate under 5% on the competing-speaker set, where Silero's is measured and published alongside.
3. Under 0.5 ms per frame in WASM, under 1 MB gzipped in total with Silero kept as the fallback.
4. Parity test between the Rust port and the training-time reference, same as today.
5. The bench scripts run unchanged against it.

## What we trained (October 2026)

Pipeline in the companion repository `VAD-train`. Model: 40-band log-mel every 16 ms, two causal convolutions, a two-layer GRU of 64 units, one sigmoid per frame; 83,393 parameters, 334 KB of float weights. Labels: Silero v6 on clean LibriSpeech (961 h, three train splits), kept fixed while the student hears the clip after room impulse responses, MUSAN noise and music, a competing talker 3 to 25 dB down in 45% of clips, crowd babble, VocalSound laughs and coughs at user level, bandlimiting, clipping and gain. 20,000 steps of 128 five-second clips on a six-core laptop CPU, 9.7 hours.

### v1: the result

| | Student v1 | Silero v6 | Criterion |
|---|---|---|---|
| TEN F1 at 0.5, 30 human-labelled recordings | 92.7% | 93.6% | within 1 point: pass |
| Competing talker, foreground F1 | 94.2% | 89.4% | within 3 points: pass, 4.8 points ahead |
| Competing talker, foreground recall | 97.3% | 97.4% | |
| Background-only frames called speech | 35.4% | 81.7% | under 5%: fail, 2.3x fewer than Silero |

Export parity between the NumPy reference and PyTorch: 2.4e-7. The competing set is 500 generated clips from held-out LibriSpeech speakers, both talkers through a room, background 3 to 15 dB below the foreground.

### v1.5: a negative result worth keeping

Hypothesis: the 35% floor is a training-pressure problem. Recipe: 4,000 more steps from the v1 weights, competing talker in 70% of clips at 0 to 20 dB down, more background-only and babble clips, 4x loss weight on frames where only the far talker speaks.

| | Recall | Background false alarms | TEN F1 |
|---|---|---|---|
| v1 at threshold 0.5 | 97.3% | 35.4% | 92.7% |
| v1.5 at threshold 0.3 | 97.3% | 35.4% | 92.2% |
| v1.5 at threshold 0.5 | 93.7% | 25.3% | 91.4% |

At matched recall the false-alarm rate is identical to the decimal. The post-training moved the operating point, which a threshold does for free, and cost 1.3 points on real speech; dev F1 fell at every evaluation after step 500. Conclusion: with the mixed audio as its only input, a small causal model separates talkers by level and nothing else, and level is ambiguous at 0 to 20 dB. Reweighting the loss, harder mixing and more steps do not add information and do not move the curve. Rejected. The rule that follows: post-training is worth running only when it adds inputs the model has never seen (new languages, field microphones, sharper labels, an enrolment embedding, the playback reference), never when it only reweights what it already saw.

### v1 across languages

FLEURS test splits, 2 hours per language, labelled by Silero on the clean audio; the competing-talker test uses a same-language background talker.

| Language | Clean agreement F1 | Competing F1, v1 vs Silero | Background false alarms, v1 vs Silero |
|---|---|---|---|
| English (held-out LibriSpeech) | 98.6% | 94.6% vs 89.7% | 35% vs 82% |
| Spanish | 98.6% | 95.9% vs 93.3% | 52% vs 87% |
| German | 98.4% | 94.5% vs 92.3% | 61% vs 83% |
| Arabic | 98.1% | 96.4% vs 94.5% | 59% vs 83% |
| Hindi | 97.4% | 95.3% vs 93.4% | 60% vs 80% |
| Mandarin | 95.0% | 89.5% vs 86.1% | 59% vs 84% |
| Japanese | 93.3% | 90.1% vs 87.3% | 63% vs 77% |

The model transfers: recall holds and it beats Silero on the competing test in every language. Mandarin and Japanese lose 3.6 and 5.3 points on clean agreement, and same-language background talkers trigger it far more than English ones. v1.6, in progress, adds about 10 hours of FLEURS training speech per language to the mix with the v1 objective, 2,500 steps from the v1 weights; it replaces v1 only if TEN stays within a point, English competing F1 holds, and the Mandarin and Japanese rows improve.

### Revised acceptance

Criterion 2 above asked for under 5% background false alarms. v1.5 showed that number is out of reach for a level-only model, and the honest claim that ships is the measured ratio against Silero on the same clips. The 5% target moves to the enrolment model (Phase C of `docs/ROADMAP.md`), which gives the detector the identity cue it lacks.
