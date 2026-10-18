# Changelog

## 0.1.0 (unreleased)

First release candidate.

- `auricle`: the `Auricle` class (listener, turn policy, ducking, barge-in, heard-text history, OpenAI-compatible STT, chat and TTS), `createVad` for the detector alone, and `auricle/compat` with the `@ricky0123/vad-web` shaped `MicVAD` API.
- One bundled detector, Auricle v1 (83k parameters, 334 KB, MIT), as a hand-written Rust forward pass in one WebAssembly module: no inference runtime, no assets to copy, 350 KB gzipped for the whole engine. Silero VAD v6 stays available as an opt-in cargo feature of `auricle-vad` for the engine benchmark against vad-web.
- `auricle-react`: `useAuricle`, `useVad` (with `pause`, `resume`, `setThresholds`, `stats`) and the `@ricky0123/vad-react` shaped `useMicVAD`.
- Session accounting: `auricle.stats`, `vad.stats` and `formatStats` count audio sent to speech-to-text against session length, text-to-speech characters avoided by interruption, and stop latency, from the session's own events.
- JSDoc on every public option, method, event and type.
- A `<script>` build (`dist/auricle.iife.js`, `window.Auricle`) served by jsDelivr and unpkg.
- Rust crates `auricle-dsp`, `auricle-vad`, `auricle-turn`, `auricle-text`, `auricle-worklet`, `auricle-brain`, `auricle-eval`.
- Benchmarks in `bench/` and the numbers in `docs/COMPARISON.md`.
