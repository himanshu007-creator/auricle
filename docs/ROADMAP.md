# Auricle: what lies ahead

Written 2026-10-17, after the first day of measurements. Everything in the "Now" column is measured; everything after it is a plan.

## Where we are

| | State |
|---|---|
| Detector | Silero v6 in hand-written Rust, 0.05 ms per 32 ms frame in Chromium and WebKit, outputs within 1e-6 of onnxruntime, F1 93.7% on TEN's human-labelled recordings |
| Turn policy | Graded events, barge-in arbiter, 100% recall and no false alarms on a 240-utterance labelled set, turn end 324 ms after speech end |
| Payload | 925 KB gzipped including weights, no runtime, nothing to copy |
| Conversation path | OpenAI-compatible STT, chat and TTS; eager STT during the hangover; first clause to TTS; heard-text truncation |
| Proof | 16 Rust tests, 13 vitest tests, four bench scripts, Playwright matrix for Chromium and WebKit |
| Not proven | Firefox and mobile, live end-to-end latency, real users on laptop speakers |

## The architecture we are building towards

The listener lives on the client; the business brings a model endpoint and, optionally, a server.

```
                 user's browser / app                               business
  mic ─► Auricle listener (local, <1 MB) ─► STT ──┐
          detects, ducks, stops, cleans      LLM ──┼──► any OpenAI-compatible endpoint
  speaker ◄─ playback worklet ◄── TTS ◄─────────┘     (hosted API, a thin proxy, or their own backend)
```

Three things run only on the client: the detector and turn policy, the playback
with ducking and hard stop, and the conversation state (what was heard, what was
cut). Nothing leaves the device except the speech segments the client decides to
transcribe. The business then chooses one of three shapes, and the SDK supports all:

1. **No server**: the client calls a hosted API with a scoped key. Prototypes, internal tools, privacy-first apps.
2. **Thin proxy**: a function that adds the real key and CORS, forwards to the vendor, logs and bills. Fifteen lines; in the README.
3. **Own backend**: the business exposes the three OpenAI-shaped endpoints and runs anything behind them (tools, retrieval, its own models). The listener does not care.

The consequence: the business pays for tokens and a cheap proxy, not for a media
server per session, and the part that decides how the conversation feels runs
on hardware the user already owns. This is the inverse of LiveKit, Vapi and
Retell, where the listener is rented by the minute on the vendor's server. It
is also why Auricle can be a drop-in for those stacks later: a duck that happens
locally beats one that round-trips through a room.

## Principles that do not change

1. **No claim without a script that reproduces it.** `bench/` is part of the product.
2. **The audio path never waits for the main thread, the network or a garbage collector.**
3. **Deletion-only text.** The agent never receives a word the user did not say.
4. **Any OpenAI-compatible endpoint.** The user brings the model; we bring the ears and the manners.
5. **Permissive licence for everything we ship**, which is why TEN VAD stays a benchmark reference only.

## Roadmap

### Phase A: field truth (weeks 1 to 2)

- Collect sessions with the demo's export on laptop speakers, headphones, Bluetooth and phones; add the plan's missing metrics (stop latency on real interruptions, false barge-ins per ten agent turns, cut-offs).
- Firefox and mobile in the Playwright matrix (needs a machine where Playwright's Firefox runs; iOS Safari through WebKit on device).
- Live latency table in `docs/COMPARISON.md` from `bench/latency.mjs` against Groq, OpenAI and a self-hosted server.

### Phase B: echo and devices (weeks 2 to 4)

- Rust reference-aware echo filter in the capture worklet (delay estimate, frequency-domain subtraction, NLMS), behind a flag, measured by echo self-trigger rate against the loopback-only path.
- Echo ratio and agent-playing state as inputs to a small logistic scorer in front of the turn machine, trained on exported sessions.
- Headphone and Bluetooth profiles detected from measured playback-to-capture delay instead of device labels.

### Phase C: who is talking (weeks 4 to 8)

The one failure a plain VAD cannot fix is background speech: a TV, a colleague, a second person. Two routes, both evaluated on TEN's set plus a competing-speaker mix:

- **Foreground VAD**: a small streaming model trained with competing-speaker mixing and foreground-only labels, enrollment-free, behind the same `Vad` trait. Target: the same F1 on single-speaker audio and a background false-alarm rate under 5% on mixes.
- **Lazy speaker gate**: embed the first confirmed turn, add cosine similarity as evidence, never block on it.

### Phase D: learned endpointing (weeks 6 to 10)

- Replace the fixed hangover with a learned time-to-next-onset head fused with the text cues already shipped; keep the hangover as the fallback.
- Audio end-of-turn model (Smart Turn class) in WASM, with the same budget rule: under 3 ms per decision or it does not ship.
- Eager response mode: start generation at silence candidate, discard on resume, measured in discarded tokens per turn.

### Phase E: one core, many languages (weeks 8 to 14)

The Rust crates already compile natively. The plan is one C ABI and thin bindings, so any agent harness can use the same detector and policy:

| Target | Route | What you get |
|---|---|---|
| C / C++ | `auricle-ffi` crate, `cbindgen` header, static and shared library | `auricle_new`, `auricle_push_frame`, `auricle_poll_event`, `auricle_clean_text` |
| Python | PyO3 wheel on top of the same crates (`pip install auricle-vad`) | `Auricle().feed(np.ndarray)` yielding events; NumPy in, events out; works with sounddevice |
| Node (native) | N-API via `napi-rs`, or the existing WASM in Node | the same `Brain` class as the browser |
| Go | cgo over the C ABI | for the optional server-side proxy |
| Swift / Kotlin | C ABI via XCFramework and JNI | iOS and Android apps with the same policy as the web |
| CLI | `auricle listen` | prints JSON events and segment WAVs to stdout for any agent harness to consume |

Every binding runs the same eval harness on the same fixtures, so a number measured in the browser means the same thing in Python.

### Phase F: agents that listen (weeks 10 to 16)

- **MCP server**: `auricle-mcp` exposing `listen` (returns clean text when a turn ends), `speak` (plays through the ducking worklet), and `interrupts` as events, so a coding agent or a desktop assistant can take voice tasks with no audio code of its own.
- **Streaming STT adapter** for endpoints that offer it, so the eager path becomes a true stream and the post-speech round trip disappears.
- **Realtime WebSocket adapter** behind the same `ConversationBackend` interface, with a `turnTaking` capability flag so a server that owns turn detection is not second-guessed.
- A Vue composable next to the shipped `auricle-react` hook, generated from the same event types.

### Phase G: release discipline

- Semantic versioning from 0.1; a published comparison on a shared test set with every release.
- Model table in the README with size, licence and the F1 on TEN's set for every bundled model.
- A "golden sessions" corpus: exported real sessions replayed in CI through `auricle-eval`, so a regression in the policy fails a build.

## Ideas we tested and will not pursue

- Causal smoothing of the probability (EMA, median, hysteresis): lowers F1.
- 16 ms hop by overlapping windows on the Silero weights: recall collapses; a 16 ms model must be trained for 16 ms.
- TEN VAD as the base: licence forbids library use, and Silero v6 scores higher F1 on TEN's own set at our thresholds.
- Threads or SharedArrayBuffer in the browser: rules out GitHub Pages and buys nothing at 0.05 ms per frame.

## How to help

Run `bench/latency.mjs` against your endpoint and send the output. Export a session from the demo on a device we have not measured. Both go straight into the next comparison table.
