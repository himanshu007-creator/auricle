# Migrating from `@ricky0123/vad-web`

Auricle ships a `compat` entry with the same shape as vad-web's `MicVAD`, so the first step is one import. The second step, optional, is to use what vad-web does not have.

## Step 1: change the import

Before:

```ts
import { MicVAD, utils } from '@ricky0123/vad-web';

const vad = await MicVAD.new({
  positiveSpeechThreshold: 0.5,
  negativeSpeechThreshold: 0.35,
  redemptionMs: 240,
  onSpeechStart: () => ui.listening(),
  onSpeechEnd: (audio) => upload(utils.encodeWAV(audio)),
});
vad.start();
```

After:

```ts
import { MicVAD, utils } from 'auricle/compat';

const vad = await MicVAD.new({
  positiveSpeechThreshold: 0.5,
  negativeSpeechThreshold: 0.35,
  redemptionMs: 240,
  onSpeechStart: () => ui.listening(),
  onSpeechEnd: (audio) => upload(utils.encodeWAV(audio)),
});
vad.start();
```

Delete `baseAssetPath`, `onnxWASMBasePath` and the copied `.onnx` and `.wasm` files: Auricle has no assets. The model and the runtime are inside the JavaScript.

React: `import { useMicVAD } from 'auricle-react'` returns the same `{ listening, errored, loading, userSpeaking, pause, start, toggle }`.

Script tag, no bundler:

```html
<script src="https://cdn.jsdelivr.net/npm/auricle@0.1.0/dist/auricle.iife.js"></script>
<script>
  const { MicVAD } = Auricle.compat;
  MicVAD.new({ startOnLoad: true, onSpeechEnd: (audio) => console.log(audio.length) });
</script>
```

## What maps one to one

| vad-web | Auricle compat | Note |
|---|---|---|
| `onSpeechStart` | same | fires on the speech candidate, three voiced frames in |
| `onSpeechRealStart` | same | fires on confirmation, six of the last ten frames voiced |
| `onSpeechEnd(audio)` | same | `Float32Array` at 16 kHz with 512 ms of pre-roll |
| `onVADMisfire` | same | |
| `onFrameProcessed({isSpeech, notSpeech}, frame)` | same signature | `frame` is empty: audio stays in the worklet |
| `positiveSpeechThreshold`, `negativeSpeechThreshold` | same | Auricle defaults 0.5 / 0.35 |
| `redemptionMs` | same | the hangover; Auricle default 240, vad-web default 1400 |
| `minSpeechMs` | same | Auricle default 100, vad-web default 400 |
| `startOnLoad` | same | |
| `start()`, `pause()`, `destroy()`, `listening` | same | |
| `utils.encodeWAV` | same | |

## What is different

- **Defaults end turns sooner.** vad-web waits 1,400 ms of silence; Auricle waits 240 ms, which on the labelled turn set closes a turn 1.1 s sooner at the same recall. If you were relying on long pauses merging into one segment, pass `redemptionMs: 1400`.
- **Pre-roll is fixed** at 512 ms. `preSpeechPadMs` is accepted and ignored.
- **No 8 kHz input, no legacy models.** 16 kHz only; the detector is Auricle v1.
- **Inference runs in a worker**, not on the main thread.

## Step 2: what vad-web does not have

Once the import works, the same package gives you, without new dependencies:

- `createVad()` with graded events: candidate, confirmed, end, misfire, and a `thresholds` object.
- `Auricle.create({ backend })`: the whole conversation loop against any OpenAI-compatible endpoint, with ducking in 30 ms, hard stop in 15 ms on interruption, and the agent's history cut to what the user actually heard.
- The detector is Auricle v1, which fires less than half as often on a second talker than Silero v6 at the same foreground recall; there is no model option to set.
- `playClip()`: play anything through the ducking path.

The README shows each in five lines.
