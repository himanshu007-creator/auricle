<p align="center"><img src="https://raw.githubusercontent.com/himanshu007-creator/auricle/main/demo/public/brand/logo.svg" alt="Auricle" height="64" /></p>

# auricle

The listener for voice agents, in the browser. Voice activity detection, turn taking, barge-in and ducking as one Rust program compiled to WebAssembly, plus an adapter for any OpenAI-compatible speech-to-text, chat and text-to-speech API. No inference runtime, no model files to copy.

```sh
npm install auricle
```

```ts
import { createVad } from 'auricle';

const vad = await createVad({
  onSpeechStart: () => ui.listening(),
  onSpeechEnd: (audio) => upload(audio),   // Float32Array, 16 kHz, 512 ms pre-roll
});
await vad.start();                          // from a click: it asks for the microphone
```

A whole interruptible conversation:

```ts
import { Auricle, OpenAICompat } from 'auricle';

const auricle = await Auricle.create({
  backend: new OpenAICompat({ baseURL: 'https://api.groq.com/openai/v1', apiKey, models: { stt: 'whisper-large-v3-turbo', llm: 'openai/gpt-oss-20b', tts: 'canopylabs/orpheus-v1-english' } }),
});
auricle.on('turn', (t) => log(t.clean));
auricle.on('interrupted', (e) => log(e.heardText));   // what the user actually heard
await auricle.start();
```

Coming from `@ricky0123/vad-web`? `import { MicVAD, utils } from 'auricle/compat'` is the same API.

Script tag: `https://cdn.jsdelivr.net/npm/auricle@0.1.0/dist/auricle.iife.js` defines `window.Auricle`.

Session accounting: `auricle.stats` / `vad.stats` / `formatStats()` count audio sent to speech-to-text against session length and text-to-speech characters avoided by interruption, from the session's own events. Every option, method, event and type has a JSDoc comment.

Numbers, the comparison with vad-web, the model card and the full API are in the repository: https://github.com/himanshu007-creator/auricle and the docs at https://himanshu007-creator.github.io/auricle/docs.html
