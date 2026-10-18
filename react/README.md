<p align="center"><img src="https://raw.githubusercontent.com/himanshu007-creator/auricle/main/demo/public/brand/logo.svg" alt="Auricle" height="64" /></p>

# auricle-react

React hooks over [`auricle`](https://www.npmjs.com/package/auricle): voice activity detection and interruptible voice conversations in the browser.

```sh
npm install auricle auricle-react
```

```tsx
import { useVad } from 'auricle-react';

function Recorder() {
  const { listening, speaking, probability, segments, stats, toggle } = useVad({
    onSpeechEnd: (audio) => upload(audio),
  });
  return <button onClick={toggle}>{listening ? (speaking ? 'speaking' : 'listening') : 'start'}</button>;
}
```

```tsx
import { useAuricle } from 'auricle-react';

function Agent({ backend }) {
  const { state, turns, agentText, heardText, stats, start, interrupt } = useAuricle({ backend });
  ...
}
```

Coming from `@ricky0123/vad-react`? `useMicVAD` from this package returns the same `{ listening, errored, loading, userSpeaking, pause, start, toggle }`.

Repository, numbers and docs: https://github.com/himanshu007-creator/auricle

Both hooks expose `stats`, the session counters (audio sent to speech-to-text against session length, text-to-speech characters avoided by interruption, stop latency), refreshed every second. `useVad` also returns `pause`, `resume` and `setThresholds`. Every option and return field has a JSDoc comment.
