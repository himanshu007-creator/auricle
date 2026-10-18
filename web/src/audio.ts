// Web Audio setup: mic -> capture worklet -> brain worker -> playback worklet,
// with the optional Chromium loopback so the browser's echo canceller sees
// the agent's audio.
import captureSource from './worklets/capture.worklet.js?raw';
import playbackSource from './worklets/playback.worklet.js?raw';
import { WORKLET_WASM_B64 } from './generated/worklet_wasm.ts';
import { base64ToBytes } from './wasm.ts';

export type LoopbackMode = 'auto' | 'on' | 'off';

export interface AudioGraphOptions {
  loopback: LoopbackMode;
  playbackCapacitySeconds?: number;
}

/** Chromium applies browser AEC only to audio that leaves through a media element, not Web Audio. */
export function needsLoopback(): boolean {
  return /Chrom(e|ium)|Edg\/|OPR\//.test(navigator.userAgent);
}

export class AudioGraph {
  private constructor(
    readonly context: AudioContext,
    readonly capture: AudioWorkletNode,
    readonly playback: AudioWorkletNode,
    private readonly keepAlive: GainNode,
    private loopback: { element: HTMLAudioElement; close(): void } | null,
    private micStream: MediaStream,
  ) {}

  static async create(options: AudioGraphOptions): Promise<AudioGraph> {
    const micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: false, channelCount: 1 },
    });
    const context = new AudioContext({ latencyHint: 'interactive' });
    await context.resume();

    const module = await WebAssembly.compile(base64ToBytes(WORKLET_WASM_B64));
    const workletUrl = (src: string) => URL.createObjectURL(new Blob([src], { type: 'text/javascript' }));
    await context.audioWorklet.addModule(workletUrl(captureSource));
    await context.audioWorklet.addModule(workletUrl(playbackSource));

    const capture = new AudioWorkletNode(context, 'auricle-capture', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
      processorOptions: { module },
    });
    const playback = new AudioWorkletNode(context, 'auricle-playback', {
      numberOfInputs: 0,
      numberOfOutputs: 1,
      outputChannelCount: [1],
      processorOptions: { module, inputRate: 24000, capacitySeconds: options.playbackCapacitySeconds ?? 60 },
    });

    const mic = context.createMediaStreamSource(micStream);
    mic.connect(capture);
    // A worklet with nothing downstream can be skipped by the renderer; keep it pulled, silently.
    const keepAlive = context.createGain();
    keepAlive.gain.value = 0;
    capture.connect(keepAlive).connect(context.destination);

    const useLoopback = options.loopback === 'on' || (options.loopback === 'auto' && needsLoopback());
    let loopback: { element: HTMLAudioElement; close(): void } | null = null;
    if (useLoopback) {
      try {
        loopback = await createLoopback(context, playback);
      } catch (err) {
        console.warn('loopback failed, playing through the AudioContext instead', err);
      }
    }
    if (!loopback) playback.connect(context.destination);
    return new AudioGraph(context, capture, playback, keepAlive, loopback, micStream);
  }

  get loopbackActive(): boolean {
    return this.loopback !== null;
  }

  close(): void {
    this.capture.port.postMessage({ type: 'stop' });
    this.micStream.getTracks().forEach((t) => t.stop());
    this.loopback?.close();
    void this.context.close();
  }
}

/**
 * Routes `source` through a local RTCPeerConnection into an <audio> element.
 * Chromium's echo canceller then treats the agent's voice as far-end audio.
 */
async function createLoopback(context: AudioContext, source: AudioNode): Promise<{ element: HTMLAudioElement; close(): void }> {
  const destination = context.createMediaStreamDestination();
  source.connect(destination);
  const sender = new RTCPeerConnection();
  const receiver = new RTCPeerConnection();
  sender.onicecandidate = (e) => e.candidate && receiver.addIceCandidate(e.candidate);
  receiver.onicecandidate = (e) => e.candidate && sender.addIceCandidate(e.candidate);
  destination.stream.getAudioTracks().forEach((track) => sender.addTrack(track, destination.stream));

  const element = document.createElement('audio');
  element.autoplay = true;
  element.setAttribute('playsinline', '');
  element.style.display = 'none';
  document.body.appendChild(element);
  const connected = new Promise<void>((resolve) => {
    receiver.ontrack = (e) => {
      element.srcObject = e.streams[0];
      void element.play().then(resolve, resolve);
    };
  });
  const offer = await sender.createOffer();
  await sender.setLocalDescription(offer);
  await receiver.setRemoteDescription(offer);
  const answer = await receiver.createAnswer();
  await receiver.setLocalDescription(answer);
  await sender.setRemoteDescription(answer);
  await connected;
  return {
    element,
    close() {
      sender.close();
      receiver.close();
      element.remove();
    },
  };
}
