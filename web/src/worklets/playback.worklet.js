// Playback worklet: agent audio chunks -> Rust ring -> device rate, with
// ducking, a fade-out on hard stop and a played-sample counter. Control
// commands (duck / unduck / stop) arrive on a direct port from the brain
// worker, so stopping never waits for the main thread.
const DUCK_GAIN = 0.25; // about -12 dB
const DUCK_RAMP_MS = 30;
const STOP_FADE_MS = 15;
const SILENCE_HOLD_QUANTA = 40; // ~110 ms at 48 kHz before "playing" goes false

class AuriclePlayback extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const { module, inputRate, capacitySeconds } = options.processorOptions;
    this.exports = new WebAssembly.Instance(module, {}).exports;
    this.handle = this.exports.playback_new(sampleRate, inputRate, capacitySeconds || 60);
    this.inputPtr = this.exports.playback_input_ptr(this.handle);
    this.outputPtr = this.exports.playback_output_ptr(this.handle);
    this.buffer = null;
    this.pending = []; // chunks that did not fit yet: { samples, rate, offset }
    this.playing = false;
    this.silentQuanta = 0;
    this.brainPort = null;
    this.port.onmessage = (e) => this.onMessage(e.data);
  }

  onMessage(msg) {
    switch (msg.type) {
      case 'connect':
        this.brainPort = msg.port;
        this.brainPort.onmessage = (e) => this.onMessage(e.data);
        break;
      case 'chunk':
        this.pending.push({ samples: msg.samples, rate: msg.rate, offset: 0 });
        break;
      case 'duck':
        this.exports.playback_set_gain(this.handle, DUCK_GAIN, DUCK_RAMP_MS);
        break;
      case 'unduck':
        this.exports.playback_set_gain(this.handle, 1.0, DUCK_RAMP_MS);
        break;
      case 'stop':
        this.pending = [];
        this.exports.playback_stop(this.handle, STOP_FADE_MS);
        break;
    }
  }

  views() {
    const mem = this.exports.memory.buffer;
    if (this.buffer !== mem) {
      this.buffer = mem;
      this.input = new Float32Array(mem, this.inputPtr, 4096);
      this.output = new Float32Array(mem, this.outputPtr, 4096);
    }
  }

  // Moves as much pending audio into the ring as fits this quantum.
  feed() {
    while (this.pending.length) {
      const chunk = this.pending[0];
      if (!this.exports.playback_set_input_rate(this.handle, chunk.rate)) return; // wait until drained
      const n = Math.min(4096, chunk.samples.length - chunk.offset);
      this.views();
      this.input.set(chunk.samples.subarray(chunk.offset, chunk.offset + n));
      const accepted = this.exports.playback_write(this.handle, n);
      chunk.offset += accepted;
      if (chunk.offset >= chunk.samples.length) this.pending.shift();
      if (accepted < n) return; // ring full
    }
  }

  process(_inputs, outputs) {
    this.feed();
    const out = outputs[0];
    const n = out[0].length;
    const consumed = this.exports.playback_render(this.handle, n) === 1;
    this.views();
    for (const channel of out) channel.set(this.output.subarray(0, n));

    const played = this.exports.playback_played(this.handle);
    if (this.exports.playback_take_flushed(this.handle) === 1) {
      this.port.postMessage({ type: 'flushed', played });
    }
    this.silentQuanta = consumed ? 0 : this.silentQuanta + 1;
    const playing = consumed || (this.playing && this.silentQuanta < SILENCE_HOLD_QUANTA);
    if (playing !== this.playing) {
      this.playing = playing;
      const msg = { type: 'playing', value: playing, played };
      this.port.postMessage(msg);
      if (this.brainPort) this.brainPort.postMessage(msg);
    }
    return true;
  }
}

registerProcessor('auricle-playback', AuriclePlayback);
