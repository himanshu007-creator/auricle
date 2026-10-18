// Capture worklet: device-rate mic input -> Rust resampler -> 512-sample 16 kHz
// frames posted straight to the brain worker. The WASM module is compiled on
// the main thread and handed over as a WebAssembly.Module in processorOptions.
class AuricleCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const { module } = options.processorOptions;
    this.instance = new WebAssembly.Instance(module, {});
    this.exports = this.instance.exports;
    this.handle = this.exports.capture_new(sampleRate);
    this.inputPtr = this.exports.capture_input_ptr(this.handle);
    this.framesPtr = this.exports.capture_frames_ptr(this.handle);
    this.frameLen = this.exports.capture_frame_len();
    this.capacity = this.exports.capture_input_capacity();
    this.buffer = null;
    this.brainPort = null;
    this.running = true;
    this.port.onmessage = (e) => {
      if (e.data.type === 'connect') this.brainPort = e.data.port;
      if (e.data.type === 'stop') this.running = false;
    };
  }

  // Views must be rebuilt if WASM memory grows (the old ArrayBuffer detaches).
  views() {
    const mem = this.exports.memory.buffer;
    if (this.buffer !== mem) {
      this.buffer = mem;
      this.input = new Float32Array(mem, this.inputPtr, this.capacity);
      this.frames = new Float32Array(mem, this.framesPtr, this.frameLen * 8);
    }
  }

  process(inputs) {
    if (!this.running) return false;
    const channels = inputs[0];
    if (!channels || channels.length === 0 || !this.brainPort) return true;
    this.views();
    const n = Math.min(channels[0].length, this.capacity);
    if (channels.length === 1) {
      this.input.set(channels[0].subarray(0, n));
    } else {
      for (let i = 0; i < n; i++) {
        let sum = 0;
        for (let c = 0; c < channels.length; c++) sum += channels[c][i];
        this.input[i] = sum / channels.length;
      }
    }
    const ready = this.exports.capture_push(this.handle, n);
    this.views();
    for (let f = 0; f < ready; f++) {
      const frame = this.frames.slice(f * this.frameLen, (f + 1) * this.frameLen);
      this.brainPort.postMessage(frame, [frame.buffer]);
    }
    return true;
  }
}

registerProcessor('auricle-capture', AuricleCapture);
