import { defineConfig } from 'vite';

// The <script src> build: one classic script defining window.Auricle, with
// the vad-web shaped API at window.Auricle.compat.
export default defineConfig({
  build: {
    lib: { entry: 'src/iife.ts', formats: ['iife'], name: 'Auricle', fileName: () => 'auricle.iife.js' },
    target: 'es2022',
    emptyOutDir: false,
    sourcemap: true,
  },
  worker: { format: 'iife' },
});
