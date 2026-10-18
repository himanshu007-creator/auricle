import { defineConfig } from 'vite';

// Library build: ESM entries for `auricle` and `auricle/compat`. Worklet
// sources (?raw) and the worker (?worker&inline) are inlined, and the WASM is
// already base64 in src/generated, so consumers copy no assets.
// The <script> build (dist/auricle.iife.js) is vite.iife.config.ts.
export default defineConfig({
  build: {
    lib: { entry: { auricle: 'src/index.ts', compat: 'src/compat.ts' }, formats: ['es'] },
    target: 'es2022',
    sourcemap: true,
    rollupOptions: { output: { entryFileNames: '[name].js', chunkFileNames: 'chunks/[name]-[hash].js' } },
  },
  worker: { format: 'es' },
});
