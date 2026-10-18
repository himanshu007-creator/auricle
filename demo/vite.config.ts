import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';
import { resolve, dirname } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));

// The demo builds the library from source so one `vite build` produces the whole site.
export default defineConfig({
  base: process.env.AURICLE_BASE ?? '/',
  resolve: { alias: { 'auricle': fileURLToPath(new URL('../web/src/index.ts', import.meta.url)) } },
  build: {
    target: 'es2022',
    rollupOptions: { input: { index: resolve(here, 'index.html'), philosophy: resolve(here, 'philosophy.html'), docs: resolve(here, 'docs.html'),
        benchmarks: resolve(here, 'benchmarks.html'), model: resolve(here, 'model.html'), roadmap: resolve(here, 'roadmap.html'), blog: resolve(here, 'blog/index.html'), blog1: resolve(here, 'blog/browser-vad-without-a-runtime.html'), blog2: resolve(here, 'blog/the-other-person-in-the-room.html'), blog3: resolve(here, 'blog/what-silence-costs.html') } },
  },
  worker: { format: 'es' },
});
