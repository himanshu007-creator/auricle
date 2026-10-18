import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    lib: { entry: 'src/index.ts', formats: ['es'], fileName: 'auricle-react' },
    rollupOptions: { external: ['react', 'auricle'] },
    target: 'es2022',
  },
  test: { environment: 'jsdom' },
});
