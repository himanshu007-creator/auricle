// Entry for the single-file <script> build: the core API plus the vad-web
// shaped API under `compat`, so a page can use either without a bundler.
export * from './index.ts';
export * as compat from './compat.ts';
