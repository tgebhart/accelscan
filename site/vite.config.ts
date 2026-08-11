import { defineConfig } from 'vite';
import { resolve } from 'node:path';

// Two static pages, no framework, no server-side code. `base: './'` keeps
// asset URLs relative so the build works unmodified when served from a
// CloudFront distribution rooted at the bucket (no subpath assumptions).
export default defineConfig({
  base: './',
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        papers: resolve(__dirname, 'papers.html'),
      },
    },
  },
});
