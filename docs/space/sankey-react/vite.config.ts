import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// HF Static Space serves from root, so relative paths keep the bundle
// portable (works at any URL prefix and on local file:// preview).
export default defineConfig({
  plugins: [react()],
  base: './',
  build: { outDir: 'dist' },
})
