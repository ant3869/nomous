import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  root: 'src/frontend',
  publicDir: '../../public',
  build: {
    outDir: '../../dist',
    emptyOutDir: true
  },
  server: { 
    port: 5173, 
    host: true,
    proxy: {
      '/ws': {
        target: 'ws://localhost:8765',
        ws: true
      }
    }
  },
  resolve: {
    alias: [
      {
        find: '@',
        replacement: path.resolve(__dirname, './src/frontend')
      }
    ]
  }
})
