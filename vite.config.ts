import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

const rawPort = Number.parseInt(process.env.VITE_DEV_SERVER_PORT ?? '5173', 10)
const devPort = Number.isNaN(rawPort) ? 5173 : rawPort
const devHost = process.env.VITE_DEV_SERVER_HOST
const wsProxyTarget = process.env.VITE_WS_PROXY_TARGET ?? 'ws://localhost:8765'
const resolvedHost = devHost && devHost.trim().length > 0 ? devHost : true

export default defineConfig({
  plugins: [react()],
  root: 'src/frontend',
  publicDir: '../../public',
  build: {
    outDir: '../../dist',
    emptyOutDir: true
  },
  server: {
    port: devPort,
    host: resolvedHost,
    strictPort: true,
    proxy: {
      '/ws': {
        target: wsProxyTarget,
        ws: true,
        changeOrigin: true,
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
