import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const previewMode = process.env.BASE44_PREVIEW_MODE === '1'
const sandboxDomain = process.env.BASE44_SANDBOX_HOST_DOMAIN || ''

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    // Allow the preview proxy host when in Base44 preview mode
    allowedHosts: previewMode && sandboxDomain
      ? [`.${sandboxDomain}`]
      : true,
  },
  resolve: {
    alias: {
      '@': '/src',
    },
  },
})
