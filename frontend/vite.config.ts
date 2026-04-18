import { defineConfig } from 'vite'
import path from 'path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'


function figmaAssetResolver() {
  return {
    name: 'figma-asset-resolver',
    resolveId(id) {
      if (id.startsWith('figma:asset/')) {
        const filename = id.replace('figma:asset/', '')
        return path.resolve(__dirname, 'src/assets', filename)
      }
    },
  }
}

export default defineConfig({
  plugins: [
    figmaAssetResolver(),
    // The React and Tailwind plugins are both required for Make, even if
    // Tailwind is not being actively used – do not remove them
    react(),
    tailwindcss(),
  ],
  resolve: {
    alias: {
      // Alias @ to the src directory
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    proxy: {
      // Each FastAPI route the frontend talks to needs an entry here —
      // Vite's SPA fallback otherwise answers unknown POSTs with an
      // empty 404 body, and the frontend's `response.json()` throws
      // "Unexpected end of JSON input". Regex key matches the exact
      // path and any sub-path, so "/upload" also catches "/upload/resolve".
      "^/(health|query|query/stream|upload|upload/resolve|uploads|upload_pdf)(/|$)":
        "http://127.0.0.1:8000",
    },
  },

  // File types to support raw imports. Never add .css, .tsx, or .ts files to this.
  assetsInclude: ['**/*.svg', '**/*.csv'],
})
