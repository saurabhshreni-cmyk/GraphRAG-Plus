import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// API base URL is read at runtime from `import.meta.env.VITE_API_BASE`.
// Defaults to the local backend dev server.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: false,
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
