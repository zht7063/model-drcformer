import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

// Tauri loads devUrl from src-tauri/tauri.conf.json; keep the port fixed.
export default defineConfig({
  plugins: [vue()],
  clearScreen: false,
  server: { host: "127.0.0.1", port: 1420, strictPort: true },
  build: { target: "es2021" },
});
