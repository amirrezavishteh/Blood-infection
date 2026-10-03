import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const api = process.env.SEPSIS_API_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: { "/v1": api, "/health": api },
  },
  test: { environment: "jsdom", globals: true },
});
