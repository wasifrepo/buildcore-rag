import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/query": "http://backend:8000",
      "/evaluate": "http://backend:8000",
      "/traces": "http://backend:8000",
    },
  },
});
