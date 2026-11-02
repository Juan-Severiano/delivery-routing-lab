import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig(({ mode }) => ({
  base: "./",
  // O build do Pages recebe apenas o snapshot público validado por build_pages.py.
  publicDir: mode === "pages" ? false : "public",
  plugins: [react()],
}));
