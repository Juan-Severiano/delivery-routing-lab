import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Os fluxogramas têm bundle próprio; o replay não carrega o editor de grafos.
export default defineConfig({
  publicDir: false,
  plugins: [react()],
  define: { "process.env.NODE_ENV": JSON.stringify("production") },
  build: {
    outDir: "dist-flows",
    lib: {
      entry: "src/flows/main.tsx",
      formats: ["es"],
      fileName: "flow-canvas",
      cssFileName: "flow-canvas",
    },
  },
});
