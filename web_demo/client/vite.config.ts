import { ProxyOptions, defineConfig, loadEnv } from "vite";
import topLevelAwait from "vite-plugin-top-level-await";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd());

  const simplexTarget = env.VITE_SIMPLEX_TARGET || "http://localhost:11236";
  
  // proxy config for half-duplex (11236)
  const proxyConf: Record<string, string | ProxyOptions> = {
    "/api/simplex": {
      target: simplexTarget,
      changeOrigin: true,
      ws: true,
      secure: false,
      rewrite: (path) => path.replace(/^\/api\/simplex/, '/api/chat'),
      configure: (proxy, _options) => {
        proxy.on('error', (err, _req, _res) => {
          console.log('proxy error (simplex)', err);
        });
        proxy.on('proxyReq', (proxyReq, req, _res) => {
          console.log('Sending Request to Simplex:', req.method, req.url);
        });
        proxy.on('proxyRes', (proxyRes, req, _res) => {
          console.log('Received Response from Simplex:', proxyRes.statusCode, req.url);
        });
      },
    },
  };
  
  return {
    server: {
      host: "0.0.0.0",
      port: 80,
      https: {
        cert: "./cert.pem",
        key: "./key.pem",
      },
      proxy: {
        ...proxyConf,
      }
    },
    plugins: [
      topLevelAwait({
        // The export name of top-level await promise for each chunk module
        promiseExportName: "__tla",
        // The function to generate import names of top-level await promise in each chunk module
        promiseImportName: i => `__tla_${i}`,
      }),
    ],
  };
});
