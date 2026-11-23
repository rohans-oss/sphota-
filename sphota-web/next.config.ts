import type { NextConfig } from "next";

// The Python service (python -m truesight.app.api). The browser only ever talks to this origin, so images, the .sphota download
// and the event stream need no CORS and work from any host that serves the site.
const API = process.env.SPHOTA_API ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
  },
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
