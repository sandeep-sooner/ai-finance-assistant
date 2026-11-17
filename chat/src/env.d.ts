interface ImportMetaEnv {
  /** Backend origin, e.g. "https://api.example.com". Leave empty to use same-origin /api (dev proxy or FastAPI serving the build). */
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
