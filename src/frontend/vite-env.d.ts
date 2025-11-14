/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_DEFAULT_WS_URL?: string;
  readonly MODE: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
