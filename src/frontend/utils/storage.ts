const isBrowser = typeof window !== "undefined" && typeof window.localStorage !== "undefined";

export function readJson<T>(key: string): T | undefined {
  if (!isBrowser) {
    return undefined;
  }
  try {
    const value = window.localStorage.getItem(key);
    if (!value) {
      return undefined;
    }
    return JSON.parse(value) as T;
  } catch (error) {
    console.warn(`[storage] failed to read key "${key}":`, error);
    return undefined;
  }
}

export function writeJson<T>(key: string, value: T): void {
  if (!isBrowser) {
    return;
  }
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch (error) {
    console.warn(`[storage] failed to persist key "${key}":`, error);
  }
}

export function normaliseVoiceFilename(value: string): string {
  if (!value) {
    return value;
  }
  const parts = value.split("/");
  const name = parts[parts.length - 1] ?? value;
  if (name.endsWith(".onnx")) {
    return value;
  }
  const normalised = `${name}.onnx`;
  if (parts.length === 1) {
    return normalised;
  }
  parts[parts.length - 1] = normalised;
  return parts.join("/");
}
