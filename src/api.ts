import type {
  IndexFailure,
  IndexedChunk,
  IndexedDocument,
  FileIndexTiming,
  Page,
  SearchMode,
  SearchResponse,
  ServiceStats,
} from "./types";

const CORE_URL = "http://127.0.0.1:47653";

export class CoreRequestError extends Error {
  constructor(public kind: "timeout" | "network" | "http" | "invalid_response", public path: string, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit, timeoutMs = 15_000): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${CORE_URL}${path}`, {
      ...init,
      signal: controller.signal,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
    if (!response.ok) {
      const body = (await response.json().catch((error) => {
        if (controller.signal.aborted) throw error;
        return {};
      })) as { error?: string; message?: string };
      throw new CoreRequestError("http", path, body.error ?? body.message ?? `HTTP ${response.status}`);
    }
    try {
      return await response.json() as T;
    } catch (error) {
      if (controller.signal.aborted || !(error instanceof SyntaxError)) throw error;
      throw new CoreRequestError("invalid_response", path, `Invalid JSON response: ${path}`);
    }
  } catch (error) {
    if (error instanceof CoreRequestError) throw error;
    const detail = error instanceof Error ? error.message : String(error);
    throw new CoreRequestError(controller.signal.aborted ? "timeout" : "network", path, `${path}: ${detail}`);
  } finally {
    window.clearTimeout(timer);
  }
}

export const coreApi = {
  health: () => request<{ status: string }>("/health"),
  stats: () => request<ServiceStats>("/stats", undefined, 5_000),
  index: (paths: string[]) =>
    request<{ accepted: boolean; message: string }>("/index", {
      method: "POST",
      body: JSON.stringify({ paths }),
    }),
  pauseIndex: () => request<{ accepted: boolean; message: string }>("/index/pause", { method: "POST" }),
  resumeIndex: () => request<{ accepted: boolean; message: string }>("/index/resume", { method: "POST" }),
  cancelIndex: () => request<{ accepted: boolean; message: string }>("/index/cancel", { method: "POST" }),
  search: (query: string, mode: SearchMode, extension: string) =>
    request<SearchResponse>("/search", {
      method: "POST",
      body: JSON.stringify({ query, mode, extension: extension || null, limit: 50 }),
    }, 120_000),
  documents: (offset = 0, limit = 50) => request<Page<IndexedDocument>>(`/documents?offset=${offset}&limit=${limit}`),
  chunks: (offset = 0, limit = 50) => request<Page<IndexedChunk>>(`/chunks?offset=${offset}&limit=${limit}`),
  failures: (offset = 0, limit = 50) => request<Page<IndexFailure>>(`/failures?offset=${offset}&limit=${limit}`),
  fileTimings: (offset = 0, limit = 100) => request<Page<FileIndexTiming>>(`/file-timings?offset=${offset}&limit=${limit}`),
};
