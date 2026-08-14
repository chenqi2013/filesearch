import type { IndexFailure, SearchMode, SearchResponse, ServiceStats } from "./types";

const CORE_URL = "http://127.0.0.1:47653";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${CORE_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => ({}))) as { error?: string; message?: string };
    throw new Error(body.error ?? body.message ?? `HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const coreApi = {
  health: () => request<{ status: string }>("/health"),
  stats: () => request<ServiceStats>("/stats"),
  index: (paths: string[]) =>
    request<{ accepted: boolean; message: string }>("/index", {
      method: "POST",
      body: JSON.stringify({ paths }),
    }),
  search: (query: string, mode: SearchMode, extension: string) =>
    request<SearchResponse>("/search", {
      method: "POST",
      body: JSON.stringify({ query, mode, extension: extension || null, limit: 50 }),
    }),
  failures: async () => (await request<{ failures: IndexFailure[] }>("/failures")).failures,
};

