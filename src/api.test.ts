import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { coreApi } from "./api";

describe("search service request diagnostics", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.stubGlobal("window", globalThis);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("preserves HTTP conflicts as application errors, not disconnections", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ error: "索引任务正在运行" }), { status: 409 },
    )));
    await expect(coreApi.index([])).rejects.toMatchObject({
      kind: "http", path: "/index", message: "索引任务正在运行",
    });
    expect(vi.getTimerCount()).toBe(0);
  });

  it("identifies a failed connection", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    await expect(coreApi.stats()).rejects.toMatchObject({ kind: "network", path: "/stats" });
    expect(vi.getTimerCount()).toBe(0);
  });

  it("identifies malformed JSON separately", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("not JSON")));
    await expect(coreApi.stats()).rejects.toMatchObject({ kind: "invalid_response" });
    expect(vi.getTimerCount()).toBe(0);
  });

  it("does not call an interrupted body an invalid JSON response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true, json: () => Promise.reject(new TypeError("Connection interrupted")),
    }));
    await expect(coreApi.stats()).rejects.toMatchObject({ kind: "network" });
  });

  it.each([true, false])("times out a stalled response body (HTTP success: %s)", async (ok) => {
    vi.stubGlobal("fetch", vi.fn().mockImplementation((_url, options: RequestInit) => Promise.resolve({
      ok,
      json: () => new Promise((_resolve, reject) => {
        options.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
      }),
    })));
    const result = expect(coreApi.stats()).rejects.toMatchObject({ kind: "timeout", path: "/stats" });
    await vi.advanceTimersByTimeAsync(5_000);
    await result;
    expect(vi.getTimerCount()).toBe(0);
  });
});
