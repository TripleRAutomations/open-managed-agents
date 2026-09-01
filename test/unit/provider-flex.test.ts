// Unit tests for the OpenAI flex-tier fetch wrapper (createOaiFetch in
// apps/agent/src/harness/provider.ts). Drives the wrapper directly with a
// mocked globalThis.fetch, covering:
//
//   1. Non-flex path: max_tokens capped when unset, no service_tier injected,
//      an existing max_tokens is left alone.
//   2. Flex happy path: service_tier:"flex" injected, single upstream call.
//   3. Capacity 429: flex attempt rejected → automatic retry WITHOUT
//      service_tier, and the failure arms a cooldown so the next call goes
//      straight to the standard tier.
//   4. Schedule timeout: upstream never responds within scheduleMs → the
//      attempt is aborted and retried on the standard tier.
//   5. Genuine (non-capacity) 429 passes through untouched so the SDK's own
//      backoff applies.
//
// What this does NOT prove: that OpenAI/OpenRouter actually honor the flex
// tier or bill it at 50% — that is verified against the live gateway.

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { createOaiFetch, _resetFlexCooldown } from "../../apps/agent/src/harness/provider";

const URL_ = "https://openrouter.ai/api/v1/chat/completions";

function req(body: Record<string, unknown>): RequestInit {
  return { method: "POST", body: JSON.stringify(body) };
}

function okResponse(): Response {
  return new Response(JSON.stringify({ choices: [] }), { status: 200 });
}

let calls: Array<Record<string, unknown>>;
const realFetch = globalThis.fetch;

beforeEach(() => {
  calls = [];
  _resetFlexCooldown();
});

afterEach(() => {
  globalThis.fetch = realFetch;
});

function mockFetch(handler: (body: Record<string, unknown>, init?: RequestInit) => Response | Promise<Response>) {
  globalThis.fetch = vi.fn(async (_url: RequestInfo | URL, init?: RequestInit) => {
    const body = JSON.parse((init?.body as string) ?? "{}") as Record<string, unknown>;
    calls.push(body);
    return handler(body, init);
  }) as unknown as typeof fetch;
}

describe("createOaiFetch — standard tier", () => {
  it("caps max_tokens when unset and never injects service_tier", async () => {
    mockFetch(() => okResponse());
    const f = createOaiFetch(false);
    const res = await f(URL_, req({ model: "gpt-5.6-terra", messages: [] }));
    expect(res.status).toBe(200);
    expect(calls).toHaveLength(1);
    expect(calls[0].max_tokens).toBe(16384);
    expect(calls[0].service_tier).toBeUndefined();
  });

  it("leaves an explicit max_tokens alone", async () => {
    mockFetch(() => okResponse());
    const f = createOaiFetch(false);
    await f(URL_, req({ model: "m", max_tokens: 512 }));
    expect(calls[0].max_tokens).toBe(512);
  });
});

describe("createOaiFetch — flex tier", () => {
  it("injects service_tier:flex and keeps the max_tokens cap", async () => {
    mockFetch(() => okResponse());
    const f = createOaiFetch(true);
    const res = await f(URL_, req({ model: "gpt-5.6-terra", messages: [] }));
    expect(res.status).toBe(200);
    expect(calls).toHaveLength(1);
    expect(calls[0].service_tier).toBe("flex");
    expect(calls[0].max_tokens).toBe(16384);
  });

  it("capacity 429 → retries on standard tier and arms the cooldown", async () => {
    mockFetch((body) => {
      if (body.service_tier === "flex") {
        return new Response(
          JSON.stringify({ error: { message: "Resource unavailable for the flex service tier" } }),
          { status: 429 },
        );
      }
      return okResponse();
    });
    const f = createOaiFetch(true);
    const res = await f(URL_, req({ model: "m", messages: [] }));
    expect(res.status).toBe(200);
    expect(calls).toHaveLength(2);
    expect(calls[0].service_tier).toBe("flex");
    expect(calls[1].service_tier).toBeUndefined();

    // Cooldown armed: the next call skips flex entirely.
    calls = [];
    const res2 = await f(URL_, req({ model: "m", messages: [] }));
    expect(res2.status).toBe(200);
    expect(calls).toHaveLength(1);
    expect(calls[0].service_tier).toBeUndefined();
  });

  it("schedule timeout → aborts the flex attempt and falls back", async () => {
    mockFetch((body, init) => {
      if (body.service_tier === "flex") {
        // Never respond; resolve only when the wrapper aborts us.
        return new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => reject(new Error("aborted")));
        });
      }
      return okResponse();
    });
    const f = createOaiFetch(true, { scheduleMs: 30 });
    const res = await f(URL_, req({ model: "m", messages: [] }));
    expect(res.status).toBe(200);
    expect(calls).toHaveLength(2);
    expect(calls[0].service_tier).toBe("flex");
    expect(calls[1].service_tier).toBeUndefined();
  });

  it("passes a genuine rate-limit 429 through for SDK backoff", async () => {
    mockFetch(() =>
      new Response(JSON.stringify({ error: { message: "Rate limit exceeded: free tier" } }), {
        status: 429,
      }),
    );
    const f = createOaiFetch(true);
    const res = await f(URL_, req({ model: "m", messages: [] }));
    expect(res.status).toBe(429);
    expect(calls).toHaveLength(1);
  });
});

describe("createOaiFetch — 200-with-error envelope", () => {
  it("falls back to standard when flex returns HTTP 200 carrying a rate-limit error body", async () => {
    mockFetch((body) => {
      if (body.service_tier === "flex") {
        return new Response(
          JSON.stringify({ error: { code: 429, message: "model is temporarily rate-limited upstream. Please retry shortly" } }),
          { status: 200, headers: { "content-type": "application/json" } },
        );
      }
      return okResponse();
    });
    const f = createOaiFetch(true);
    const res = await f(URL_, req({ model: "m", messages: [] }));
    expect(res.status).toBe(200);
    expect(calls).toHaveLength(2);
    expect(calls[0].service_tier).toBe("flex");
    expect(calls[1].service_tier).toBeUndefined();
  });

  it("does not misfire on a normal JSON completion", async () => {
    mockFetch(() =>
      new Response(JSON.stringify({ choices: [{ message: { content: "hello, no rate limit here" } }] }), {
        status: 200, headers: { "content-type": "application/json" },
      }),
    );
    const f = createOaiFetch(true);
    const res = await f(URL_, req({ model: "m", messages: [] }));
    expect(res.status).toBe(200);
    expect(calls).toHaveLength(1);
  });
});
