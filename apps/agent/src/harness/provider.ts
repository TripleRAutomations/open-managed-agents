import { createAnthropic } from "@ai-sdk/anthropic";
import { createOpenAI } from "@ai-sdk/openai";
import type { LanguageModel } from "ai";

/**
 * API compatibility types:
 * - "ant"            — Anthropic official API
 * - "ant-compatible" — Third-party Anthropic-compatible API
 * - "oai"            — OpenAI official API
 * - "oai-compatible" — Third-party OpenAI-compatible API (DeepSeek, Groq, etc.)
 */
export type ApiCompat = "ant" | "ant-compatible" | "oai" | "oai-compatible";

const KNOWN_CLAUDE_PREFIX = "claude-";

// Cap for non-Claude models on the Anthropic-compat path. The SDK hard-codes
// max_tokens=4096 for unknown models, which truncates extended thinking
// (MiniMax-M2 thinking alone exceeds that). Earlier code deleted the field
// entirely, but the Anthropic spec marks it required — DeepSeek's strict
// (Rust serde) implementation rejects with `missing field max_tokens` and a
// generic 400 that surfaces as `Bad Request` upstream. Setting a high value
// satisfies the spec and gives every provider room for thinking + tool_use.
const NON_CLAUDE_MAX_TOKENS = 32768;

/**
 * Fetch wrapper that overrides @ai-sdk/anthropic's hard-coded max_tokens=4096
 * with NON_CLAUDE_MAX_TOKENS for non-Claude models on the Anthropic-compat
 * path.
 */
async function setMaxTokensFetch(url: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const finalInit = (() => {
    if (init?.body && typeof init.body === "string") {
      try {
        const body = JSON.parse(init.body);
        body.max_tokens = NON_CLAUDE_MAX_TOKENS;
        return { ...init, body: JSON.stringify(body) };
      } catch {
        return init;
      }
    }
    return init;
  })();
  return observingFetch(url, finalInit);
}

/**
 * Wraps globalThis.fetch with always-on observability for provider rate
 * limiting. Logs (via console) + surfaces:
 *  - HTTP status code (so 429 is visible immediately)
 *  - retry-after header (if present)
 *  - x-ratelimit-* headers (any provider that exposes them)
 *  - response body preview when status >= 400 (truncated)
 *
 * Without this we only see indirect signals (model_first_token + no
 * model_request_end → "stalled stream"), which conflates rate limiting
 * with real model slowness, network issues, or provider hangs.
 *
 * timeoutMs is a hard cap on the whole HTTP exchange (including streaming
 * body). Without it a silent provider stream hangs the SessionDO
 * indefinitely. Flex-tier attempts pass a longer cap — see createOaiFetch.
 */
async function observingFetch(
  url: RequestInfo | URL,
  init?: RequestInit,
  timeoutMs = 5 * 60_000,
): Promise<Response> {
  const startedAt = Date.now();
  const method = init?.method ?? "GET";
  const urlStr = typeof url === "string" ? url : url instanceof URL ? url.toString() : url.url;
  const signal = init?.signal
    ? AbortSignal.any([init.signal, AbortSignal.timeout(timeoutMs)])
    : AbortSignal.timeout(timeoutMs);
  let res: Response;
  try {
    res = await globalThis.fetch(url, { ...init, signal });
  } catch (err) {
    const elapsed = Date.now() - startedAt;
    console.warn(`[provider.fetch] ${method} ${urlStr} → THROW after ${elapsed}ms: ${err instanceof Error ? err.message : String(err)}`);
    throw err;
  }
  const elapsed = Date.now() - startedAt;
  const status = res.status;
  // Collect rate-limit signals from common header names across providers.
  const retryAfter = res.headers.get("retry-after");
  const limitRemaining =
    res.headers.get("x-ratelimit-remaining-requests") ??
    res.headers.get("x-ratelimit-remaining-tokens") ??
    res.headers.get("x-ratelimit-remaining");
  const limitReset =
    res.headers.get("x-ratelimit-reset-requests") ??
    res.headers.get("x-ratelimit-reset-tokens") ??
    res.headers.get("x-ratelimit-reset");
  const interesting = status >= 400 || retryAfter || (limitRemaining && parseInt(limitRemaining, 10) < 5);
  if (interesting) {
    let bodyPreview = "";
    if (status >= 400) {
      try {
        bodyPreview = (await res.clone().text()).slice(0, 500);
      } catch {}
    }
    console.warn(
      `[provider.fetch] ${method} ${urlStr} → ${status} (${elapsed}ms)` +
        (retryAfter ? ` retry-after=${retryAfter}` : "") +
        (limitRemaining ? ` remaining=${limitRemaining}` : "") +
        (limitReset ? ` reset=${limitReset}` : "") +
        (bodyPreview ? ` body=${JSON.stringify(bodyPreview)}` : ""),
    );
  } else if (status >= 200 && status < 300 && elapsed > 5000) {
    // Slow OK response — useful for diagnosing per-call latency
    console.log(`[provider.fetch] ${method} ${urlStr} → ${status} (${elapsed}ms slow)`);
  }
  return res;
}

function useOpenAI(compat: ApiCompat): boolean {
  return compat === "oai" || compat === "oai-compatible";
}

// --- OpenAI flex service tier -----------------------------------------------
//
// Opt-in per model card by suffixing the card's `model` string with ":flex"
// (e.g. "openai/gpt-5.6-terra:flex"). The suffix is OMA-local: it is stripped
// before any request, so the provider only ever sees the clean model id.
// Flex halves OpenAI token pricing on synchronous chat/completions (and
// passes through gateways like OpenRouter), at the cost of queueing: under
// load a flex request waits server-side before it is scheduled, and may be
// rejected with an uncharged capacity 429 ("resource unavailable").
//
// Failsafe: every flex attempt automatically falls back to the standard tier
// when it is not scheduled in time, errors, or returns a capacity 429 — the
// caller (the AI SDK loop) just sees a normal response. After a flex failure
// we skip flex entirely for a cooldown window so a long agent loop doesn't
// pay the schedule timeout on every turn.
const FLEX_MODEL_SUFFIX = ":flex";
// Response headers arrive when the provider starts processing, so the wait
// for headers ≈ the flex queue time. Give up and fall back after this long.
const FLEX_SCHEDULE_TIMEOUT_MS = 3 * 60_000;
// Cap on the whole scheduled flex exchange (queue + stream) — flex also
// streams slower than standard, so the default 5-min guard is too tight.
const FLEX_TOTAL_TIMEOUT_MS = 10 * 60_000;
const FLEX_FAILURE_COOLDOWN_MS = 10 * 60_000;
// How long to wait for the first SSE data event when sniffing for an
// in-stream rate-limit error. Error events arrive within ~3s; a healthy flex
// stream may legitimately take longer to start (queueing) — then we stop
// peeking and pass the stream through untouched.
const FLEX_SSE_PEEK_MS = 8_000;

// Isolate-wide, deliberately: one flex outage should pause flex for every
// concurrent session in this worker, not be rediscovered per session.
let flexCooldownUntil = 0;

/** Test hook: clear the flex failure cooldown. */
export function _resetFlexCooldown(): void {
  flexCooldownUntil = 0;
}

interface FlexTimeouts {
  scheduleMs?: number;
  totalMs?: number;
  cooldownMs?: number;
  peekMs?: number;
}

/**
 * Fetch wrapper for the OpenAI-compat path. Always: caps max_tokens when the
 * harness didn't set one. Two reasons:
 *  (1) gateways that reserve credits per in-flight request (OpenRouter)
 *      size the reservation from worst-case output — uncapped 1M-context
 *      reasoning models reserve dollars per call and starve concurrent
 *      sessions into 402s long before actual spend reaches the balance;
 *  (2) a runaway generation is bounded. 16k is ample for tool-call turns
 *      and final reports.
 * With flexTier: attempts service_tier:"flex" first, falling back to the
 * standard tier as described above.
 */
export function createOaiFetch(flexTier: boolean, timeouts: FlexTimeouts = {}) {
  const scheduleMs = timeouts.scheduleMs ?? FLEX_SCHEDULE_TIMEOUT_MS;
  const totalMs = timeouts.totalMs ?? FLEX_TOTAL_TIMEOUT_MS;
  const cooldownMs = timeouts.cooldownMs ?? FLEX_FAILURE_COOLDOWN_MS;
  const peekMs = timeouts.peekMs ?? FLEX_SSE_PEEK_MS;

  return async (url: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    let body: Record<string, unknown> | null = null;
    if (init?.body && typeof init.body === "string") {
      try {
        body = JSON.parse(init.body) as Record<string, unknown>;
      } catch {
        /* non-JSON body — leave untouched */
      }
    }
    if (body && body.max_tokens == null && body.max_completion_tokens == null) {
      body.max_tokens = 16384;
    }
    if (body) init = { ...init, body: JSON.stringify(body) };

    if (!flexTier || !body || Date.now() < flexCooldownUntil) {
      return observingFetch(url, init);
    }

    const ctrl = new AbortController();
    const flexSignal = init?.signal ? AbortSignal.any([init.signal, ctrl.signal]) : ctrl.signal;
    const attempt = observingFetch(
      url,
      { ...init, body: JSON.stringify({ ...body, service_tier: "flex" }), signal: flexSignal },
      totalMs,
    );
    let scheduleTimer: ReturnType<typeof setTimeout> | undefined;
    const TIMED_OUT = Symbol("flex-schedule-timeout");
    const outcome = await Promise.race([
      attempt.then(
        (res) => ({ res }),
        (err: unknown) => ({ err: err instanceof Error ? err : new Error(String(err)) }),
      ),
      new Promise<typeof TIMED_OUT>((resolve) => {
        scheduleTimer = setTimeout(() => resolve(TIMED_OUT), scheduleMs);
      }),
    ]);
    clearTimeout(scheduleTimer);

    if (outcome === TIMED_OUT) {
      ctrl.abort();
      attempt.catch(() => {});
      flexCooldownUntil = Date.now() + cooldownMs;
      console.warn(
        `[provider.flex] not scheduled within ${scheduleMs}ms — falling back to standard tier (flex paused ${cooldownMs}ms)`,
      );
      return observingFetch(url, init);
    }
    if ("err" in outcome) {
      // The caller itself aborted — propagate instead of burning a retry.
      if (init?.signal?.aborted) throw outcome.err;
      flexCooldownUntil = Date.now() + cooldownMs;
      console.warn(
        `[provider.flex] attempt failed (${outcome.err.message}) — falling back to standard tier`,
      );
      return observingFetch(url, init);
    }
    const res = outcome.res;
    if (res.status === 429) {
      // Distinguish flex capacity rejection (uncharged; retry on standard is
      // the documented remedy) from a genuine rate limit (pass through so the
      // SDK's backoff applies).
      let preview = "";
      try {
        preview = (await res.clone().text()).slice(0, 300);
      } catch {}
      if (/resource[\s_]?unavailable|capacity|service[\s_]?tier/i.test(preview)) {
        try {
          await res.body?.cancel();
        } catch {}
        flexCooldownUntil = Date.now() + cooldownMs;
        console.warn("[provider.flex] capacity 429 — falling back to standard tier");
        return observingFetch(url, init);
      }
    }
    // Gateways (OpenRouter) can also signal upstream rate limiting as
    // HTTP 200 with a JSON error envelope instead of an SSE stream —
    // {"error":{"code":429,"message":"...rate-limited upstream..."}}.
    // The SDK parses that as an EMPTY stream: zero tokens, no exception,
    // and the agent turn silently produces nothing. Detect the envelope
    // and retry on the standard tier.
    const ct = res.headers.get("content-type") ?? "";
    if (res.ok && ct.includes("application/json")) {
      let preview = "";
      try {
        preview = (await res.clone().text()).slice(0, 500);
      } catch {}
      if (/"error"/.test(preview) && /429|rate.?limit/i.test(preview)) {
        try {
          await res.body?.cancel();
        } catch {}
        flexCooldownUntil = Date.now() + cooldownMs;
        console.warn("[provider.flex] 200-with-error envelope (upstream rate limit) — falling back to standard tier");
        return observingFetch(url, init);
      }
    }
    // SSE variant of the same failure: OpenRouter can also deliver the
    // upstream rate-limit as the FIRST (and only) event of a text/event-stream
    // response — {"choices":[],"error":{"code":429,...}} then [DONE]. The SDK
    // ends the stream with finish_reason "other" and zero output. Peek at the
    // first data event before handing the stream to the SDK.
    const sse = res.ok && ct.includes("text/event-stream") && res.body;
    if (sse) {
      const [peek, pass] = res.body!.tee();
      const verdict = await peekSseForError(peek, peekMs);
      if (verdict === "rate_limited") {
        try { await pass.cancel(); } catch {}
        flexCooldownUntil = Date.now() + cooldownMs;
        console.warn("[provider.flex] SSE error event (upstream rate limit) — falling back to standard tier");
        return observingFetch(url, init);
      }
      return new Response(pass, { status: res.status, statusText: res.statusText, headers: res.headers });
    }
    return res;
  };
}

/**
 * Read the first SSE data event (skipping ": OPENROUTER PROCESSING" keepalive
 * comments) within `maxMs`. Returns "rate_limited" when it is an error event
 * carrying a 429 / rate-limit message, "ok" for a normal chunk, and
 * "unknown" when nothing arrived in time (genuine flex queueing — let the
 * stream through untouched).
 */
export async function peekSseForError(
  stream: ReadableStream<Uint8Array>,
  maxMs: number,
): Promise<"rate_limited" | "ok" | "unknown"> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  const deadline = Date.now() + maxMs;
  try {
    while (Date.now() < deadline) {
      const remaining = deadline - Date.now();
      const next = await Promise.race([
        reader.read(),
        new Promise<null>((r) => setTimeout(() => r(null), remaining)),
      ]);
      if (next === null) break;
      if (next.done) break;
      buf += decoder.decode(next.value, { stream: true });
      const dataLine = buf.split("\n").find((l) => l.startsWith("data:") && l.trim() !== "data: [DONE]");
      if (dataLine) {
        const payload = dataLine.slice(5).trim();
        if (/"error"\s*:/.test(payload) && /429|rate.?limit/i.test(payload)) return "rate_limited";
        return "ok";
      }
      if (buf.length > 64_000) return "ok";
    }
  } catch {
    /* treat as unknown */
  } finally {
    try { reader.releaseLock(); } catch {}
  }
  return "unknown";
}

export function resolveModel(
  model: string | { id: string; speed?: "standard" | "fast" },
  apiKey: string,
  baseURL?: string,
  compat?: ApiCompat,
  customHeaders?: Record<string, string>,
): LanguageModel {
  const modelString = typeof model === "string" ? model : model.id;

  // Strip provider prefix if present: "anthropic/claude-sonnet-4-6" → "claude-sonnet-4-6"
  const modelId = modelString.includes("/")
    ? modelString.split("/").slice(1).join("/")
    : modelString;

  const effectiveCompat = compat || "ant";

  if (useOpenAI(effectiveCompat)) {
    let oaiModelId = modelId;
    let flexTier = false;
    if (oaiModelId.endsWith(FLEX_MODEL_SUFFIX)) {
      flexTier = true;
      oaiModelId = oaiModelId.slice(0, -FLEX_MODEL_SUFFIX.length);
    }
    const openai = createOpenAI({
      apiKey,
      baseURL: baseURL || undefined,
      headers: customHeaders,
      fetch: createOaiFetch(flexTier),
    });
    // Use chat/completions endpoint, not Responses API.
    // Reasons:
    //   - Third-party OpenAI-compat gateways (CF AI Gateway, Groq, DeepSeek,
    //     xAI Grok, etc.) only support /v1/chat/completions
    //   - Responses API requires server-side persistence of function call IDs;
    //     orgs with Zero Data Retention enabled get "Item with id 'fc_...' not
    //     found" errors mid-loop
    //   - chat/completions is the de-facto standard contract for OpenAI-compat
    return openai.chat(oaiModelId);
  }

  // ant / ant-compatible
  const isKnownClaude = modelId.startsWith(KNOWN_CLAUDE_PREFIX);

  const headers: Record<string, string> = {};
  if (baseURL) headers["X-Sub-Module"] = "managed-agents";
  if (customHeaders) Object.assign(headers, customHeaders);

  // @ai-sdk/anthropic appends `/messages` directly to baseURL — no `/v1`
  // segment is added. Real api.anthropic.com endpoints include `/v1` in the
  // SDK default, so deployments pointing at proxies must too. Auto-append
  // `/v1` if the user supplied a bare host so common env values work.
  const normalizedBaseURL = baseURL
    ? /\/v\d+(\/)?$/.test(baseURL)
      ? baseURL.replace(/\/$/, "")
      : `${baseURL.replace(/\/$/, "")}/v1`
    : undefined;

  const anthropic = createAnthropic({
    apiKey,
    baseURL: normalizedBaseURL,
    headers: Object.keys(headers).length > 0 ? headers : undefined,
    // setMaxTokensFetch composes observingFetch internally for non-Claude;
    // Claude path uses observingFetch directly so 429/rate-limit logging
    // applies regardless of which provider/model we're talking to.
    fetch: isKnownClaude ? observingFetch : setMaxTokensFetch,
  });

  return anthropic(modelId);
}
