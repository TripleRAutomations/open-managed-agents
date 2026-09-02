// Unit tests for truncateMcpResult (apps/agent/src/harness/tools.ts): remote
// MCP results were unbounded; a ~1 MB run_select result (full knowledge
// table) caused D1 write timeouts and SessionDO evictions in production.
import { describe, it, expect } from "vitest";
import { truncateMcpResult } from "../../apps/agent/src/harness/tools";

describe("truncateMcpResult", () => {
  it("passes small results through untouched", () => {
    const r = { content: [{ type: "text", text: "hello" }] };
    expect(truncateMcpResult(r)).toEqual(r);
  });
  it("truncates an oversized text part and says so", () => {
    const big = "x".repeat(400_000);
    const out = truncateMcpResult({ content: [{ type: "text", text: big }] }) as { content: Array<{ text: string }> };
    expect(out.content[0].text.length).toBeLessThan(160_000);
    expect(out.content[0].text).toContain("truncated by the platform");
    expect(out.content[0].text).toContain("400000");
  });
  it("truncates plain string results", () => {
    const out = truncateMcpResult("y".repeat(200_000)) as string;
    expect(out.length).toBeLessThan(160_000);
  });
  it("leaves non-text parts alone", () => {
    const r = { content: [{ type: "image", data: "abc" }] };
    expect(truncateMcpResult(r)).toEqual(r);
  });
});
