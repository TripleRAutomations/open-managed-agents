import { describe, it, expect } from "vitest";
import { isEmptyFinalStep } from "../../apps/agent/src/harness/default-loop";

describe("isEmptyFinalStep", () => {
  it("fires when the last step produced nothing, even after earlier tool calls", () => {
    // sess-45k053fxvunuoy42: claim + knowledge reads + 20 get_offense calls,
    // then an empty model response. The aggregate has 22 tool calls.
    const steps = [
      { text: "", toolCalls: [{ toolName: "claim_deep_queue" }] },
      { text: "", toolCalls: new Array(20).fill({ toolName: "get_offense" }) },
      { text: "", toolCalls: [] },
    ];
    expect(isEmptyFinalStep(steps, "", steps.flatMap((s) => s.toolCalls))).toBe(true);
  });

  it("does not fire when the last step called a tool", () => {
    const steps = [{ text: "", toolCalls: [{ toolName: "run_select" }] }];
    expect(isEmptyFinalStep(steps, "", [{ toolName: "run_select" }])).toBe(false);
  });

  it("does not fire when the last step produced text", () => {
    const steps = [
      { text: "", toolCalls: [{ toolName: "run_select" }] },
      { text: "Pass complete: 12 clusters dispositioned.", toolCalls: [] },
    ];
    expect(isEmptyFinalStep(steps, "Pass complete: 12 clusters dispositioned.", [])).toBe(false);
  });

  it("falls back to the aggregate when steps are unavailable", () => {
    expect(isEmptyFinalStep(undefined, "", [])).toBe(true);
    expect(isEmptyFinalStep(undefined, "done", [])).toBe(false);
    expect(isEmptyFinalStep([], "", [{ toolName: "x" }])).toBe(false);
  });
});
