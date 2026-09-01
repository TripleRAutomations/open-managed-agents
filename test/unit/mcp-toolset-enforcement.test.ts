// Unit tests for mcpToolEnabled (apps/agent/src/harness/tools.ts) — the
// filter that makes an agent's mcp_toolset config actually remove remote MCP
// tools. Found in production shape: an MCP server advertises its full tool
// list to every caller, and before this filter a per-tool
// { enabled: false } was purely decorative — a "read-only" reviewer agent
// still received (and used) update_offense.

import { describe, it, expect } from "vitest";
import { mcpToolEnabled } from "../../apps/agent/src/harness/tools";
import type { AgentConfig } from "@open-managed-agents/api-types";

function agentWith(tools: unknown[]): AgentConfig {
  return { tools } as unknown as AgentConfig;
}

const QRADAR_LOCKED = {
  type: "mcp_toolset",
  mcp_server_name: "qradar",
  default_config: { enabled: true, permission_policy: { type: "always_allow" } },
  configs: [{ name: "update_offense", enabled: false, permission_policy: { type: "always_allow" } }],
};

describe("mcpToolEnabled", () => {
  it("removes a per-tool disabled MCP tool", () => {
    const a = agentWith([QRADAR_LOCKED]);
    expect(mcpToolEnabled(a, "qradar", "update_offense")).toBe(false);
  });

  it("keeps other tools of the same server enabled", () => {
    const a = agentWith([QRADAR_LOCKED]);
    expect(mcpToolEnabled(a, "qradar", "get_offense")).toBe(true);
  });

  it("does not affect other servers", () => {
    const a = agentWith([QRADAR_LOCKED]);
    expect(mcpToolEnabled(a, "soc-pg", "set_offense_verdict")).toBe(true);
  });

  it("respects default_config.enabled=false for unlisted tools", () => {
    const a = agentWith([
      {
        type: "mcp_toolset",
        mcp_server_name: "qradar",
        default_config: { enabled: false },
        configs: [{ name: "get_offense", enabled: true }],
      },
    ]);
    expect(mcpToolEnabled(a, "qradar", "update_offense")).toBe(false);
    expect(mcpToolEnabled(a, "qradar", "get_offense")).toBe(true);
  });

  it("fully enables a server with no mcp_toolset entry", () => {
    const a = agentWith([{ type: "agent_toolset_20260401", default_config: { enabled: true } }]);
    expect(mcpToolEnabled(a, "qradar", "update_offense")).toBe(true);
  });

  it("handles an agent with no tools at all", () => {
    expect(mcpToolEnabled(agentWith([]), "qradar", "anything")).toBe(true);
  });
});
