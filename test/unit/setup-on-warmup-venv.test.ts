// Unit tests for the restored-state venv verification in
// ensureSetupApplied (apps/agent/src/runtime/setup-on-warmup.ts). Found in
// production: a container restored from a workspace backup carried the
// RESTORED_MARKER but not /workspace/.venv, so the apt-only fast path ran,
// pip packages were never installed, and the agent had no weasyprint /
// markdown — the daily report PDF fell back to a hand-rolled renderer.

import { describe, it, expect } from "vitest";
import { ensureSetupApplied } from "../../apps/agent/src/runtime/setup-on-warmup";

const PKGS = { pip: ["weasyprint", "markdown"], apt: ["pandoc"] };

// Build an exec fake. `hasVenv` controls the venv probe; markers are set so
// probeSetupState lands on "restored". Records every command.
function fakeExec(opts: { hasVenv: boolean; restMarker: string }) {
  const commands: string[] = [];
  const exec = async (cmd: string): Promise<string> => {
    commands.push(cmd);
    if (cmd.includes("WARM=$(cat")) {
      return `exit=0\nWARM=MISSING\nREST=${opts.restMarker}`;
    }
    if (cmd.includes("/workspace/.venv/bin/python")) {
      return `exit=0\n${opts.hasVenv ? "VENV_OK" : "VENV_MISSING"}`;
    }
    return "exit=0\n";
  };
  return { exec, commands };
}

// The restored marker must equal langHash(PKGS); we don't export the hash,
// so capture it by running once with a probe that reports whatever marker
// the code writes at the end of a fresh run.
async function learnLangHash(): Promise<string> {
  let written = "";
  const exec = async (cmd: string): Promise<string> => {
    if (cmd.includes("WARM=$(cat")) return "exit=0\nWARM=MISSING\nREST=MISSING";
    const m = cmd.match(/echo "(\w+)" > \/workspace\/\.oma-setup-restored/);
    if (m) written = m[1];
    return "exit=0\n";
  };
  await ensureSetupApplied({ exec }, PKGS);
  return written;
}

describe("ensureSetupApplied restored-state venv verification", () => {
  it("takes the apt-only path when the venv survived the restore", async () => {
    const lang = await learnLangHash();
    const f = fakeExec({ hasVenv: true, restMarker: lang });
    const r = await ensureSetupApplied(f, PKGS);
    expect(r.path).toBe("restored");
    expect(f.commands.some((c) => c.includes("uv pip install"))).toBe(false);
  });

  it("falls back to full setup when the marker survived but the venv did not", async () => {
    const lang = await learnLangHash();
    const f = fakeExec({ hasVenv: false, restMarker: lang });
    const r = await ensureSetupApplied(f, PKGS);
    expect(r.path).toBe("fresh");
    expect(f.commands.some((c) => c.includes("uv pip install"))).toBe(true);
  });

  it("skips the venv probe when no pip packages are configured", async () => {
    const f = fakeExec({ hasVenv: false, restMarker: "anything" });
    const r = await ensureSetupApplied(f.exec ? f : f, { apt: ["pandoc"] });
    expect(f.commands.some((c) => c.includes("/workspace/.venv/bin/python"))).toBe(false);
    expect(r.path).toBe("fresh"); // marker mismatch -> fresh, but no venv probe ran
  });
});
