import { describe, expect, it } from "vitest";
import { summaryText } from "../components/Checks";
import { aiState } from "../hooks/useChecks";
import type { CheckItem, CheckReport } from "../types";

function report(items: Partial<CheckItem>[]): CheckReport {
  const checks = items.map((item, i) => ({ id: `c${i}`, group: "AI model", label: "Model loaded", status: "ok",
    detail: "Loaded", fix: "", ...item }) as CheckItem);
  const count = (s: string) => checks.filter((c) => c.status === s).length;
  return { checked_at: "2026-10-02T08:00:00Z", model: "m", provider: "lmstudio", checks,
    summary: { ok: count("ok"), warn: count("warn"), fail: count("fail"), info: count("info") } };
}

describe("checks", () => {
  it("sums up what passed and what needs attention", () => {
    expect(summaryText(report([{}, {}]))).toBe("2 passed");
    expect(summaryText(report([{}, { status: "warn" }, { status: "fail" }, { status: "fail" }])))
      .toBe("1 passed, 1 to look at, 2 problems");
  });
  it("tells the run controls whether the AI model will be ready", () => {
    expect(aiState(null).ready).toBe(true);
    expect(aiState(report([{}])).ready).toBe(true);
    const missing = aiState(report([{ status: "fail", label: "Model downloaded", detail: "m isn't downloaded" }]));
    expect(missing).toEqual({ ready: false, blocking: true, note: "Model downloaded: m isn't downloaded." });
    const notLoaded = aiState(report([{ status: "warn", detail: "Not loaded yet" }]));
    expect(notLoaded.blocking).toBe(false);
    expect(aiState(report([{ group: "UEFA feeds", status: "fail" }])).ready).toBe(true);
  });
});
