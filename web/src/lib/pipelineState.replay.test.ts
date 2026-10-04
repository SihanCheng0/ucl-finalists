import { describe, expect, it } from "vitest";
import type { PipelineEvent, PipelineState } from "../types";
import { replay } from "./pipelineState";

const counters = { requests: { network: 0, cache: 0 }, seasons: { done: 0, of: 0 }, validated: null,
  folds: { done: 0, of: 0 }, ablation: { done: 0, of: 0 }, narratives: { done: 0, of: 0 }, players: { done: 0, of: 0 } };

describe("a recorded run", () => {
  it("replays into the view a dashboard would show after the run, without counting it as just finished", () => {
    const state: PipelineState = { running: false, run_id: "b-r1", boot: "b", counters,
      stages: [{ name: "live", status: "warning", started: 1, finished: 9, message: "UEFA was slow" }] };
    const base = { run_id: "b-r1", ts: 5 };
    const events: PipelineEvent[] = [
      { ...base, id: "b-1", stage: "live", type: "stage", data: { name: "live", status: "running", started: 1, finished: null, message: "" } },
      { ...base, id: "b-2", stage: "live", type: "log", data: { level: "info", text: "Fetching the 2026-27 season" } },
      { ...base, id: "b-3", stage: "live", type: "stage", data: state.stages[0] },
      { ...base, id: "b-4", stage: null, type: "done", data: { status: "warning", stages: [{ name: "live", status: "warning" }] } },
    ];
    const view = replay(state, events);
    expect(view.state?.running).toBe(false);
    expect(view.state?.stages[0].status).toBe("warning");
    expect(view.log.map((line) => line.text)).toEqual(["Fetching the 2026-27 season"]);
    expect(view.outcome).toEqual({ status: "warning", errors: [], runId: "b-r1" });
    expect(view.finished).toBe(0);
  });
});
