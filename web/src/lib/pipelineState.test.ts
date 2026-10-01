import { describe, expect, it } from "vitest";
import type { Counters, PipelineEvent, PipelineState } from "../types";
import { applyEvent, initialView, MAX_LOG, withState } from "./pipelineState";

const COUNTERS: Counters = {
  requests: { network: 0, cache: 0 }, seasons: { done: 0, of: 15 }, validated: null, folds: { done: 0, of: 15 },
  ablation: { done: 0, of: 7 }, narratives: { done: 0, of: 11 }, players: { done: 0, of: 0 },
};
const STATE: PipelineState = {
  running: false, run_id: "b-r1", boot: "b", counters: COUNTERS,
  stages: ["fetch", "build"].map((name) => ({ name, status: "done" as const, started: 1, finished: 2, message: "" })),
};

function ev(id: string, body: Omit<PipelineEvent, "id" | "ts" | "stage"> & { stage?: string | null }): PipelineEvent {
  return { id, ts: 1, stage: null, ...body } as PipelineEvent;
}

describe("pipeline view", () => {
  it("logs lines once, even when an event is replayed", () => {
    const line = ev("b-1", { type: "log", run_id: "b-r2", data: { level: "info", text: "hello" } });
    const view = applyEvent(applyEvent(withState(initialView(), STATE), line), line);
    expect(view.log.map((l) => l.text)).toEqual(["hello"]);
  });
  it("starts a new run from its first event and resets the other stages", () => {
    let view = withState(initialView(), STATE);
    view = applyEvent(view, ev("b-2", { type: "stage", run_id: "b-r2",
      data: { name: "fetch", status: "running", started: 5, finished: null, message: "" } }));
    expect(view.state?.running).toBe(true);
    expect(view.state?.stages.map((s) => s.status)).toEqual(["running", "idle"]);
  });
  it("replaces counters on progress and records how a run ended", () => {
    let view = withState(initialView(), STATE);
    view = applyEvent(view, ev("b-3", { type: "progress", run_id: "b-r2",
      data: { counters: { ...COUNTERS, seasons: { done: 4, of: 15 } } } }));
    expect(view.state?.counters.seasons.done).toBe(4);
    view = applyEvent(view, ev("b-4", { type: "run_failed", run_id: "b-r2", stage: "build",
      data: { stage: "build", errors: ["2016 Roma: x"] } }));
    expect(view.outcome).toEqual({ status: "failed", errors: ["2016 Roma: x"], runId: "b-r2" });
    expect(view.state?.running).toBe(false);
    expect(view.finished).toBe(1);
  });
  it("ignores an earlier run's replayed events except for the log", () => {
    let view = withState(initialView(), { ...STATE, run_id: "b-r3" });
    view = applyEvent(view, ev("b-5", { type: "done", run_id: "b-r2", data: { status: "done", stages: [] } }));
    view = applyEvent(view, ev("b-6", { type: "log", run_id: "b-r2", data: { level: "warn", text: "old" } }));
    expect(view.outcome).toBeNull();
    expect(view.log.map((l) => l.text)).toEqual(["old"]);
  });
  it("starts over when the server restarted", () => {
    let view = withState(initialView(), STATE);
    view = applyEvent(view, ev("b-7", { type: "log", run_id: null, data: { level: "info", text: "before" } }));
    view = applyEvent(view, ev("c-1", { type: "log", run_id: null, data: { level: "info", text: "after" } }));
    expect(view.boot).toBe("c");
    expect(view.state).toBeNull();
    expect(view.log.map((l) => l.text)).toEqual(["after"]);
  });
  it("keeps the log bounded", () => {
    let view = withState(initialView(), STATE);
    for (let i = 1; i <= MAX_LOG + 5; i++) {
      view = applyEvent(view, ev(`b-${i}`, { type: "log", run_id: null, data: { level: "info", text: String(i) } }));
    }
    expect(view.log).toHaveLength(MAX_LOG);
    expect(view.log[0].text).toBe("6");
  });
});
