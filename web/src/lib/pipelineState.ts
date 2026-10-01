// The Pipeline screen's view of the runner: a /state snapshot plus the SSE events applied to it in order.
// Event ids are "{boot}-{n}": a new boot means the server restarted, so the view starts over (spec §4.3).
import type { LogLevel, PipelineEvent, PipelineState } from "../types";

export interface LogLine { id: string; ts: number; stage: string | null; level: LogLevel; text: string }
export interface Outcome { status: "done" | "warning" | "failed"; errors: string[]; runId: string | null }
export interface PipelineView {
  boot: string | null;
  lastN: number;
  state: PipelineState | null;
  log: LogLine[];
  outcome: Outcome | null;
  finished: number; // runs seen to end, so screens know when to reload their data
}

export const MAX_LOG = 2000;

export function initialView(): PipelineView {
  return { boot: null, lastN: 0, state: null, log: [], outcome: null, finished: 0 };
}

export function eventSeq(id: string): { boot: string; n: number } {
  const cut = id.lastIndexOf("-");
  return { boot: id.slice(0, cut), n: Number(id.slice(cut + 1)) };
}

function runSeq(runId: string | null): number {
  if (!runId) return 0;
  const n = Number(runId.slice(runId.lastIndexOf("-r") + 2));
  return Number.isFinite(n) ? n : 0;
}

/** A fresh /state snapshot is the truth for stages and counters. */
export function withState(view: PipelineView, state: PipelineState): PipelineView {
  if (view.boot !== null && view.boot !== state.boot) return { ...initialView(), boot: state.boot, state };
  return { ...view, boot: state.boot, state };
}

export function applyEvent(view: PipelineView, event: PipelineEvent): PipelineView {
  const { boot, n } = eventSeq(event.id);
  let next = view;
  if (view.boot !== null && boot !== view.boot) next = { ...initialView(), boot, finished: view.finished };
  else if (n <= view.lastN) return view; // already applied (a replay after reconnecting)
  next = { ...next, boot, lastN: n };

  if (event.type === "log") {
    const line = { id: event.id, ts: event.ts, stage: event.stage, level: event.data.level, text: event.data.text };
    const log = next.log.length >= MAX_LOG ? [...next.log.slice(1 - MAX_LOG), line] : [...next.log, line];
    return { ...next, log };
  }
  const state = next.state;
  if (!state) return next;
  const known = runSeq(state.run_id);
  const incoming = runSeq(event.run_id);
  if (event.run_id && incoming < known) return next; // an earlier run's replayed event
  // /state already says this run is over, so its events are a replay of the backlog: they fill in the stages and
  // the outcome, but they don't restart the run or count as a run that just finished.
  const replay = !!event.run_id && event.run_id === state.run_id && !state.running;
  let current = state;
  let outcome = next.outcome;
  if (event.run_id && incoming > known) {
    current = { ...state, run_id: event.run_id, running: true,
      stages: state.stages.map((s) => ({ ...s, status: "idle" as const, started: null, finished: null, message: "" })) };
    outcome = null;
  }
  switch (event.type) {
    case "stage":
      return { ...next, outcome, state: { ...current, running: !replay,
        stages: current.stages.map((s) => (s.name === event.data.name ? event.data : s)) } };
    case "progress":
      return { ...next, outcome, state: { ...current, counters: event.data.counters } };
    case "done":
      return { ...next, finished: next.finished + (replay ? 0 : 1), state: { ...current, running: false },
        outcome: { status: event.data.status, errors: [], runId: event.run_id } };
    case "run_failed":
      return { ...next, finished: next.finished + (replay ? 0 : 1), state: { ...current, running: false },
        outcome: { status: "failed", errors: event.data.errors, runId: event.run_id } };
  }
  return next;
}
