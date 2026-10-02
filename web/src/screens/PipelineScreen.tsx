import { useState } from "react";
import { Checks } from "../components/Checks";
import { CounterStrip } from "../components/CounterStrip";
import { EventLog } from "../components/EventLog";
import { StageRail } from "../components/StageRail";
import type { Loaded } from "../hooks/useApi";
import { localAiState } from "../hooks/useChecks";
import type { Pipeline } from "../hooks/usePipeline";
import { ago } from "../lib/format";
import type { CheckReport, Meta } from "../types";

export function PipelineScreen({ meta, pipeline, checks }: { meta: Meta; pipeline: Pipeline; checks: Loaded<CheckReport> }) {
  const { view, start, startError, connected } = pipeline;
  const [skipAi, setSkipAi] = useState(false);
  const [refreshLive, setRefreshLive] = useState(false);
  const state = view.state;
  const hasLive = meta.stages.some((s) => s.name === "live");
  const hasPlayers = meta.stages.some((s) => s.name === "players");
  if (!state) {
    return <div className="empty"><h2>Connecting to the pipeline…</h2><p>If this stays here, check that `uv run ucl web` is still running.</p></div>;
  }
  const running = state.running;
  const outcome = view.outcome;
  const ai = localAiState(checks.data);
  return (
    <>
      <div className="section-head">
        <div>
          <h2>Pipeline</h2>
          <p className="small muted">Data built {ago(meta.data.built_at)}, model trained {ago(meta.data.modelled_at)},
            write-ups from {ago(meta.data.analysed_at)}.</p>
        </div>
        {!connected && <span className="chip">Reconnecting to the event stream…</span>}
      </div>
      <Checks checks={checks} />
      <section className="panel controls" aria-label="Run controls">
        <button className="button primary" type="button" disabled={running}
                onClick={() => start({ skip_ai: skipAi, refresh_live: refreshLive && hasLive })}>Run all stages</button>
        <label className="check"><input type="checkbox" checked={skipAi} onChange={(e) => setSkipAi(e.target.checked)} />Skip AI write-ups</label>
        {hasLive && <label className="check"><input type="checkbox" checked={refreshLive} onChange={(e) => setRefreshLive(e.target.checked)} />Also refresh the live season</label>}
        {hasPlayers && <button className="button" type="button" disabled={running} onClick={() => start({ stages: ["players"] })}>Build player index</button>}
        <span className="note">The AI step needs LM Studio open. Inputs that haven't changed replay from the cache, so a rerun takes seconds.</span>
        {!ai.ready && !skipAi && (
          <p className={ai.blocking ? "error-note" : "warn-note"} role="status">
            {ai.note} {ai.blocking
              ? "The AI step would stop with a warning, so tick Skip AI write-ups or fix it first (see Before you run)."
              : "The AI step starts LM Studio and loads the model when it gets there, which adds a minute or two."}
            {ai.blocking && <> <button className="button small" type="button" onClick={() => setSkipAi(true)}>Skip AI this time</button></>}
          </p>
        )}
        {startError && <p className="error-note" role="alert">{startError}</p>}
      </section>
      <StageRail stages={state.stages} meta={meta.stages} running={running}
                 onRun={(stage) => start({ stages: [stage], skip_ai: stage === "analyze" ? false : skipAi })} />
      <CounterStrip counters={state.counters} />
      {outcome && !running && (
        <section className={`panel outcome`} role="status">
          <h3>{outcome.status === "failed" ? "The last run stopped with an error" : outcome.status === "warning"
            ? "The last run finished with warnings" : "The last run finished"}</h3>
          {outcome.errors.length > 0 && <ul>{outcome.errors.map((error, i) => <li key={i}>{error}</li>)}</ul>}
        </section>
      )}
      <EventLog lines={view.log} stages={meta.stages.map((s) => s.name)} />
    </>
  );
}
