import type { Counters } from "../types";

function Meter({ done, of }: { done: number; of: number }) {
  return <div className="meter" aria-hidden><i style={{ width: `${of ? Math.min(100, (done / of) * 100) : 0}%` }} /></div>;
}

function Of({ label, done, of }: { label: string; done: number; of: number }) {
  return (
    <div className="counter">
      <div className="label">{label}</div>
      <div className="value num">{done.toLocaleString("en-US")} <small>of {of.toLocaleString("en-US")}</small></div>
      <Meter done={done} of={of} />
    </div>
  );
}

/** The run's counters, straight from the runner (spec §4.3). */
export function CounterStrip({ counters }: { counters: Counters }) {
  const requests = counters.requests.network + counters.requests.cache;
  return (
    <div className="panel counters">
      <div className="counter">
        <div className="label">UEFA requests</div>
        <div className="value num">{counters.requests.network.toLocaleString("en-US")} <small>fetched</small></div>
        <div className="small muted">{counters.requests.cache.toLocaleString("en-US")} from the cache</div>
        <Meter done={counters.requests.network} of={requests} />
      </div>
      <Of label="Seasons fetched" done={counters.seasons.done} of={counters.seasons.of} />
      <div className="counter">
        <div className="label">Dataset checks</div>
        <div className="value">{counters.validated === null ? "Not run" : counters.validated ? "Passed" : "Failed"}</div>
        <Meter done={counters.validated ? 1 : 0} of={1} />
      </div>
      <Of label="Season folds" done={counters.folds.done} of={counters.folds.of} />
      <Of label="Feature sets" done={counters.ablation.done} of={counters.ablation.of} />
      <Of label="AI write-ups" done={counters.narratives.done} of={counters.narratives.of} />
      {counters.players.of > 0 && <Of label="Squads indexed" done={counters.players.done} of={counters.players.of} />}
    </div>
  );
}
