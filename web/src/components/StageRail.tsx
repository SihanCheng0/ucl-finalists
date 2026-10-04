import { useEffect, useState } from "react";
import { duration } from "../lib/format";
import { STATIC_SITE } from "../site";
import type { StageMeta, StageState } from "../types";
import { AlertIcon, CheckIcon, CrossIcon, SkipIcon } from "./Icons";

const WORDS: Record<string, string> = { idle: STATIC_SITE ? "Not in this run" : "Waiting", running: "Running", done: "Done",
  warning: "Done with a warning", skipped: "Skipped", failed: "Failed" };

function Ring({ status }: { status: string }) {
  const Icon = status === "done" ? CheckIcon : status === "failed" ? CrossIcon : status === "warning" ? AlertIcon
    : status === "skipped" ? SkipIcon : null;
  return <span className="ring">{Icon && <Icon />}</span>;
}

/** The stages left to right, in the order they run; a running stage counts its time up. */
export function StageRail({ stages, meta, running, onRun }: {
  stages: StageState[]; meta: StageMeta[]; running: boolean; onRun?: (stage: string) => void;
}) {
  const [now, setNow] = useState(() => Date.now() / 1000);
  const ticking = stages.some((s) => s.status === "running");
  useEffect(() => {
    if (!ticking) return;
    const timer = window.setInterval(() => setNow(Date.now() / 1000), 500);
    return () => window.clearInterval(timer);
  }, [ticking]);
  const label = new Map(meta.map((m) => [m.name, m]));
  return (
    <div className="rail" role="list" aria-label="Pipeline stages">
      {stages.map((stage) => {
        const info = label.get(stage.name);
        const elapsed = stage.started === null ? null : (stage.finished ?? now) - stage.started;
        return (
          <div key={stage.name} role="listitem" className={`stage ${stage.status}${info?.optional ? " optional" : ""}`}>
            <div className="stage-node"><Ring status={stage.status} /><span className="link" /></div>
            <h3>{info?.label ?? stage.name}{info?.optional && <span className="muted small"> (optional)</span>}</h3>
            <div className="status">
              <span>{WORDS[stage.status]}</span>
              {elapsed !== null && <span className="muted num">{duration(elapsed)}</span>}
            </div>
            {stage.message && <div className="message" title={stage.message}>{stage.message}</div>}
            {onRun && <div><button className="button small" type="button" disabled={running} onClick={() => onRun(stage.name)}>
              Run {info?.label.toLowerCase() ?? stage.name}</button></div>}
          </div>
        );
      })}
    </div>
  );
}
