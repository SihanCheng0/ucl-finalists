import { useEffect, useMemo, useRef, useState } from "react";
import { clockTime } from "../lib/format";
import type { LogLine } from "../lib/pipelineState";

/** Every log line the runner sent, newest at the bottom. Follow keeps the newest in view; pause freezes the list. */
export function EventLog({ lines, stages }: { lines: LogLine[]; stages: string[] }) {
  const [stage, setStage] = useState<string>("all");
  const [level, setLevel] = useState<"all" | "problems">("all");
  const [follow, setFollow] = useState(true);
  const [paused, setPaused] = useState<LogLine[] | null>(null);
  const box = useRef<HTMLDivElement>(null);
  const source = paused ?? lines;
  const shown = useMemo(() => source.filter((l) => (stage === "all" || l.stage === stage)
    && (level === "all" || l.level !== "info")), [source, stage, level]);
  useEffect(() => {
    if (follow && box.current) box.current.scrollTop = box.current.scrollHeight;
  }, [shown, follow]);
  return (
    <section className="panel" aria-label="Event log">
      <div className="log-head">
        <h3>Event log</h3>
        <div className="segmented" role="group" aria-label="Stage">
          {["all", ...stages].map((name) => (
            <button key={name} type="button" aria-pressed={stage === name} onClick={() => setStage(name)}>{name === "all" ? "All stages" : name}</button>
          ))}
        </div>
        <div className="segmented" role="group" aria-label="Level">
          <button type="button" aria-pressed={level === "all"} onClick={() => setLevel("all")}>Everything</button>
          <button type="button" aria-pressed={level === "problems"} onClick={() => setLevel("problems")}>Warnings and errors</button>
        </div>
        <label className="check"><input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} />Follow</label>
        <label className="check"><input type="checkbox" checked={paused !== null} onChange={(e) => setPaused(e.target.checked ? lines : null)} />Pause</label>
        {paused && lines.length > paused.length && <span className="small muted">{lines.length - paused.length} new while paused</span>}
      </div>
      <div className="log" ref={box} role="log" aria-live="off">
        {shown.length === 0 && <p className="muted small" style={{ padding: "0 14px" }}>Nothing yet. Start a run and its log appears here.</p>}
        {shown.map((line) => (
          <div key={line.id} className={`log-line ${line.level}`}>
            <span className="t">{clockTime(line.ts)}</span>
            <span className="s">{line.stage ?? "run"}</span>
            <span className="x">{line.text}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
