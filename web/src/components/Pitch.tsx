import { useEffect, useMemo, useRef, useState } from "react";
import { num, plural } from "../lib/format";
import { layoutSquad, perNinety, type Circle } from "../lib/pitchLayout";
import type { Line, PlayerStatMeta, SquadPlayer } from "../types";

const W = 1050, H = 680; // a 105 m x 68 m pitch at 10 units per metre

function Markings() {
  const stripes = Array.from({ length: 10 }, (_, i) => i);
  return (
    <g>
      <rect width={W} height={H} fill="var(--pitch)" />
      {stripes.map((i) => i % 2 === 1 && <rect key={i} x={(i * W) / 10} width={W / 10} height={H} fill="var(--pitch-stripe)" />)}
      <g className="chalk">
        <rect x={20} y={20} width={W - 40} height={H - 40} />
        <line x1={W / 2} y1={20} x2={W / 2} y2={H - 20} />
        <circle cx={W / 2} cy={H / 2} r={91.5} />
        <rect x={20} y={H / 2 - 201.6} width={165} height={403.2} />
        <rect x={W - 185} y={H / 2 - 201.6} width={165} height={403.2} />
        <rect x={20} y={H / 2 - 91.6} width={55} height={183.2} />
        <rect x={W - 75} y={H / 2 - 91.6} width={55} height={183.2} />
        <path d={`M 185 ${H / 2 - 73} A 91.5 91.5 0 0 1 185 ${H / 2 + 73}`} />
        <path d={`M ${W - 185} ${H / 2 - 73} A 91.5 91.5 0 0 0 ${W - 185} ${H / 2 + 73}`} />
      </g>
      <circle className="spot" cx={W / 2} cy={H / 2} r={4} />
      <circle className="spot" cx={130} cy={H / 2} r={4} />
      <circle className="spot" cx={W - 130} cy={H / 2} r={4} />
    </g>
  );
}

function fill(circle: Circle): string {
  if (circle.value01 === null) return "var(--ramp-none)";
  return `color-mix(in oklab, var(--ramp-hi) ${Math.round(circle.value01 * 100)}%, var(--ramp-lo))`;
}

function surname(name: string): string {
  const parts = name.split(" ");
  return parts.length > 1 ? parts.slice(1).join(" ") : name;
}

/** The squad on its four lines; circles move, resize and recolour when the team, season or stat changes. */
export function Pitch({ players, stat, minutesPublished, selected, onSelect }: {
  players: SquadPlayer[]; stat: PlayerStatMeta; minutesPublished: boolean;
  selected: string | null; onSelect: (playerId: string) => void;
}) {
  const { key: statKey, kind: statKind } = stat;
  const layout = useMemo(() => layoutSquad(players, { key: statKey, kind: statKind }, W, H, minutesPublished),
                         [players, statKey, statKind, minutesPublished]);
  const byId = useMemo(() => new Map(players.map((p) => [p.player_id, p])), [players]);
  const [hover, setHover] = useState<{ circle: Circle; x: number; y: number } | null>(null);
  const [leaving, setLeaving] = useState<Circle[]>([]);
  const previous = useRef<Circle[]>([]);
  const wrap = useRef<HTMLDivElement>(null);

  // A player who arrives fades in by CSS when their circle mounts; one who leaves is drawn a little longer, fading
  // out, and each batch of leavers expires on its own timer so a quick second change can't cut it short.
  useEffect(() => {
    const now = new Set(layout.circles.map((c) => c.player_id));
    const gone = previous.current.filter((c) => !now.has(c.player_id));
    previous.current = layout.circles;
    setLeaving((list) => [...list.filter((c) => !now.has(c.player_id)), ...gone]);
    if (gone.length) window.setTimeout(() => setLeaving((list) => list.filter((c) => !gone.includes(c))), 320);
  }, [layout]);

  const unit = perNinety(stat) ? " per 90" : "";
  const describe = (circle: Circle) => {
    const p = byId.get(circle.player_id);
    if (!p) return "";
    const value = circle.value === null ? "not recorded" : `${num(circle.value, stat.decimals)}${unit}`;
    return `${p.shirt ? `No. ${p.shirt} ` : ""}${p.name}, ${plural(Math.round(p.minutes), "minute")}, ${stat.label.toLowerCase()} ${value}`;
  };
  const show = (circle: Circle) => {
    const box = wrap.current?.getBoundingClientRect();
    if (!box) return;
    const scale = box.width / W;
    setHover({ circle, x: circle.x * scale, y: (circle.y - circle.r) * scale });
  };

  const overflowLines = (Object.entries(layout.overflow) as [Line, number][]).filter(([, n]) => n > 0);
  return (
    <div className="pitch-wrap" ref={wrap}>
      <svg className="pitch" viewBox={`0 0 ${W} ${H}`} role="group" aria-label="Squad on the pitch">
        <Markings />
        {leaving.map((c) => (
          <g key={`gone-${c.player_id}`} className="player leaving" style={{ transform: `translate(${c.x}px, ${c.y}px)` }}>
            <circle className="disc" r={1} style={{ transform: `scale(${c.r})`, fill: fill(c) }} />
          </g>
        ))}
        {layout.circles.map((c) => {
          const p = byId.get(c.player_id);
          const dark = c.value01 !== null && c.value01 < 0.45;
          return (
            <g key={c.player_id} tabIndex={0} role="button" aria-label={describe(c)}
               className={`player${selected === c.player_id ? " selected" : ""}`}
               style={{ transform: `translate(${c.x}px, ${c.y}px)` }}
               onClick={() => onSelect(c.player_id)}
               onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(c.player_id); } }}
               onMouseEnter={() => show(c)} onMouseLeave={() => setHover(null)}
               onFocus={() => show(c)} onBlur={() => setHover(null)}>
              <circle className={`disc${c.value01 === null ? " unknown" : ""}`} r={1} style={{ transform: `scale(${c.r})`, fill: fill(c) }} />
              <text className={`shirt${dark || c.value01 === null ? " light" : ""}`}>{p?.shirt ?? ""}</text>
              <text className="tag" y={c.r + 15}>{p ? surname(p.name) : ""}</text>
            </g>
          );
        })}
        {overflowLines.map(([line, n]) => {
          const x = { GK: 0.08, DEF: 0.3, MID: 0.55, FWD: 0.8 }[line] * W;
          return <text key={line} className="overflow-tag" x={x} y={H - 34}>+{n} more</text>;
        })}
      </svg>
      <div className="pitch-legend" aria-hidden>
        <span>{layout.low === null ? "–" : num(layout.low, stat.decimals)}</span>
        <span className="ramp" />
        <span>{layout.high === null ? "–" : num(layout.high, stat.decimals)}</span>
        <span>{stat.label.toLowerCase()}{unit}{minutesPublished ? ", size: minutes" : ""}</span>
      </div>
      {hover && <div className="pitch-tip" style={{ left: hover.x, top: hover.y }}>{describe(hover.circle)}</div>}
    </div>
  );
}
