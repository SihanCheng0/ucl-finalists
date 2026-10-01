import { stat as fmtStat } from "../lib/format";
import type { FeatureMeta, Stat, TrendPoint } from "../types";

/** "Teams beaten" across seasons: the season shown is the solid dot; a live season is hollow. */
export function Sparkline({ points, season }: { points: TrendPoint[]; season: number }) {
  const known = points.filter((p) => p.beats_season !== null);
  if (known.length < 2) return <svg className="spark" aria-hidden />;
  const w = 104, h = 28, pad = 3;
  const x = (i: number) => pad + (i * (w - 2 * pad)) / (points.length - 1);
  const y = (v: number) => h - pad - (v / 100) * (h - 2 * pad);
  const line = points.map((p, i) => (p.beats_season === null ? null : `${x(i)},${y(p.beats_season)}`)).filter(Boolean).join(" ");
  const label = known.map((p) => `${p.label}: ${p.beats_season}%`).join(", ");
  return (
    <svg className="spark" viewBox={`0 0 ${w} ${h}`} role="img" aria-label={`Teams beaten by season: ${label}`}>
      <polyline points={line} />
      {points.map((p, i) => p.season === season && p.beats_season !== null && (
        <circle key={p.season} cx={x(i)} cy={y(p.beats_season)} r={3.2} className={p.live ? "live" : "now"} />
      ))}
    </svg>
  );
}

export function StatRow({ stat, meta, trend, season }: { stat: Stat; meta: FeatureMeta; trend: TrendPoint[]; season: number }) {
  const beat = stat.beats_season;
  const tip = beat === null ? "Not recorded"
    : `Beat ${beat}% of that season's teams${stat.beats_all !== null ? `, and ${stat.beats_all}% of every team since 2011-12` : ""}`;
  return (
    <div className="stat-row" title={tip}>
      <div className="stat-label">
        {stat.label}
        {meta.lower_is_better && <span className="hint">lower is better</span>}
      </div>
      <div className="stat-value">{fmtStat(stat.value, meta)}</div>
      <div className="beats">
        <div className="beats-track" aria-hidden>
          {beat !== null && <div className="beats-fill" style={{ width: `${beat}%` }} />}
          {stat.beats_all !== null && <div className="beats-tick" style={{ left: `calc(${stat.beats_all}% - 1px)` }} />}
        </div>
        <div className="beats-text">
          {beat === null ? "not recorded" : `beat ${beat}% of the season`}
          {stat.provisional && <span className="provisional"> · early-season</span>}
        </div>
      </div>
      <Sparkline points={trend} season={season} />
    </div>
  );
}
