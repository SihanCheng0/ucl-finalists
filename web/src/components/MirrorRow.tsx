import { stat as fmtStat } from "../lib/format";
import type { CompareRow, FeatureMeta } from "../types";

const LIMIT = 2.5; // z beyond this is drawn at full length

/** One stat, two teams. Each side's track has z = 0 in its middle and draws "better" outward, so a long bar
 * pointing away from the label is good for that team; the side ahead is drawn solid. */
export function MirrorRow({ row, meta, nameA, nameB }: { row: CompareRow; meta: FeatureMeta; nameA: string; nameB: string }) {
  const bar = (z: number | null, side: "a" | "b") => {
    if (z === null) return null;
    const length = (Math.min(Math.abs(z), LIMIT) / LIMIT) * 50;
    const outward = (side === "a" && z > 0) || (side === "b" && z < 0) ? "left" : "right";
    const style = outward === "left" ? { right: "50%", width: `${length}%` } : { left: "50%", width: `${length}%` };
    const lead = row.ahead === side;
    return <span className={`zb${lead ? " lead" : ""}`} style={{ ...style, background: side === "a" ? "var(--s1)" : "var(--s2)" }} />;
  };
  const verdict = row.ahead === "tie" ? "level" : `${row.ahead === "a" ? nameA : nameB} ahead`;
  return (
    <div className="mirror-row" title={`${row.label}: ${verdict}`}>
      <div className="val">{fmtStat(row.a_value, meta)}<small>{row.a_beats === null ? "–" : `beat ${row.a_beats}%`}</small></div>
      <div className="zt" aria-hidden>{bar(row.a_z, "a")}</div>
      <div className="label">{row.label}{meta.lower_is_better && <span className="muted"> (lower is better)</span>}</div>
      <div className="zt" aria-hidden>{bar(row.b_z, "b")}</div>
      <div className="val b">{fmtStat(row.b_value, meta)}<small>{row.b_beats === null ? "–" : `beat ${row.b_beats}%`}</small></div>
    </div>
  );
}
