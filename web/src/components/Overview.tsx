import { num, share } from "../lib/format";
import type { Summary } from "../types";
import { Narrative } from "./Narrative";

const GROUP_COLOR: Record<string, string> = { pedigree: "var(--s1)", results: "var(--s2)", style: "var(--s3)" };
const GROUP_NAME: Record<string, string> = { pedigree: "Pedigree", results: "Results", style: "Style" };

/** What the model learned, before any team is picked. */
export function Overview({ summary, onAbout }: { summary: Summary; onAbout: () => void }) {
  const m = summary.metrics;
  const top = Math.max(...summary.drivers.map((d) => d.importance), 0.01);
  const ci = (lo: number, hi: number, f: (v: number) => string) => `95% interval ${f(lo)} to ${f(hi)}`;
  return (
    <div className="overview">
      <div className="overview-lede">
        <h1>What makes a Champions League finalist?</h1>
        <p>Fifteen seasons of group and league-phase stats, a model tested on seasons it never saw, and a local AI
          that writes up what it found. Search a club to explore one season, or compare two.</p>
        <p><button className="link-button" type="button" onClick={onAbout} aria-haspopup="dialog">How it works, step by step</button></p>
      </div>
      <div className="metric-grid">
        <div className="panel metric">
          <div className="what">Ranks teams within a season</div>
          <div className="value">{num(m.spearman.value, 2)}</div>
          <div className="ci">Spearman correlation, {ci(m.spearman.ci[0], m.spearman.ci[1], (v) => num(v, 2))}</div>
        </div>
        <div className="panel metric">
          <div className="what">Spots the finalists</div>
          <div className="value">{num(m.auc.value, 2)}</div>
          <div className="ci">AUC (0.5 is a coin flip), {ci(m.auc.ci[0], m.auc.ci[1], (v) => num(v, 2))}</div>
        </div>
        <div className="panel metric">
          <div className="what">Finalists in the model's top four</div>
          <div className="value">{share(m.top4_share.value)}</div>
          <div className="ci">against {share(m.top4_share.chance)} by chance, {ci(m.top4_share.ci[0], m.top4_share.ci[1], (v) => share(v))}</div>
        </div>
        <div className="panel metric">
          <div className="what">Edge over guessing the base rate</div>
          <div className="value">{num(m.brier_skill.value, 2)}</div>
          <div className="ci">Brier skill, {ci(m.brier_skill.ci[0], m.brier_skill.ci[1], (v) => num(v, 2))}
            {m.brier_skill.ci[0] <= 0 ? ": inconclusive" : ""}</div>
        </div>
      </div>
      <section className="panel card" aria-label="What drives deep runs">
        <div className="card-title"><h3>What drives deep runs</h3>
          <div className="legend">{Object.entries(GROUP_NAME).map(([g, name]) => <span key={g}><i style={{ background: GROUP_COLOR[g] }} />{name}</span>)}</div>
        </div>
        <div className="drivers">
          {summary.drivers.map((d) => (
            <div className="driver" key={d.feature}>
              <span>{d.label}</span>
              <div className="bar" style={{ width: `${(d.importance / top) * 100}%`, background: GROUP_COLOR[d.group] }} />
              <span className="num">{num(d.importance, 3)}</span>
              <span className="chip">{d.kind ?? "unlabelled"}{d.helps ? `, ${d.helps} helps` : ""}</span>
            </div>
          ))}
        </div>
        <p className="small muted">Mean absolute SHAP value in knockout rounds. Robust: the simpler model agrees and the stat points the
          same way on its own. Conditional: only with the other stats held fixed. Model-dependent: the simpler model disagrees.</p>
      </section>
      {summary.synthesis && <Narrative narrative={summary.synthesis} title="AI summary" />}
    </div>
  );
}
