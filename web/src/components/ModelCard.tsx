import { share, signed } from "../lib/format";
import type { ModelCardData, Profile } from "../types";

const STAGES = ["KO", "QF", "SF", "Final", "Won"];

/** Where the team finished against where the model expected it, and what moved that expectation. */
export function ModelCard({ model, profile }: { model: ModelCardData; profile: Profile }) {
  const reached = profile.ko_stage ?? -1;
  const marker = Math.min(Math.max(model.exp_stage, 0), 4);
  const maxContribution = Math.max(...[...model.top_up, ...model.top_down].map((c) => Math.abs(c.contribution)), 0.01);
  return (
    <section className="panel card" aria-label="Model card">
      <div className="card-title"><h3>Model card</h3><span className="muted small">trained on the other 14 seasons</span></div>
      <div className="figures">
        <div className="figure"><div className="label">Chance of reaching the final</div><div className="value">{share(model.p_final)}</div></div>
        <div className="figure"><div className="label">Rank that season</div><div className="value">#{model.rank} <span className="muted small">of {model.ko_size}</span></div></div>
        <div className="figure"><div className="label">Random knockout team</div><div className="value">{share(model.base_rate)}</div></div>
      </div>
      <div>
        <div className="ladder" role="img"
             aria-label={`Reached ${STAGES[Math.max(reached, 0)]}; the model expected stage ${model.exp_stage.toFixed(2)} of 4, nearest ${model.nearest_stage}`}>
          {STAGES.map((stage, i) => <span key={stage} className={i === reached ? "final" : i < reached ? "reached" : undefined}>{stage}</span>)}
          <i className="marker" style={{ left: `${((marker + 0.5) / 5) * 100}%` }} />
        </div>
        <p className="small muted">▲ the model's expected stage, {model.exp_stage.toFixed(2)} of 4 (nearest: {model.nearest_stage}).</p>
      </div>
      <div className="contrib">
        <h3 className="small">What moved the prediction</h3>
        {[...model.top_up, ...[...model.top_down].reverse()].map((c) => (
          <div className="contrib-row" key={c.feature}>
            <span>{c.label}</span>
            <div className={`bar${c.contribution < 0 ? " down" : ""}`} style={{
              width: `${(Math.abs(c.contribution) / maxContribution) * 100}%`,
              background: c.contribution >= 0 ? "var(--s1)" : "var(--neg)" }} />
            <span className="val">{signed(c.contribution)}</span>
          </div>
        ))}
        <p className="small muted">Knockout rounds added to or taken from the expected stage (SHAP values).</p>
      </div>
    </section>
  );
}
