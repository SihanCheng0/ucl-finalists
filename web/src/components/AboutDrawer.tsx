import { useEffect, useRef } from "react";
import { api } from "../api";
import { useApi } from "../hooks/useApi";
import { num, share } from "../lib/format";
import type { Meta } from "../types";
import { CloseIcon } from "./Icons";

const FOCUSABLE = 'a[href], button:not([disabled]), input, select, textarea, [tabindex]:not([tabindex="-1"])';

/** "How UCL Lab works": a slide-over that explains the whole process, from UEFA's feeds to the dashboard. It sits
 * over whatever screen is open, so reading it never loses your place. */
export function AboutDrawer({ meta, onClose }: { meta: Meta; onClose: () => void }) {
  const panel = useRef<HTMLDivElement>(null);
  const summary = useApi(meta.ready ? () => api.summary() : null, [meta.ready]);
  const historical = meta.seasons.filter((s) => !s.live);
  const live = meta.seasons.find((s) => s.live);
  const first = historical[0]?.label ?? "2011-12";
  const last = historical[historical.length - 1]?.label ?? "2025-26";

  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    panel.current?.querySelector<HTMLElement>("button")?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") { onClose(); return; }
      if (event.key !== "Tab" || !panel.current) return;
      const items = [...panel.current.querySelectorAll<HTMLElement>(FOCUSABLE)];
      if (!items.length) return;
      const [head, tail] = [items[0], items[items.length - 1]];
      if (event.shiftKey && document.activeElement === head) { event.preventDefault(); tail.focus(); }
      else if (!event.shiftKey && document.activeElement === tail) { event.preventDefault(); head.focus(); }
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = overflow;
      opener?.focus();
    };
  }, [onClose]);

  const m = summary.data?.metrics;
  const robust = summary.data?.drivers.filter((d) => d.kind === "robust").map((d) => d.label.toLowerCase()) ?? [];
  return (
    <div className="drawer-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="drawer" ref={panel} role="dialog" aria-modal="true" aria-labelledby="about-title">
        <div className="drawer-head">
          <h2 id="about-title">How UCL Lab works</h2>
          <button className="icon-button" type="button" onClick={onClose} aria-label="Close"><CloseIcon /></button>
        </div>
        <p className="drawer-lede">
          UCL Lab asks one question: what do Champions League finalists have in common? It collects {historical.length} seasons
          of UEFA data ({first} to {last}), learns which first-phase numbers go with deep knockout runs, tests that on
          seasons it never saw, and has a local AI explain the result. This page follows that process from start to end.
        </p>

        <div className="flow" aria-hidden>
          {["Fetch", "Build", "Model", "Analyze", "Report"].map((stage) => <span key={stage} className="flow-step">{stage}</span>)}
        </div>

        <section>
          <h3>The pipeline</h3>
          <ol className="steps">
            <li>
              <h4>Fetch</h4>
              <p>Every Champions League match, each team's stats from every first-phase match, and the UEFA club
                coefficients, all from UEFA's public feeds. Everything is saved on this machine, so later runs work offline
                and only fetch what's new.</p>
            </li>
            <li>
              <h4>Build</h4>
              <p>Match stats become {meta.features.length || 15} per-game numbers for each team-season, in three groups:
                <em> results</em> (points, goal difference), <em>style</em> (shots, possession, passing, distance, fouls) and
                <em> pedigree</em> (the club coefficient before the season). Units UEFA mixed up are converted, partial
                tracking is ignored, and a plausibility check stops the build if a value looks impossible. Each number is
                then compared with the rest of that season's field, so different eras are measured fairly.</p>
            </li>
            <li>
              <h4>Model</h4>
              <p>Only teams that reached the knockouts are used, and the target is how far each went. The model trains on
                14 seasons and predicts the 15th, rotating until every season has been predicted blind
                (leave-one-season-out). One model estimates how far a team goes; a simpler one gives its chance of reaching
                the final. SHAP values show which stats moved each prediction, and resampling whole seasons gives 95%
                intervals for every headline figure.</p>
            </li>
            <li>
              <h4>Analyze</h4>
              <p>A local AI model, Qwen running in LM Studio on this computer, writes a scouting report on each finalist of
                2022 to 2026 and a summary of the findings. It sees only a fact sheet of the numbers, never the internet.
                Every figure it writes is checked against the data, and answers with a slip in wording get one retry.
                The badge on each report says whether all its figures were found.</p>
            </li>
            <li>
              <h4>Report</h4>
              <p>Everything above is written into one shareable report page, the same findings you can explore here.</p>
            </li>
          </ol>
          <p className="small muted">Two optional stages sit beside the pipeline. <strong>Live season</strong> builds
            {live ? ` ${live.label}` : " the current season"} from the matches finished so far, refreshed every six hours. It's shown
            for comparison only and never used to train the model. <strong>Player index</strong> downloads every squad since {first},
            so a player's seasons line up across all their clubs.</p>
        </section>

        <section>
          <h3>Reading the screens</h3>
          <ul className="reading">
            <li><strong>Pipeline</strong> shows each stage as it runs, with live counters and the log. <em>Before you run</em> checks
              LM Studio, UEFA and the saved data first.</li>
            <li><strong>Explore:</strong> each bar is the share of that season's teams this one did better than, so a longer bar
              is always better, even for stats where lower wins. The small tick marks the same share against every team since
              {` ${first}`}. The line beside it traces that share across the club's seasons.</li>
            <li><strong>Model card:</strong> the chance of reaching the final, the team's rank that season, and a ladder from the
              knockout round to the trophy. The ▲ is where the model expected the team to finish.</li>
            <li><strong>Compare:</strong> each bar is how far a team was from its own season's average. The middle of each track is
              average, and better always points outward.</li>
            <li><strong>Squad:</strong> circles grow with minutes played and glow brighter with the stat you pick. Counts are per
              90 minutes, so substitutes and starters compare fairly.</li>
            <li><strong>Predict:</strong> each club's chance of reaching every round of {live ? live.label : "the current season"} and
              of winning it, a head-to-head between any two team-seasons, and how well these odds did on past seasons.</li>
          </ul>
        </section>

        <section>
          <h3>Predictions</h3>
          <p>The Predict screen and the "If they met" card on Compare share one model. Every Champions League match since {first}
            moves a rating for each club, by the result and the margin, and each season starts by pulling ratings part of the
            way back toward the club's UEFA coefficient. A rating gap becomes expected goals, so the same numbers give one match,
            a two-legged tie and a final. The deeper the round, the less a gap counts: that's what past knockouts show.</p>
          <p>Title odds come from playing the rest of the season 20,000 times, with the league-phase table, the play-offs and the
            seeded bracket as UEFA runs them. The ratings were tuned on 2013-14 to 2018-19 and tested on the seasons after.</p>
          <p className="small muted">They see only Champions League matches, not domestic form, injuries or transfers. Even a clear
            favourite usually doesn't win the trophy.</p>
        </section>

        <section>
          <h3>What it found</h3>
          {m ? (
            <p>
              First-phase play only modestly predicts deep runs. Within a season, the model's ranking correlates
              {` ${num(m.spearman.value, 2)}`} with how far teams really went, and it separates finalists from the rest with an AUC
              of {num(m.auc.value, 2)}. {share(m.top4_share.value)} of finalists were in its top four, against
              {` ${share(m.top4_share.chance)}`} by chance. Its edge over simply guessing the base rate is
              {m.brier_skill.ci[0] <= 0 ? " inconclusive" : ` ${num(m.brier_skill.value, 2)}`}.
              {robust.length > 0 && <> The stats that hold up most consistently: {robust.join(", ")}.</>}
            </p>
          ) : <p className="muted">{meta.ready ? "Loading the headline results…" : "Run the pipeline to see the results here."}</p>}
        </section>

        <section>
          <h3>Keep in mind</h3>
          <ul className="reading">
            <li>These are associations, not causes, and first-phase numbers depend on who a team was drawn against.</li>
            <li>Knockout football is noisy, and {historical.length} seasons hold only {historical.length * 2} finalists.</li>
            <li>UEFA publishes no expected-goals data for these seasons, and its feeds are used as published.</li>
            <li>Live-season figures are early: one result moves them a long way until all first-phase matches are played.</li>
          </ul>
        </section>
      </div>
    </div>
  );
}
