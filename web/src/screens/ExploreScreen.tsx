import { api } from "../api";
import { ModelCard } from "../components/ModelCard";
import { Narrative } from "../components/Narrative";
import { Overview } from "../components/Overview";
import { SeasonScrubber } from "../components/SeasonScrubber";
import { StatRow } from "../components/StatRow";
import { TeamSearch } from "../components/TeamSearch";
import { useApi } from "../hooks/useApi";
import { ago, plural } from "../lib/format";
import type { Route } from "../lib/route";
import type { Meta, Profile } from "../types";

function capitalised(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function TeamView({ profile, meta, navigate }: { profile: Profile; meta: Meta; navigate: (route: Route) => void }) {
  const features = new Map(meta.features.map((f) => [f.feature, f]));
  const pick = `${profile.team_id}:${profile.season}`;
  return (
    <>
      <section className="team-hero">
        <h1>{profile.name} <span className="season">{profile.label}</span></h1>
        <div className="facts">
          {profile.live
            ? <span className="chip live">In progress, {plural(profile.matches_played, "match", "matches")} of {profile.phase_matches}</span>
            : <span className={`chip${profile.result ? " final" : ""}`}>{profile.stage_label}</span>}
          {profile.result && <span>{capitalised(profile.result.replace(/^(Winner|Runner-up): /, ""))}</span>}
        </div>
        <div className="actions">
          <button className="button small" type="button"
                  onClick={() => navigate({ screen: "compare", params: { a: pick } })}>Compare with another team</button>
          <button className="button small" type="button"
                  onClick={() => navigate({ screen: "squad", params: { t: profile.team_id, s: String(profile.season) } })}>See the squad</button>
        </div>
        {profile.live && profile.stale && <p className="warn-note">UEFA couldn't be reached, so these are the numbers from {ago(profile.fetched_at)}.</p>}
        {profile.provisional && <p className="warn-note">Early-season numbers: one result swings a "teams beaten" figure a long way until all {profile.phase_matches} first-phase matches are played.</p>}
      </section>
      <div className="explore-grid">
        <div>
          {meta.sections.map((section) => {
            const stats = profile.features.filter((s) => s.section === section);
            if (!stats.length) return null;
            return (
              <section key={section} className="stat-section" aria-label={section}>
                <h3>{section}</h3>
                {stats.map((stat) => {
                  const featureMeta = features.get(stat.feature);
                  return featureMeta && <StatRow key={stat.feature} stat={stat} meta={featureMeta}
                                                 trend={profile.trend[stat.feature] ?? []} season={profile.season} />;
                })}
              </section>
            );
          })}
          <p className="small muted" style={{ marginTop: 14 }}>Bars: the share of that season's teams this one did better than.
            The tick marks the same against every team since 2011-12. Lines: the same share across the club's seasons.</p>
        </div>
        <div className="side">
          {profile.model ? <ModelCard model={profile.model} profile={profile} /> : (
            <section className="panel card"><h3>Model card</h3>
              <p className="small muted">{profile.live ? "The model only scores completed seasons." :
                "The model scores knockout teams only; this team went out in the first phase."}</p></section>
          )}
          {profile.narrative && <Narrative narrative={profile.narrative} />}
        </div>
      </div>
    </>
  );
}

export function ExploreScreen({ meta, route, navigate, dataVersion, onAbout }: {
  meta: Meta; route: Route; navigate: (route: Route) => void; dataVersion: number; onAbout: () => void;
}) {
  const teamId = route.params.t ?? null;
  const season = Number(route.params.s) || null;
  const profile = useApi(teamId && season ? () => api.profile(teamId, season) : null, [teamId, season, dataVersion]);
  const summary = useApi(!teamId ? () => api.summary() : null, [teamId, dataVersion]);
  const available = new Set((profile.data?.seasons ?? []).map((s) => s.season));
  const go = (t: string, s: number) => navigate({ screen: "explore", params: { t, s: String(s) } });
  return (
    <>
      <div className="explore-bar">
        <TeamSearch onPick={(hit) => go(hit.team_id, hit.seasons[0].season)} initial={profile.data?.name ?? ""} />
        <SeasonScrubber seasons={meta.seasons} available={available} value={season} onChange={(s) => teamId && go(teamId, s)} />
      </div>
      {profile.error && <p className="error-note">{profile.error.message}</p>}
      {profile.data && <TeamView profile={profile.data} meta={meta} navigate={navigate} />}
      {!teamId && summary.data && <Overview summary={summary.data} onAbout={onAbout} />}
      {!teamId && summary.error && <p className="error-note">{summary.error.message}</p>}
    </>
  );
}
