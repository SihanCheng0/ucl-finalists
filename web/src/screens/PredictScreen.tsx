import { useState } from "react";
import { api } from "../api";
import { HeadToHeadCard } from "../components/HeadToHeadCard";
import { SwapIcon } from "../components/Icons";
import { TeamPicker } from "../components/TeamPicker";
import { useApi } from "../hooks/useApi";
import { ago, num, share } from "../lib/format";
import { chance, parseVenue, STAGE_COLUMNS, swapVenue } from "../lib/predict";
import { formatPick, parsePick, type Pick, type Route } from "../lib/route";
import type { Forecast, Meta, TrackRecord } from "../types";

const SHOWN = 16;

function day(iso: string | null): string {
  return iso ? new Date(iso).toLocaleDateString("en-GB", { day: "numeric", month: "long" }) : "";
}

function OddsTable({ forecast }: { forecast: Forecast }) {
  const [all, setAll] = useState(false);
  const rows = all ? forecast.teams : forecast.teams.slice(0, SHOWN);
  const top = Math.max(...forecast.teams.map((t) => t.p_win), 0.01);
  return (
    <section className="panel card" aria-label="Every club's chances">
      <div className="card-title">
        <h3>Every club's chances</h3>
        <span className="small muted">Points from {forecast.played} of {forecast.league_matches} league-phase matches</span>
      </div>
      <div className="table-scroll">
        <table className="data odds">
          <thead>
            <tr>
              <th className="r">#</th><th>Club</th><th className="r rating" title="Rating going into the next match">Rating</th>
              <th className="r" title="League-phase points (matches played)">Pts</th>
              {STAGE_COLUMNS.map((c) => <th key={c.key} className={c.key === "p_win" ? "win" : "r"} title={c.title}>{c.label}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((t, i) => (
              <tr key={t.team_id}>
                <td className="r muted">{i + 1}</td>
                <td><a href={`#/explore?t=${t.team_id}&s=${forecast.season}`}>{t.name}</a></td>
                <td className="r muted rating">{t.rating.toLocaleString("en-US")}</td>
                <td className="r">{t.points} <span className="muted small">({t.played})</span></td>
                <td className="win">
                  <div className="odds-bar"><span style={{ width: `${(t.p_win / top) * 100}%` }} /><b>{chance(t.p_win)}</b></div>
                </td>
                {STAGE_COLUMNS.slice(1).map((c) => <td key={c.key} className="r">{chance(t[c.key])}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {forecast.teams.length > SHOWN && (
        <p><button className="link-button" type="button" onClick={() => setAll(!all)}>
          {all ? `Show the top ${SHOWN}` : `Show all ${forecast.teams.length} clubs`}
        </button></p>
      )}
    </section>
  );
}

function TrackRecordCard({ record }: { record: TrackRecord }) {
  const { matches: m, ties, titles } = record;
  const mean = (values: number[]) => values.reduce((sum, v) => sum + v, 0) / Math.max(values.length, 1);
  const favouriteWon = titles.filter((t) => t.winner_id === t.favourite_id).length;
  const counts = new Map<string, number>();
  titles.forEach((t) => counts.set(t.favourite, (counts.get(t.favourite) ?? 0) + 1));
  const [usual, times] = [...counts.entries()].sort((x, y) => y[1] - x[1])[0] ?? ["", 0];
  const usualWon = titles.filter((t) => t.favourite === usual && t.winner_id === t.favourite_id).length;
  return (
    <section className="panel card" aria-label="How good these odds are">
      <div className="card-title"><h3>How good are these odds?</h3>
        <span className="small muted">Tested on {record.seasons[0]} to {record.seasons[1]}, which the settings never saw</span></div>
      <div className="record-grid">
        <ul className="record">
        <li>The likeliest 90-minute result happened in <b>{share(m.accuracy)}</b> of {m.matches.toLocaleString("en-US")} matches.</li>
        <li>Its probabilities scored a log loss of <b>{num(m.log_loss, 3)}</b> (lower is better), against {num(m.pedigree_log_loss, 3)} from
          club coefficients alone and {num(m.base_rate_log_loss, 3)} from always guessing the usual split of home wins, draws and away wins.</li>
        <li>Favourites went through <b>{share(ties.happened)}</b> of {ties.ties} two-legged ties, when it gave them {share(ties.predicted)}.</li>
        <li>When the knockouts began, it gave the eventual winner <b>{share(mean(titles.map((t) => t.p_winner)))}</b> on average,
          against {share(mean(titles.map((t) => 1 / t.teams_left)))} for a random pick. Its favourite won {favouriteWon} of {titles.length}.</li>
      </ul>
      <div className="table-scroll">
        <table className="data record-table">
          <thead><tr><th>Season</th><th>Winner</th><th className="r">Its chance</th><th>Favourite</th><th className="r">Chance</th></tr></thead>
          <tbody>
            {titles.map((t) => (
              <tr key={t.season}>
                <td>{t.label}</td><td>{t.winner}</td>
                <td className="r">{chance(t.p_winner)} <span className="muted small">#{t.winner_rank}</span></td>
                <td>{t.favourite}</td><td className="r">{chance(t.p_favourite)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      </div>
      {times > 1 && (
        <p className="small muted">The ratings see only Champions League matches, so a club that dominates its group rates highly: {usual} was
          the favourite in {times} of {titles.length} seasons and won {usualWon === 0 ? "none of them" : usualWon === 1 ? "once" : `${usualWon} times`}.
          Knockout football is close to a coin flip between the best clubs, so even a clear favourite usually loses.</p>
      )}
    </section>
  );
}

export function PredictScreen({ meta, route, navigate, dataVersion }: {
  meta: Meta; route: Route; navigate: (route: Route) => void; dataVersion: number;
}) {
  const forecast = useApi(() => api.forecast(), [dataVersion, meta.live_status]);
  const f = forecast.data;
  const params = (changes: Record<string, string>) => navigate({ screen: "predict", params: { ...route.params, ...changes } });
  const favourite = (i: number): Pick | null => (f && f.teams[i] ? { teamId: f.teams[i].team_id, season: f.season } : null);
  const a = parsePick(route.params.a) ?? favourite(0);
  const b = parsePick(route.params.b) ?? favourite(1);
  const venue = parseVenue(route.params.v);
  const [first, second, third] = f?.teams ?? [];
  return (
    <>
      <section className="predict-hero">
        <h1>Who wins the <span className="nowrap">{f?.label ?? meta.seasons.find((s) => s.live)?.label ?? "next"}</span> Champions League?</h1>
        {first && second && third && (
          <p>Favourites: <b>{first.name}</b> at {chance(first.p_win)}, then {second.name} at {chance(second.p_win)} and {third.name} at {chance(third.p_win)}.</p>
        )}
        {f && (
          <p className="small muted">{f.simulations.toLocaleString("en-US")} simulations of the rest of the season, after {f.played} of {f.league_matches} league-phase
            matches{f.as_of ? ` (the last on ${day(f.as_of)})` : ""}. Each club's rating comes from every Champions League match since
            2011-12; the stronger side is more likely to win a match, never certain to.</p>
        )}
        {f?.stale && <p className="warn-note">UEFA couldn't be reached, so these odds use the results from {ago(f.fetched_at)}.</p>}
      </section>
      {forecast.error && (
        <div className="empty"><h2>No title odds yet</h2><p>{forecast.error.message}.</p>
          <p className="small">Head-to-heads between past seasons still work below.</p></div>
      )}
      {!f && !forecast.error && <p className="muted">Simulating the season…</p>}
      {f && <OddsTable forecast={f} />}
      <section className="predict-h2h" aria-label="Head to head">
        <div className="section-head"><h2>Head to head</h2>
          <span className="small muted">Any two clubs, any seasons since 2011-12: this season's are a forecast, past ones a dream match</span></div>
        <div className="pickers">
          <TeamPicker value={a} onChange={(pick) => params({ a: formatPick(pick) })} side="a" label="First team" />
          <button className="icon-button" type="button" aria-label="Swap the teams" title="Swap the teams" disabled={!a || !b}
                  onClick={() => a && b && params({ a: formatPick(b), b: formatPick(a), v: swapVenue(venue) })}>
            <SwapIcon />
          </button>
          <TeamPicker value={b} onChange={(pick) => params({ b: formatPick(pick) })} side="b" label="Second team" />
        </div>
        {a && b && <HeadToHeadCard a={a} b={b} venue={venue} onVenue={(v) => params({ v })} dataVersion={dataVersion} />}
      </section>
      {f && <TrackRecordCard record={f.track_record} />}
    </>
  );
}
