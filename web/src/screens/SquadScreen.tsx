import { useEffect, useState } from "react";
import { api } from "../api";
import { Pitch } from "../components/Pitch";
import { PlayerPanel } from "../components/PlayerPanel";
import { TeamPicker } from "../components/TeamPicker";
import { useApi } from "../hooks/useApi";
import type { Pipeline } from "../hooks/usePipeline";
import { ago, num } from "../lib/format";
import { colourValue, perNinety } from "../lib/pitchLayout";
import type { Route } from "../lib/route";
import type { Meta, PlayerStatMeta, SquadPlayer } from "../types";

function SquadTable({ players, stat, selected, onSelect }: {
  players: SquadPlayer[]; stat: PlayerStatMeta; selected: string | null; onSelect: (id: string) => void;
}) {
  return (
    <div className="panel table-scroll">
      <table className="data">
        <caption className="small muted" style={{ textAlign: "left", padding: "10px 8px 4px" }}>The same squad as a table</caption>
        <thead>
          <tr><th>No.</th><th>Player</th><th>Position</th><th className="r">Minutes</th>
            <th className="r">{stat.label}{perNinety(stat) ? " per 90" : ""}</th><th className="r">Goals</th><th className="r">Assists</th></tr>
        </thead>
        <tbody>
          {players.map((p) => (
            <tr key={p.player_id} className={selected === p.player_id ? "selected" : undefined} onClick={() => onSelect(p.player_id)}>
              <td>{p.shirt ?? ""}</td><td>{p.name}</td><td>{p.position ?? "–"}</td><td className="r">{num(p.minutes, 0)}</td>
              <td className="r">{num(colourValue(p, stat), stat.decimals)}</td>
              <td className="r">{num(p.stats.goals, 0)}</td><td className="r">{num(p.stats.assists, 0)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function SquadScreen({ meta, route, navigate, pipeline }: {
  meta: Meta; route: Route; navigate: (route: Route) => void; pipeline: Pipeline;
}) {
  const teamId = route.params.t ?? null;
  const season = Number(route.params.s) || null;
  const stats = meta.player_stats.filter((s) => s.colourable);
  const stat = stats.find((s) => s.key === route.params.stat) ?? stats[0];
  const squad = useApi(teamId && season ? () => api.squad(teamId, season) : null, [teamId, season]);
  const [selected, setSelected] = useState<string | null>(null);
  const players = squad.data?.players ?? [];
  useEffect(() => {
    if (selected && !players.some((p) => p.player_id === selected)) setSelected(null);
  }, [players, selected]);
  const params = (changes: Record<string, string>) => navigate({ screen: "squad", params: { ...route.params, ...changes } });
  const chosen = players.find((p) => p.player_id === selected) ?? null;
  const listed = squad.data?.minutes_published ? players.filter((p) => p.minutes > 0) : players;
  if (!stat) return <div className="empty"><h2>Squads aren't available</h2><p>This server has no player data.</p></div>;
  return (
    <>
      <div className="squad-bar">
        <TeamPicker value={teamId && season ? { teamId, season } : null} label="Club and season"
                    onChange={(pick) => params({ t: pick.teamId, s: String(pick.season) })} />
        <div className="segmented" role="group" aria-label="Colour the players by">
          {stats.map((s) => (
            <button key={s.key} type="button" aria-pressed={s.key === stat.key} onClick={() => params({ stat: s.key })}>{s.label}</button>
          ))}
        </div>
      </div>
      {!teamId && <div className="empty"><h2>Pick a club to see its squad on the pitch</h2>
        <p>Circles grow with minutes played and glow brighter with the stat you choose. Pick a player for their seasons across clubs.</p></div>}
      {squad.error && <p className="error-note">{squad.error.message}</p>}
      {squad.data && (
        <>
          <div className="section-head">
            <h2>{squad.data.name} <span className="muted" style={{ fontWeight: 400 }}>{squad.data.label}</span></h2>
            <span className="small muted">
              {listed.length} players{squad.data.live ? `, updated ${ago(squad.data.fetched_at)}` : ""}
              {!squad.data.minutes_published && ", minutes not published so every circle is the same size"}
            </span>
          </div>
          {squad.data.stale && <p className="warn-note">UEFA couldn't be reached, so this is the squad from {ago(squad.data.fetched_at)}.</p>}
          <div className="squad-layout">
            <div style={{ display: "grid", gap: 16 }}>
              <Pitch players={players} stat={stat} minutesPublished={squad.data.minutes_published} selected={selected} onSelect={setSelected} />
              <SquadTable players={listed} stat={stat} selected={selected} onSelect={setSelected} />
            </div>
            {chosen ? (
              <PlayerPanel player={chosen} stat={stat} stats={meta.player_stats} onClose={() => setSelected(null)}
                           running={pipeline.view.state?.running ?? false}
                           onBuildIndex={() => { void pipeline.start({ stages: ["players"] }); }} />
            ) : (
              <aside className="panel card player-panel"><h3>Pick a player</h3>
                <p className="small muted">Click a circle or a table row for that player's numbers this season and every season
                  of theirs in the cached squads, at any club.</p></aside>
            )}
          </div>
        </>
      )}
    </>
  );
}
