import { api } from "../api";
import { useApi } from "../hooks/useApi";
import { num, plural } from "../lib/format";
import { colourValue, perNinety } from "../lib/pitchLayout";
import type { PlayerStatMeta, SquadPlayer } from "../types";
import { CloseIcon } from "./Icons";

const KEY_STATS = ["goals", "assists", "attempts", "key_passes", "passes_accuracy", "distance_covered", "top_speed",
  "tackles_won", "saves"];
const POSITION = { GK: "Goalkeeper", DEF: "Defender", MID: "Midfielder", FWD: "Forward" } as const;

function initials(name: string): string {
  return name.split(" ").map((part) => part[0]).slice(0, 2).join("");
}

/** One player: this season's numbers, then every cached season across clubs, with the chosen stat traced. */
export function PlayerPanel({ player, stat, stats, onClose, onBuildIndex, running }: {
  player: SquadPlayer; stat: PlayerStatMeta; stats: PlayerStatMeta[]; onClose: () => void;
  onBuildIndex?: () => void; running: boolean;
}) {
  const history = useApi(() => api.player(player.player_id), [player.player_id]);
  const meta = new Map(stats.map((s) => [s.key, s]));
  const shown = KEY_STATS.map((key) => meta.get(key)).filter((m): m is PlayerStatMeta => !!m)
    .filter((m) => player.stats[m.key] !== null && player.stats[m.key] !== undefined);
  const entries = history.data?.history ?? [];
  const trace = [...entries].reverse().map((e) => ({ key: `${e.season}-${e.team_id}`, label: e.label,
    value: colourValue({ ...player, minutes: e.minutes, stats: e.stats }, stat) }));
  const known = trace.filter((t) => t.value !== null) as { label: string; value: number }[];
  const top = Math.max(...known.map((t) => t.value), 0);
  const index = history.data?.index;
  return (
    <aside className="panel player-panel" aria-label={`${player.name}`}>
      <div className="player-head">
        {player.image_url
          ? <img className="avatar" src={player.image_url} alt="" onError={(e) => { e.currentTarget.style.visibility = "hidden"; }} />
          : <div className="avatar" aria-hidden>{initials(player.name)}</div>}
        <div>
          <h2>{player.name}</h2>
          <p className="small muted">
            {player.position ? POSITION[player.position] : "Position not recorded"}
            {player.shirt ? `, No. ${player.shirt}` : ""}{player.age ? `, age ${player.age}` : ""}
          </p>
        </div>
        <button className="icon-button" type="button" onClick={onClose} aria-label="Close player"><CloseIcon /></button>
      </div>
      <p className="small">{plural(Math.round(player.minutes), "minute")} this season</p>
      <div className="stat-grid">
        {shown.map((m) => (
          <div key={m.key}>
            <div className="label">{m.label}</div>
            <div className="value">{num(player.stats[m.key], m.kind === "count" ? 0 : m.decimals)}</div>
          </div>
        ))}
      </div>
      <div>
        <h3 className="small">{stat.label}{perNinety(stat) ? " per 90" : ""}, season by season</h3>
        {known.length > 1 ? (
          <svg viewBox={`0 0 ${Math.max(trace.length * 34, 120)} 64`} width="100%" height="64" role="img"
               aria-label={known.map((t) => `${t.label}: ${num(t.value, stat.decimals)}`).join(", ")}>
            {trace.map((t, i) => t.value !== null && (
              <g key={t.key} transform={`translate(${i * 34 + 6} 0)`}>
                <rect y={50 - (t.value / (top || 1)) * 44} width={22} height={Math.max((t.value / (top || 1)) * 44, 1)} rx={3}
                      fill={i === trace.length - 1 ? "var(--accent)" : "var(--rule)"} />
                <text x={11} y={62} textAnchor="middle" fontSize="9" fill="var(--muted)">{t.label.slice(2)}</text>
              </g>
            ))}
          </svg>
        ) : <p className="small muted">{history.loading ? "Loading seasons…" : entries.length <= 1
          ? "Only this season is cached so far." : "Too few minutes in the other seasons to compare this stat."}</p>}
      </div>
      <div className="table-scroll">
        <table className="data">
          <thead><tr><th>Season</th><th>Club</th><th className="r">Min</th><th className="r">{stat.label}</th></tr></thead>
          <tbody>
            {entries.map((e) => (
              <tr key={`${e.season}-${e.team_id}`}>
                <td>{e.label}{e.live ? " (live)" : ""}</td><td>{e.team}</td>
                <td className="r">{num(e.minutes, 0)}</td>
                <td className="r">{num(e.stats[stat.key], stat.kind === "count" ? 0 : stat.decimals)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {index && !index.complete && (
        <div className="warn-note small">
          {index.running ? `Building the player index: ${index.squads.done} of ${index.squads.of} squads.`
            : `History covers the ${index.squads.done} of ${index.squads.of} squads cached so far.`}
          {!index.running && onBuildIndex && <> <button className="button small" type="button" onClick={onBuildIndex} disabled={running}>Build player index</button></>}
        </div>
      )}
      {history.data && history.data.unavailable_seasons.length > 0 && (
        <p className="small muted">{plural(history.data.unavailable_seasons.length, "season")} couldn't be fetched from UEFA.</p>
      )}
    </aside>
  );
}
