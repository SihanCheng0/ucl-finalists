import { api } from "../api";
import { useApi } from "../hooks/useApi";
import type { Pick } from "../lib/route";
import { TeamSearch } from "./TeamSearch";

/** A club and one of its seasons. Picking a club jumps to its latest season. */
export function TeamPicker({ value, onChange, side, label }: {
  value: Pick | null; onChange: (pick: Pick) => void; side?: "a" | "b"; label: string;
}) {
  const profile = useApi(value ? () => api.profile(value.teamId, value.season) : null, [value?.teamId, value?.season]);
  const seasons = profile.data?.seasons ?? [];
  return (
    <div className={`picker${side ? ` ${side}` : ""}`}>
      <span className="small muted">{label}</span>
      <div className="picker-row">
        <TeamSearch initial={profile.data?.name ?? ""} placeholder="Search a club" label={label}
                    onPick={(hit) => onChange({ teamId: hit.team_id, season: hit.seasons[0].season })} />
        <select className="select" aria-label={`${label}: season`} value={value?.season ?? ""} disabled={!seasons.length}
                onChange={(e) => value && onChange({ teamId: value.teamId, season: Number(e.target.value) })}>
          {[...seasons].reverse().map((s) => <option key={s.season} value={s.season}>{s.label}</option>)}
        </select>
      </div>
      {profile.error && <p className="error-note">{profile.error.message}</p>}
    </div>
  );
}
