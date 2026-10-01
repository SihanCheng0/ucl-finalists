import type { SeasonInfo } from "../types";

/** Every season as one row of buttons; those the team didn't play are disabled and the live one is marked. */
export function SeasonScrubber({ seasons, available, value, onChange }: {
  seasons: SeasonInfo[]; available: Set<number>; value: number | null; onChange: (season: number) => void;
}) {
  return (
    <div className="scrubber" role="group" aria-label="Season">
      {seasons.map((s) => (
        <button key={s.season} type="button" className={s.live ? "live" : undefined}
                disabled={!available.has(s.season)} aria-pressed={s.season === value}
                title={s.live ? `${s.label}, in progress` : s.label} onClick={() => onChange(s.season)}>
          {s.label.slice(2)}
        </button>
      ))}
    </div>
  );
}
