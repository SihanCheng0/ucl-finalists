import { describe, expect, it } from "vitest";
import type { FeatureMeta, Profile, Stat } from "../types";
import { ahead, compareProfiles } from "./compare";

const stat = (feature: string, value: number | null, z: number | null, beats: number | null): Stat =>
  ({ feature, label: feature.toUpperCase(), section: "Attack", value, z, beats_season: beats, beats_all: null, provisional: false });

function profile(team_id: string, features: Stat[]): Profile {
  return { team_id, name: `Team ${team_id}`, season: 2026, label: "2025-26", stage_label: "Final", live: false,
    live_available: false, stale: false, fetched_at: null, matches_played: 13, phase_matches: 8, provisional: false,
    ko_stage: 6, result: null, seasons: [], features, trend: {}, model: null, narrative: null };
}

const meta = (feature: string, lower: boolean): FeatureMeta =>
  ({ feature, label: feature, section: "Attack", group: "attack", decimals: 1, percent: false, lower_is_better: lower });

describe("comparisons in the browser", () => {
  it("calls a gap under 0.05 in oriented z level", () => {
    expect(ahead(0.5, 0.46)).toBe("tie");
    expect(ahead(0.5, 0.4)).toBe("a");
    expect(ahead(-1, 0)).toBe("b");
    expect(ahead(null, 1)).toBe("tie");
  });

  it("flips z for lower-is-better stats so that positive is always better", () => {
    const a = profile("1", [stat("goals", 2.1, 1.2, 30), stat("conceded", 0.6, -1.5, 28), stat("xg", null, null, null)]);
    const b = profile("2", [stat("goals", 1.4, -0.2, 12), stat("conceded", 1.1, 0.4, 9), stat("xg", 1.8, 0.3, 20)]);
    const result = compareProfiles(a, b, [meta("goals", false), meta("conceded", true), meta("xg", false)]);
    expect(result.a).toEqual({ team_id: "1", name: "Team 1", season: 2026, label: "2025-26", stage_label: "Final", live: false });
    expect(result.rows.map((r) => [r.feature, r.a_z, r.b_z, r.ahead])).toEqual([
      ["goals", 1.2, -0.2, "a"], ["conceded", 1.5, -0.4, "a"], ["xg", null, 0.3, "tie"]]);
    expect(result.rows[1]).toMatchObject({ a_value: 0.6, b_value: 1.1, a_beats: 28, b_beats: 9, label: "CONCEDED" });
  });
});
