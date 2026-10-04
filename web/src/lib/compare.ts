// A comparison built in the browser, for the website: queries.compare on the server reads the same numbers that the
// two team-seasons' profiles carry (tests/test_web_publish.py checks that they agree).
import type { CompareRow, CompareSide, Comparison, FeatureMeta, Profile, Stat } from "../types";

export const TIE_Z = 0.05; // queries.TIE_Z: a smaller gap in oriented z counts as level

export function ahead(aZ: number | null, bZ: number | null): CompareRow["ahead"] {
  if (aZ === null || bZ === null || Math.abs(aZ - bZ) < TIE_Z) return "tie";
  return aZ > bZ ? "a" : "b";
}

function side(p: Profile): CompareSide {
  return { team_id: p.team_id, name: p.name, season: p.season, label: p.label, stage_label: p.stage_label, live: p.live };
}

/** z-scores flipped for lower-is-better stats, so that positive is always better (as the API sends them). */
export function compareProfiles(a: Profile, b: Profile, features: FeatureMeta[]): Comparison {
  const lower = new Set(features.filter((f) => f.lower_is_better).map((f) => f.feature));
  const oriented = (stat: Stat | undefined) =>
    stat == null || stat.z == null ? null : lower.has(stat.feature) ? -stat.z : stat.z;
  const ofB = new Map(b.features.map((stat) => [stat.feature, stat]));
  const rows = a.features.map((statA): CompareRow => {
    const statB = ofB.get(statA.feature);
    const aZ = oriented(statA), bZ = oriented(statB);
    return { feature: statA.feature, label: statA.label, section: statA.section, a_value: statA.value,
      b_value: statB?.value ?? null, a_z: aZ, b_z: bZ, a_beats: statA.beats_season, b_beats: statB?.beats_season ?? null,
      ahead: ahead(aZ, bZ) };
  });
  return { a: side(a), b: side(b), rows };
}
