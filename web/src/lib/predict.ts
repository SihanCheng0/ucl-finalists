// Wording and formatting for the Predict screen and the head-to-head card.
import { DASH } from "./format";
import type { HeadToHeadSide, StageKey, Venue } from "../types";

/** A probability for people: whole percentages, with "<1%" and ">99%" at the ends, so nothing that isn't certain
 * reads as certain. */
export function chance(p: number | null | undefined): string {
  if (p === null || p === undefined || !Number.isFinite(p)) return DASH;
  if (p <= 0) return "0%";
  if (p >= 1) return "100%";
  if (p < 0.005) return "<1%";
  if (p > 0.995) return ">99%";
  return `${Math.round(p * 100)}%`;
}

export const STAGE_COLUMNS: { key: StageKey; label: string; title: string }[] = [
  { key: "p_win", label: "Win", title: "Chance of winning the final" },
  { key: "p_final", label: "Final", title: "Chance of reaching the final" },
  { key: "p_sf", label: "Semis", title: "Chance of reaching the semi-finals" },
  { key: "p_qf", label: "Quarters", title: "Chance of reaching the quarter-finals" },
  { key: "p_r16", label: "Last 16", title: "Chance of reaching the round of 16" },
  { key: "p_top8", label: "Top 8", title: "Chance of a top-eight league-phase finish, straight into the round of 16" },
];

export function parseVenue(value: string | undefined): Venue {
  return value === "a" || value === "b" ? value : "neutral";
}

/** The same ground after the two teams swap sides. */
export function swapVenue(venue: Venue): Venue {
  return venue === "a" ? "b" : venue === "b" ? "a" : "neutral";
}

/** Which version of each club plays, or null when both are this season's: past seasons are rated as they ended. */
export function versions(a: HeadToHeadSide, b: HeadToHeadSide): string | null {
  if (a.live && b.live) return null;
  if (a.season === b.season) return `Both as they finished ${a.label}.`;
  const when = (s: HeadToHeadSide) => (s.live ? "as they are now" : `as they finished ${s.label}`);
  return `${a.name} ${when(a)}, ${b.name} ${when(b)}.`;
}

/** Which of two clubs is more likely to win this season's Champions League, in a sentence. Either club may not be
 * in it; null when there is no live season to talk about. */
export function titleLine(a: HeadToHeadSide, b: HeadToHeadSide, label: string | null): string | null {
  if (!label) return null;
  const competition = `the ${label} Champions League`;
  if (a.team_id === b.team_id) {
    return a.title === null ? `${a.name} isn't in ${competition}.` : `${a.name}: ${chance(a.title)} to win ${competition}.`;
  }
  if (a.title === null && b.title === null) return `Neither club is in ${competition}.`;
  if (a.title === null || b.title === null) {
    const [present, absent] = a.title === null ? [b, a] : [a, b];
    return `${present.name}: ${chance(present.title)} to win ${competition}. ${absent.name} isn't in it.`;
  }
  if (chance(a.title) === chance(b.title)) return `${a.name} and ${b.name} are about as likely to win ${competition}: ${chance(a.title)} each.`;
  const [lead, other] = a.title > b.title ? [a, b] : [b, a];
  return `More likely to win ${competition}: ${lead.name}, ${chance(lead.title)} to ${other.name}'s ${chance(other.title)}.`;
}
