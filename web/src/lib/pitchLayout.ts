// Where each player sits on the pitch (spec §7): four position lines, circles sized by minutes and coloured by
// the chosen stat. Pure, so it is tested without a browser.
import type { Line, PlayerStatMeta, SquadPlayer } from "../types";

export const LINES: Line[] = ["GK", "DEF", "MID", "FWD"];
export const CAPS: Record<Line, number> = { GK: 3, DEF: 8, MID: 8, FWD: 6 };
export const DEPTH: Record<Line, number> = { GK: 0.08, DEF: 0.3, MID: 0.55, FWD: 0.8 };
export const R_MIN = 9;
export const R_SPAN = 15;
const R_FLAT = 14;

export interface Circle {
  player_id: string; line: Line; x: number; y: number; r: number;
  value: number | null; // the colour value: per 90 for counts and distance, raw for rates and speeds
  value01: number | null; // value scaled to the squad's range; null when unknown
}

export function perNinety(stat: Pick<PlayerStatMeta, "kind">): boolean {
  return stat.kind === "count" || stat.kind === "distance";
}

export function colourValue(player: SquadPlayer, stat: Pick<PlayerStatMeta, "key" | "kind">): number | null {
  const raw = player.stats[stat.key];
  if (raw === null || raw === undefined) return null;
  if (!perNinety(stat)) return raw;
  return player.minutes >= 90 ? (raw / player.minutes) * 90 : null;
}

export function layoutSquad(players: SquadPlayer[], stat: Pick<PlayerStatMeta, "key" | "kind">, width: number,
                            height: number, minutesPublished = true) {
  const playing = minutesPublished ? players.filter((p) => p.minutes > 0) : players;
  const maxMinutes = Math.max(0, ...playing.map((p) => p.minutes));
  const overflow: Record<Line, number> = { GK: 0, DEF: 0, MID: 0, FWD: 0 };
  const circles: Circle[] = [];
  const margin = height * 0.1;
  for (const line of LINES) {
    const inLine = playing.filter((p) => (p.position ?? "MID") === line).sort((a, b) => b.minutes - a.minutes);
    const shown = inLine.slice(0, CAPS[line]);
    overflow[line] = inLine.length - shown.length;
    shown.forEach((player, i) => {
      const r = minutesPublished && maxMinutes > 0 ? R_MIN + R_SPAN * Math.sqrt(player.minutes / maxMinutes) : R_FLAT;
      circles.push({
        player_id: player.player_id, line, r,
        x: DEPTH[line] * width,
        y: margin + ((i + 0.5) * (height - 2 * margin)) / shown.length,
        value: colourValue(player, stat), value01: null,
      });
    });
  }
  const known = circles.map((c) => c.value).filter((v): v is number => v !== null);
  const low = Math.min(...known);
  const high = Math.max(...known);
  for (const circle of circles) {
    if (circle.value === null) continue;
    circle.value01 = high > low ? (circle.value - low) / (high - low) : 0.5;
  }
  return { circles, overflow, low: known.length ? low : null, high: known.length ? high : null };
}
