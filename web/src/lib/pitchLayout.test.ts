import { describe, expect, it } from "vitest";
import type { Line, SquadPlayer } from "../types";
import { CAPS, colourValue, DEPTH, layoutSquad, R_MIN, R_SPAN } from "./pitchLayout";

let next = 0;
function player(position: Line | null, minutes: number, goals: number | null = 1): SquadPlayer {
  next += 1;
  return { player_id: String(next), name: `P${next}`, position, shirt: null, age: null, country: null, image_url: null,
           minutes, stats: { goals, passes_accuracy: minutes ? 80 : null } };
}
const GOALS = { key: "goals", kind: "count" };

describe("layoutSquad", () => {
  it("drops players without minutes and puts a missing position in midfield", () => {
    const { circles } = layoutSquad([player("FWD", 90), player(null, 45), player("DEF", 0)], GOALS, 1000, 600);
    expect(circles.map((c) => c.line)).toEqual(["MID", "FWD"]);
  });
  it("caps each line and counts the rest as overflow, most minutes first", () => {
    const defenders = Array.from({ length: 10 }, (_, i) => player("DEF", 100 + i));
    const { circles, overflow } = layoutSquad(defenders, GOALS, 1000, 600);
    expect(circles).toHaveLength(CAPS.DEF);
    expect(overflow).toEqual({ GK: 0, DEF: 2, MID: 0, FWD: 0 });
    expect(circles[0].player_id).toBe(defenders[9].player_id);
    expect(circles.every((c) => c.x === DEPTH.DEF * 1000 && c.y > 0 && c.y < 600)).toBe(true);
  });
  it("sizes circles by the square root of minutes within fixed bounds", () => {
    const { circles } = layoutSquad([player("GK", 1000), player("GK", 250)], GOALS, 1000, 600);
    expect(circles[0].r).toBeCloseTo(R_MIN + R_SPAN);
    expect(circles[1].r).toBeCloseTo(R_MIN + R_SPAN * 0.5);
  });
  it("gives every circle the same size when minutes aren't published", () => {
    const { circles } = layoutSquad([player("MID", 0), player("FWD", 0)], GOALS, 1000, 600, false);
    expect(circles).toHaveLength(2);
    expect(new Set(circles.map((c) => c.r)).size).toBe(1);
  });
  it("colours counts per 90, only from 90 minutes on, scaled to the squad's range", () => {
    const fast = player("FWD", 180, 4);
    const slow = player("FWD", 900, 4);
    const brief = player("FWD", 45, 1);
    expect(colourValue(fast, GOALS)).toBe(2);
    expect(colourValue(brief, GOALS)).toBeNull();
    const { circles } = layoutSquad([fast, slow, brief], GOALS, 1000, 600);
    const byId = Object.fromEntries(circles.map((c) => [c.player_id, c]));
    expect(byId[fast.player_id].value01).toBe(1);
    expect(byId[slow.player_id].value01).toBe(0);
    expect(byId[brief.player_id].value01).toBeNull();
  });
  it("uses the middle of the ramp when every value is the same, and raw values for rates", () => {
    const { circles } = layoutSquad([player("MID", 300), player("MID", 600)], { key: "passes_accuracy", kind: "rate" },
                                    1000, 600);
    expect(circles.map((c) => [c.value, c.value01])).toEqual([[80, 0.5], [80, 0.5]]);
  });
});
