import { describe, expect, it } from "vitest";
import type { HeadToHeadSide } from "../types";
import { chance, parseVenue, swapVenue, titleLine, versions } from "./predict";

const side = (team_id: string, name: string, title: number | null, season = 2027, live = true): HeadToHeadSide =>
  ({ team_id, name, season, label: season === 2027 ? "2026-27" : "2014-15", live, rating: 1600, title });

describe("chance", () => {
  it("rounds to whole percentages but never shows a near-certainty as certain", () => {
    expect([0.216, 0.004, 0.996, 0, 1, null].map(chance)).toEqual(["22%", "<1%", ">99%", "0%", "100%", "–"]);
  });
});

describe("parseVenue", () => {
  it("defaults to a neutral venue", () => {
    expect([parseVenue("a"), parseVenue("b"), parseVenue("x"), parseVenue(undefined)]).toEqual(["a", "b", "neutral", "neutral"]);
  });
});

describe("swapVenue", () => {
  it("keeps the match at the same ground when the teams swap sides", () => {
    expect([swapVenue("a"), swapVenue("b"), swapVenue("neutral")]).toEqual(["b", "a", "neutral"]);
  });
});

describe("titleLine", () => {
  const arsenal = side("1", "Arsenal", 0.138);
  const psg = side("2", "PSG", 0.176);
  it("names the club more likely to win", () => {
    expect(titleLine(arsenal, psg, "2026-27")).toBe("More likely to win the 2026-27 Champions League: PSG, 18% to Arsenal's 14%.");
  });
  it("copes with clubs that aren't in it, and with one club on both sides", () => {
    expect(titleLine(arsenal, side("3", "Ajax", null), "2026-27")).toBe("Arsenal: 14% to win the 2026-27 Champions League. Ajax isn't in it.");
    expect(titleLine(side("3", "Ajax", null), side("4", "Celtic", null), "2026-27")).toBe("Neither club is in the 2026-27 Champions League.");
    expect(titleLine(arsenal, side("1", "Arsenal", 0.138, 2015, false), "2026-27")).toBe("Arsenal: 14% to win the 2026-27 Champions League.");
    expect(titleLine(arsenal, side("5", "Inter", 0.141), "2026-27")).toBe("Arsenal and Inter are about as likely to win the 2026-27 Champions League: 14% each.");
    expect(titleLine(arsenal, psg, null)).toBeNull();
  });
});

describe("versions", () => {
  it("says which seasons' teams meet, unless both are this season's", () => {
    expect(versions(side("1", "Arsenal", 0.1), side("2", "PSG", 0.2))).toBeNull();
    expect(versions(side("1", "Barcelona", 0.1, 2015, false), side("2", "Real Madrid", 0.2))).toBe(
      "Barcelona as they finished 2014-15, Real Madrid as they are now.");
  });
});
