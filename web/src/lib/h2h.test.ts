import { describe, expect, it } from "vitest";
import type { HeadToHeadSide, Venue } from "../types";
import fixture from "./h2h.fixture.json";
import { headToHead, playMatchup, type Matchups, type Models, type Outcome } from "./h2h";

function close(actual: unknown, expected: unknown, path: string): void {
  if (typeof expected === "number" && !Number.isInteger(expected)) {
    expect(Math.abs((actual as number) - expected), path).toBeLessThanOrEqual(1e-12 * Math.max(1, Math.abs(expected)));
  } else if (Array.isArray(expected)) {
    expect((actual as unknown[]).length, path).toBe(expected.length);
    expected.forEach((e, i) => close((actual as unknown[])[i], e, `${path}[${i}]`));
  } else if (expected !== null && typeof expected === "object") {
    expect(Object.keys(actual as object).sort(), path).toEqual(Object.keys(expected).sort());
    for (const [key, e] of Object.entries(expected)) close((actual as Record<string, unknown>)[key], e, `${path}.${key}`);
  } else {
    expect(actual, path).toEqual(expected);
  }
}

describe("head-to-head in the browser", () => {
  it("agrees with forecast.head_to_head on every case in the fixture", () => {
    for (const c of fixture.cases) {
      const actual: Outcome = headToHead(c.a, c.b, fixture.models as Models, fixture.home, c.venue as Venue, fixture.max_goals);
      close(actual, c.expected, `${c.a} v ${c.b} (${c.venue})`);
    }
  });

  it("answers like /api/forecast/h2h, without the exact ratings", () => {
    const side = (team_id: string, exact: number): HeadToHeadSide & { exact: number } =>
      ({ team_id, name: team_id, season: 2027, label: "2026-27", live: true, rating: Math.round(exact), title: 0.1, exact });
    const matchups: Matchups = { home: fixture.home, max_goals: fixture.max_goals, models: fixture.models as Models,
      title_label: "2026-27", sides: { "1:2027": side("1", 1750.4), "2:2027": side("2", 1650) } };
    const result = playMatchup(matchups, "1:2027", "2:2027", "a");
    expect(result?.a).toEqual({ team_id: "1", name: "1", season: 2027, label: "2026-27", live: true, rating: 1750, title: 0.1 });
    expect(result?.title_label).toBe("2026-27");
    expect(result!.win + result!.draw + result!.loss).toBeCloseTo(1, 12);
    expect(playMatchup(matchups, "1:2027", "3:2027", "neutral")).toBeNull();
  });
});
