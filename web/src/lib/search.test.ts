import { describe, expect, it } from "vitest";
import { fold, searchTeams, type TeamEntry } from "./search";

function team(team_id: string, name: string, latest: number, ...others: string[]): TeamEntry {
  return { team_id, name, names: [name, ...others].map(fold), key: fold(name), latest,
    seasons: [{ season: latest, label: `${latest - 1}-${String(latest).slice(2)}`, stage_label: "Final" }] };
}

describe("team search in the browser", () => {
  it("folds case, accents and punctuation like queries.fold", () => {
    expect(fold("Atlético  Madrid")).toBe("atletico madrid");
    expect(fold("B. Dortmund")).toBe("b dortmund");
    expect(fold("Bodø/Glimt")).toBe("bodo glimt");
    expect(fold("Gladbach Straße")).toBe("gladbach strasse");
    expect(fold("Beşiktaş")).toBe("besiktas");
  });

  it("puts names that start with the query first, then the latest season, then the name", () => {
    const index = [team("1", "Real Sociedad", 2020), team("2", "Real Madrid", 2026), team("3", "Bayer Leverkusen", 2026, "leverkusen"),
      team("4", "Arsenal", 2026), team("5", "Real Betis", 2026)];
    expect(searchTeams(index, "real").map((t) => t.team_id)).toEqual(["5", "2", "1"]);
    expect(searchTeams(index, "kusen").map((t) => t.team_id)).toEqual(["3"]);
    expect(searchTeams(index, "eal").map((t) => t.team_id)).toEqual(["5", "2", "1"]);
    expect(searchTeams(index, "r")).toEqual([]);
    expect(searchTeams(index, "réal", 1)).toEqual([{ team_id: "5", name: "Real Betis", seasons: index[4].seasons }]);
  });
});
