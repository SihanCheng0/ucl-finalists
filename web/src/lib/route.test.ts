import { describe, expect, it } from "vitest";
import { formatHash, formatPick, parseHash, parsePick } from "./route";

describe("route", () => {
  it("reads the screen and its parameters from the hash", () => {
    expect(parseHash("#/compare?a=52280:2026&b=52747%3A2026")).toEqual(
      { screen: "compare", params: { a: "52280:2026", b: "52747:2026" } });
    expect(parseHash("")).toEqual({ screen: "explore", params: {} });
    expect(parseHash("#/nonsense?t=1")).toEqual({ screen: "explore", params: { t: "1" } });
    expect(parseHash("#/squad?t=52280&s=2026&stat=goals").params.stat).toBe("goals");
  });
  it("writes readable hashes and drops empty parameters", () => {
    expect(formatHash({ screen: "compare", params: { a: "52280:2026", b: "" } })).toBe("#/compare?a=52280:2026");
    expect(formatHash({ screen: "pipeline", params: {} })).toBe("#/pipeline");
  });
  it("round-trips team-season picks", () => {
    expect(parsePick("52280:2026")).toEqual({ teamId: "52280", season: 2026 });
    expect(formatPick({ teamId: "52280", season: 2026 })).toBe("52280:2026");
    for (const bad of [undefined, "", "52280", "52280:", ":2026", "52280:x", "1:2026:3"]) expect(parsePick(bad)).toBeNull();
  });
});
