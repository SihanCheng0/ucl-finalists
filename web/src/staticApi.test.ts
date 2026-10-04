import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "./apiError";
import { playerShard, staticApi } from "./staticApi";

function serve(files: Record<string, unknown>) {
  vi.stubGlobal("fetch", vi.fn(async (url: string) => {
    const path = url.replace(/^\/data\//, "");
    return path in files ? new Response(JSON.stringify(files[path]), { status: 200 }) : new Response("", { status: 404 });
  }));
}

afterEach(() => vi.unstubAllGlobals());

describe("the website's data source", () => {
  it("finds a player in the file named for the last two characters of the id", async () => {
    expect(playerShard("250010802")).toBe("02");
    expect(playerShard("7")).toBe("07");
    serve({ "players/02.json": { "250010802": { player: { name: "A" } } } });
    expect((await staticApi.player("250010802")).player.name).toBe("A");
    await expect(staticApi.player("99999902")).rejects.toMatchObject({ status: 404, code: "not_found" });
  });

  it("turns a saved error into the ApiError the server would have sent", async () => {
    serve({ "forecast.json": { error: { code: "forecast_unavailable", message: "The live season isn't loaded yet" }, status: 503 } });
    const error = await staticApi.forecast().catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ status: 503, code: "forecast_unavailable", message: "The live season isn't loaded yet" });
  });

  it("says a missing team-season isn't there, and that runs don't happen here", async () => {
    serve({});
    await expect(staticApi.profile("1", 2026)).rejects.toMatchObject({ status: 404, message: "no team 1 in 2025-26" });
    await expect(staticApi.startRun({})).rejects.toMatchObject({ code: "read_only" });
    await expect(staticApi.compare("bad", "1:2026")).rejects.toMatchObject({ status: 422 });
  });
});
