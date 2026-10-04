// The website's data source: the files `ucl publish` writes under /data, behind the same methods as the server's
// API (api.ts), so the screens can't tell them apart. Search, comparisons and head-to-heads are worked out here.
import { ApiError } from "./apiError";
import { compareProfiles } from "./lib/compare";
import { playMatchup, type Matchups } from "./lib/h2h";
import { formatPick, parsePick } from "./lib/route";
import { searchTeams, type TeamEntry } from "./lib/search";
import type { CheckReport, Comparison, Forecast, HeadToHead, Meta, ModelTest, PipelineEvent, PipelineState,
  PlayerHistory, Profile, Squad, Summary, TeamHit, Venue } from "./types";

export interface PipelineRecord { published_at: string; state: PipelineState; events: PipelineEvent[] }

const READ_ONLY = "This is the published website, so nothing runs here: the pipeline runs once a day on GitHub. "
  + "To run it yourself, clone the repo and start `uv run ucl web`.";
const id = encodeURIComponent;
const files = new Map<string, Promise<unknown>>();

/** One file, once per page load. A file that holds an error (as the API would have answered) becomes an ApiError. */
function file<T>(path: string, missing: string): Promise<T> {
  let pending = files.get(path) as Promise<T> | undefined;
  if (!pending) {
    pending = load<T>(path, missing);
    pending.catch(() => files.delete(path)); // a failed load is tried again next time
    files.set(path, pending);
  }
  return pending;
}

async function load<T>(path: string, missing: string): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/data/${path}`);
  } catch {
    throw new ApiError(0, "offline", "The site's data didn't load. Check your connection, then try again.");
  }
  if (response.status === 404) throw new ApiError(404, "not_found", missing);
  const body = await response.json().catch(() => null) as
    (T & { error?: { code: string; message: string }; status?: number }) | null;
  if (!response.ok || body === null) throw new ApiError(response.status, "http_error", `${response.status} ${response.statusText}`);
  if (body.error) throw new ApiError(body.status ?? 503, body.error.code, body.error.message);
  return body;
}

/** players/NN.json holds every player whose id ends in NN (publish.player_shard). */
export function playerShard(playerId: string): string {
  return playerId.slice(-2).padStart(2, "0");
}

function pick(text: string, which: string) {
  const parsed = parsePick(text);
  if (!parsed) throw new ApiError(422, "invalid_request", `${which}: expected team:season, like 52280:2026, not '${text}'`);
  return parsed;
}

const profile = (teamId: string, season: number) =>
  file<Profile>(`profiles/${id(teamId)}/${season}.json`, `no team ${teamId} in ${season - 1}-${String(season).slice(2)}`);

export const pipelineRecord = () => file<PipelineRecord>("pipeline.json", "This site has no record of a run yet.");

export const staticApi = {
  meta: () => file<Meta>("meta.json", "The site's data is missing."),
  summary: () => file<Summary>("summary.json", "The summary is missing."),
  teams: async (q: string): Promise<TeamHit[]> => searchTeams(await file<TeamEntry[]>("teams.json", "The team list is missing."), q),
  profile,
  compare: async (a: string, b: string): Promise<Comparison> => {
    const [pa, pb] = [pick(a, "a"), pick(b, "b")];
    const [meta, first, second] = await Promise.all([staticApi.meta(), profile(pa.teamId, pa.season), profile(pb.teamId, pb.season)]);
    return compareProfiles(first, second, meta.features);
  },
  squad: (teamId: string, season: number) =>
    file<Squad>(`squads/${id(teamId)}/${season}.json`, "No squad was saved for this club and season."),
  player: async (playerId: string): Promise<PlayerHistory> => {
    const shard = await file<Record<string, PlayerHistory>>(`players/${playerShard(playerId)}.json`,
      `no player ${playerId} in the cached squads`);
    const history = shard[playerId];
    if (!history) throw new ApiError(404, "not_found", `no player ${playerId} in the cached squads`);
    return history;
  },
  forecast: () => file<Forecast>("forecast.json", "The forecast is missing."),
  headToHead: async (a: string, b: string, venue: Venue): Promise<HeadToHead> => {
    const [pa, pb] = [pick(a, "a"), pick(b, "b")];
    const matchups = await file<Matchups>("matchups.json", "The ratings are missing.");
    const result = playMatchup(matchups, formatPick(pa), formatPick(pb), venue);
    if (!result) throw new ApiError(404, "not_found", `no rating for ${matchups.sides[formatPick(pa)] ? b : a}`);
    return result;
  },
  checks: () => file<CheckReport>("checks.json", "The nightly run's checks are missing."),
  testModel: (_load: boolean): Promise<ModelTest> => Promise.reject(new ApiError(405, "read_only", READ_ONLY)),
  pipelineState: async () => (await pipelineRecord()).state,
  startRun: (_body: { stages?: string[]; skip_ai?: boolean; refresh_live?: boolean }): Promise<{ run_id: string }> =>
    Promise.reject(new ApiError(405, "read_only", READ_ONLY)),
};
