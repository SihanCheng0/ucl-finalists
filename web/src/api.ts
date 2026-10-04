import { ApiError } from "./apiError";
import { STATIC_SITE } from "./site";
import { staticApi } from "./staticApi";
import type { CheckReport, Comparison, Forecast, HeadToHead, Meta, ModelTest, PipelineState, PlayerHistory, Profile,
  Squad, Summary, TeamHit, Venue } from "./types";

export { ApiError };

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch {
    throw new ApiError(0, "offline", "The UCL Lab server isn't answering. Is `uv run ucl web` still running?");
  }
  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const error = (body as { error?: { code: string; message: string } } | null)?.error;
    throw new ApiError(response.status, error?.code ?? "http_error", error?.message ?? `${response.status} ${response.statusText}`);
  }
  return body as T;
}

const id = encodeURIComponent;

const serverApi = {
  meta: () => request<Meta>("/api/meta"),
  summary: () => request<Summary>("/api/summary"),
  teams: (q: string) => request<TeamHit[]>(`/api/teams?q=${id(q)}`),
  profile: (teamId: string, season: number) => request<Profile>(`/api/teams/${id(teamId)}/seasons/${season}`),
  compare: (a: string, b: string) => request<Comparison>(`/api/compare?a=${id(a)}&b=${id(b)}`),
  squad: (teamId: string, season: number) => request<Squad>(`/api/squads/${id(teamId)}/${season}`),
  player: (playerId: string) => request<PlayerHistory>(`/api/players/${id(playerId)}`),
  forecast: () => request<Forecast>("/api/forecast"),
  headToHead: (a: string, b: string, venue: Venue) =>
    request<HeadToHead>(`/api/forecast/h2h?a=${id(a)}&b=${id(b)}&venue=${venue}`),
  checks: () => request<CheckReport>("/api/checks"),
  testModel: (load: boolean) => request<ModelTest>("/api/checks/model", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ load }),
  }),
  pipelineState: () => request<PipelineState>("/api/pipeline/state"),
  startRun: (body: { stages?: string[]; skip_ai?: boolean; refresh_live?: boolean }) =>
    request<{ run_id: string }>("/api/pipeline/runs", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    }),
};

export type Api = typeof serverApi;
/** The local server's API, or on the website the files `ucl publish` wrote (staticApi.ts). */
export const api: Api = STATIC_SITE ? staticApi : serverApi;

export const EVENTS_URL = "/api/pipeline/events";
