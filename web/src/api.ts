import type { CheckReport, Comparison, Meta, ModelTest, PipelineState, PlayerHistory, Profile, Squad, Summary,
  TeamHit } from "./types";

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string) {
    super(message);
  }
}

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

export const api = {
  meta: () => request<Meta>("/api/meta"),
  summary: () => request<Summary>("/api/summary"),
  teams: (q: string) => request<TeamHit[]>(`/api/teams?q=${id(q)}`),
  profile: (teamId: string, season: number) => request<Profile>(`/api/teams/${id(teamId)}/seasons/${season}`),
  compare: (a: string, b: string) => request<Comparison>(`/api/compare?a=${id(a)}&b=${id(b)}`),
  squad: (teamId: string, season: number) => request<Squad>(`/api/squads/${id(teamId)}/${season}`),
  player: (playerId: string) => request<PlayerHistory>(`/api/players/${id(playerId)}`),
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

export const EVENTS_URL = "/api/pipeline/events";
