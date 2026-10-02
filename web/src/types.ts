// Shapes of the UCL Lab API (spec §5). Everything here mirrors src/ucl/web/queries.py and app.py.

export type Status = "idle" | "running" | "done" | "warning" | "skipped" | "failed";

export interface SeasonInfo { season: number; label: string; live: boolean }
export interface FeatureMeta {
  feature: string; label: string; section: string; group: string;
  decimals: number; percent: boolean; lower_is_better: boolean;
}
export interface StageMeta { name: string; label: string; optional: boolean }
export interface PlayerStatMeta { key: string; label: string; kind: string; decimals: number; colourable: boolean }
export interface Meta {
  ready: boolean; problems: string[]; seasons: SeasonInfo[];
  live_season: number | null; live_status: "loading" | "ready" | "stale" | "unavailable"; live_message: string;
  sections: string[]; features: FeatureMeta[]; player_stats: PlayerStatMeta[]; stages: StageMeta[];
  data: { built_at: string | null; modelled_at: string | null; analysed_at: string | null };
}

export interface Metric { value: number; ci: [number, number] }
export interface Driver {
  feature: string; label: string; group: string; importance: number;
  kind: "robust" | "conditional" | "model-dependent" | null; helps: "higher" | "lower" | null;
}
export interface Badge { kind: "good" | "warn" | "stale" | "unavailable"; label: string }
export interface NarrativeData { text: string | null; html: string | null; badge: Badge }
export interface Summary {
  metrics: { spearman: Metric; auc: Metric; brier_skill: Metric; top4_share: Metric & { chance: number } };
  drivers: Driver[]; synthesis: NarrativeData | null;
}

export interface TeamHit { team_id: string; name: string; seasons: { season: number; label: string; stage_label: string }[] }
export interface Stat {
  feature: string; label: string; section: string; value: number | null; z: number | null;
  beats_season: number | null; beats_all: number | null; provisional: boolean;
}
export interface TrendPoint { season: number; label: string; value: number | null; beats_season: number | null; live: boolean }
export interface Contribution { feature: string; label: string; value: number | null; contribution: number }
export interface ModelCardData {
  p_final: number; base_rate: number; rank: number; ko_size: number; exp_stage: number; nearest_stage: string;
  top_up: Contribution[]; top_down: Contribution[];
}
export interface Profile {
  team_id: string; name: string; season: number; label: string; stage_label: string;
  live: boolean; live_available: boolean; stale: boolean; fetched_at: string | null;
  matches_played: number; phase_matches: number; provisional: boolean; ko_stage: number | null; result: string | null;
  seasons: { season: number; label: string }[];
  features: Stat[]; trend: Record<string, TrendPoint[]>; model: ModelCardData | null; narrative: NarrativeData | null;
}

export interface CompareSide { team_id: string; name: string; season: number; label: string; stage_label: string; live: boolean }
export interface CompareRow {
  feature: string; label: string; section: string; a_value: number | null; b_value: number | null;
  a_z: number | null; b_z: number | null; a_beats: number | null; b_beats: number | null; ahead: "a" | "b" | "tie";
}
export interface Comparison { a: CompareSide; b: CompareSide; rows: CompareRow[] }

export type Line = "GK" | "DEF" | "MID" | "FWD";
export interface SquadPlayer {
  player_id: string; name: string; position: Line | null; shirt: string | null; age: number | null;
  country: string | null; image_url: string | null; minutes: number; stats: Record<string, number | null>;
}
export interface Squad {
  team_id: string; name: string; season: number; label: string; live: boolean;
  players: SquadPlayer[]; minutes_published: boolean; stale: boolean; fetched_at: string | null;
}
export interface HistoryEntry {
  season: number; label: string; team_id: string; team: string; position: Line | null; minutes: number;
  stats: Record<string, number | null>; live: boolean;
}
export interface PlayerHistory {
  player: { player_id: string; name: string; position: Line | null; shirt: string | null; age: number | null;
            country: string | null; image_url: string | null };
  history: HistoryEntry[];
  index: { complete: boolean; squads: { done: number; of: number }; running: boolean };
  unavailable_seasons: string[]; stale: boolean; fetched_at: string | null;
}

export interface StageState { name: string; status: Status; started: number | null; finished: number | null; message: string }
export interface Counters {
  requests: { network: number; cache: number }; seasons: { done: number; of: number }; validated: boolean | null;
  folds: { done: number; of: number }; ablation: { done: number; of: number }; narratives: { done: number; of: number };
  players: { done: number; of: number };
}
export interface PipelineState { running: boolean; run_id: string | null; boot: string; stages: StageState[]; counters: Counters }

interface EventBase { id: string; run_id: string | null; ts: number; stage: string | null }
export type LogLevel = "info" | "warn" | "error";
export type PipelineEvent =
  | (EventBase & { type: "stage"; data: StageState })
  | (EventBase & { type: "progress"; data: { counters: Counters } })
  | (EventBase & { type: "log"; data: { level: LogLevel; text: string } })
  | (EventBase & { type: "done"; data: { status: "done" | "warning"; stages: { name: string; status: Status }[] } })
  | (EventBase & { type: "run_failed"; data: { stage: string | null; errors: string[] } });

export type StageKey = "p_win" | "p_final" | "p_sf" | "p_qf" | "p_r16" | "p_ko" | "p_top8";
export type ForecastTeam = {
  team_id: string; name: string; rating: number; played: number; points: number; goal_diff: number;
} & Record<StageKey, number>;
export interface TitleBacktest {
  season: number; label: string; winner_id: string; winner: string; p_winner: number; winner_rank: number | null;
  favourite_id: string; favourite: string; p_favourite: number; teams_left: number;
}
export interface TrackRecord {
  seasons: [string, string];
  matches: {
    matches: number; log_loss: number; rps: number; accuracy: number; draws_predicted: number; draws_seen: number;
    pedigree_log_loss: number; base_rate_log_loss: number;
  };
  ties: { ties: number; predicted: number; happened: number };
  titles: TitleBacktest[];
}
export interface Forecast {
  season: number; label: string; played: number; league_matches: number; as_of: string | null;
  fetched_at: string | null; stale: boolean; simulations: number; teams: ForecastTeam[]; track_record: TrackRecord;
}
export type Venue = "neutral" | "a" | "b";
export interface HeadToHeadSide {
  team_id: string; name: string; season: number; label: string; live: boolean; rating: number; title: number | null;
}
export interface HeadToHead {
  a: HeadToHeadSide; b: HeadToHeadSide; title_label: string | null; venue: Venue;
  win: number; draw: number; loss: number; xg_a: number; xg_b: number;
  likely_scores: { a: number; b: number; p: number }[]; tie: number; final: number;
}

export type CheckStatus = "ok" | "warn" | "fail" | "info";
export interface CheckItem { id: string; group: string; label: string; status: CheckStatus; detail: string; fix: string }
export interface CheckReport {
  checked_at: string; model: string; summary: Record<CheckStatus, number>; checks: CheckItem[];
}
export interface ModelTest {
  ok: boolean; detail: string; reply: string | null; load_seconds: number | null; answer_seconds: number | null;
}
