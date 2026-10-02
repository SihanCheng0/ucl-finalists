# UCL Lab — Local Dashboard for the UCL Finalists Project

**Date:** 2026-10-01
**Project:** `~/Projects/ucl-finalists` (extends the existing pipeline; base spec `2026-10-01-ucl-finalists-design.md`)
**Status:** Approved design, revision 3 after two spec reviews. The user chose: local web app with live pipeline,
any Champions League squad on demand, "squad on the pitch" animation.
**Prerequisite:** the base pipeline's `report` CLI stage (done at a9d1961).
**Delivery:** two plans (§10). Plan A covers the pipeline, history, Explore and Compare. Plan B covers the live
season, players and Squad.

## 1. Goal

A local web app (`uv run ucl web`) that lets the user:

1. **Watch the pipeline work**: fetch, build, model, analyze and report run live, with counters and an event log.
2. **Explore** any Champions League team-season:
   - Search plus a season picker, from 2011-12 to the live 2026-27 season.
   - Stats with "teams beaten" bars and trends across seasons.
   - The model card and the AI scouting report, where they exist.
3. **Compare** two team-seasons side by side.
4. **See a squad on an animated pitch**, and open any player's stats for this season and previous seasons, across clubs.

The design is modern and minimal, in light and dark.

## 2. Scope

**In:**
- the four screens
- a FastAPI backend over the existing outputs, plus live current-season and player data
- behaviour-preserving progress hooks
- shared stage functions for the CLI and the web runner
- tests and a visual check

**Out (YAGNI):**
- real player positions or match events (UEFA publishes no tracking data, so the pitch shows position lines)
- model predictions for the live season
- authentication, hosting, editing
- cancelling a running pipeline

## 3. Data

### 3.1 Historical (2011-12 → 2025-26)

The existing outputs:
- `data/processed/`: team_seasons, matches, finals, dataset.json with the **15 active features**
- `out/`: predictions, shap, drivers, ablation, metrics, finals_compare, analysis.json, report files

The web layer never writes these files directly. Only pipeline stages do, through the same functions the CLI uses (§4.2).

### 3.2 Live season (2026-27, `config.LIVE_SEASON = 2027`)

**When it's live:** a season is live when it is greater than `max(config.SEASONS)` and its final has not finished. The live season is **never** added to `config.SEASONS`, `team_seasons.csv`, `validate`, `plausibility_errors` or `model.run`.

**How the live snapshot is built:** `live.build_snapshot(client, history, max_age) -> LiveSnapshot` builds rows in memory, in the `team_seasons` schema.
- **Field:** `labels.match_rows` over **all** of the season's group or league-phase fixtures, whatever their status. Every team in the phase is listed, even before it has played. `labels.match_rows` itself is unchanged and gets no status column.
- **Matches for features:** only the fixtures with `status == "FINISHED"`, filtered on the raw dicts.
- **Stats:**
  - Fetched with `client.team_match_stats_many(ids, cache_missing=False, max_age=…)` on the usual 4 workers.
  - A missing or empty response is not cached, so the next refresh retries it.
  - A response is cached permanently only once the match finished (`fullTimeAt`) more than 24 h earlier, because UEFA may still be completing it. Before that, it is refetched when older than 6 h.
- **Coefficients:** `client.coefficients(2026)`, the ranking after 2025-26. It belongs to a completed season, so it is cached permanently.
- **Features:**
  - Built with `features.team_match_rows`, `season_features`, `attach_coefficients` and `zscore_within_season`, restricted to `ds.features`.
  - Missing values are imputed with the live season's median.
  - A team with no finished match keeps null features.
- **Percentiles:**
  - `pct_season` is computed within the live field, among teams with at least one finished match.
  - `pct_all` is the row's rank among the 488 historical raw values plus itself, using the average-rank rule from `add_percentiles`. So `beats_all` means "vs all CL teams since 2011-12".
  - **Provisional:** a team's live figures carry `provisional: true` until it has played the whole first phase (8 matches). The UI labels them "after N of 8 matches". For example, one win gives 3.0 points per game, which is above every historical season average.
- **No finished matches:** a snapshot whose teams all have `matches_played = 0` and null features. This is not an error.
- **`matches_played`:** every live row carries it.
- **`stale`:** the snapshot is marked stale when a refresh it needed failed and a cached copy was used instead.

**Ownership and refresh: `LiveService`.**
- The app holds one `LiveService`, which owns the live snapshot. It is not part of the disk-backed `Snapshot` (§4.3).
- `live.current() -> LiveState(snapshot | None, status, fetched_at)` never blocks and never touches the network. `status` is `loading | ready | stale | unavailable`.
- Refreshes are single-flight and run in a background thread. They happen:
  - when the app starts;
  - when a live-backed request finds the snapshot older than 6 h. The request gets the current copy immediately, and the refresh runs behind it (stale-while-revalidate).
  - in the pipeline's `live` stage. It forces `max_age = 0`, so "Refresh live season" always refetches, and it waits for the refresh to finish.
- **A failed refresh:**
  - The previous snapshot is kept with status `stale`. With no previous snapshot, the status is `unavailable`.
  - The next automatic attempt waits 5 minutes. The `live` stage ignores this backoff.
- **Historical routes never fetch.** They read `live.current()`. If it isn't ready, they omit the live point and set `live_available: false`.

### 3.3 Players

The building block is the **squad table**, one per (season, team). It comes from:

`GET https://compstats.uefa.com/v1/player-ranking?competitionId=1&seasonYear={Y}&phase=TOURNAMENT&order=DESC&optionalFields=PLAYER,TEAM&teamId={T}&limit=100&offset={k·100}&stats={PLAYER_STATS keys}`

- **Verified:** the `teamId` filter works. The response is a list of `{playerId, player:{id, internationalName, fieldPosition, clubJerseyNumber, age, imageUrl, …}, teamId, team:{…}, statistics:[{name, value}]}`.
- **Paging:** pages are fetched until one comes back empty, with a 30-page cap. This way a server-side limit below 100 can't truncate a squad.
- **Cache:** `data/raw/players/{season}/{team}.json`.
  - Completed seasons are permanent.
  - The live season is refetched when older than 6 h, and the stale copy is served meanwhile, as in §3.2.
- **Position:** `players.py` maps `fieldPosition` to `GK | DEF | MID | FWD`. When UEFA gives no position, the value is `null`.
- **Player history across clubs:** UEFA ignores the `playerId` filter, so history comes from cached squad tables.
  - `PlayerService` owns an in-memory inverted index, `player_id → [(season, team_id)]`.
  - The index is rebuilt from the cache directory at startup, and updated whenever a squad table is fetched.
- **Player index job:** fetches every squad of every season, about 524 tables (488 historical team-seasons plus 36 live).
  - It runs **only** as the pipeline stage `players`, through the runner's single slot, with 4 workers. Nothing starts it automatically.
  - Until it has run, a player's history lists only the squads already cached. The player panel shows "N of M squads indexed" and a "Build player index" button, which is disabled while any run is in progress.
  - On partial failure, the stage ends with `warning` and the history says "N seasons unavailable".
- **Seasons 2011-12 → 2014-15:** stat and minute coverage is unverified, so the implementation probes it first.
  - Missing stats are `null`.
  - If a whole squad lacks minutes, every circle uses the same radius and the panel says "minutes not published".
- **Plausibility:**
  - `PLAYER_STAT_RANGES` sets per-season and per-90 bounds, checked when a squad table is parsed.
  - The probe checks the `distance_covered` unit (km or m) against minutes played.
  - Implausible values become `null` and are logged.

**Player stat metadata** lives in new config, `PLAYER_STATS`: `key, label, kind, decimals, colourable`.

| key | kind | colourable |
|-|-|-|
| minutes_played_official | minutes | no (radius) |
| matches_appearance | count | no |
| goals, assists, key_passes, attempts, attempts_on_target | count | yes |
| passes_attempted, passes_completed | count | no |
| passes_accuracy | rate (%) | yes |
| distance_covered | distance (km, season total) | yes (per 90) |
| top_speed | speed (km/h) | yes |
| tackles, tackles_won, dribbling, dribbling_successful | count | tackles_won, dribbling_successful |
| fouls_committed, yellow_cards, red_cards | count | no |
| saves, clean_sheet, goals_conceded | count | saves |

**Per-90 rule:**
- `count` and `distance` stats are coloured per 90 minutes when minutes ≥ 90; otherwise the colour value is `null`.
- `rate` and `speed` stats are coloured raw.

## 4. Architecture

```
src/ucl/stages.py      # shared stage functions used by the CLI and the web runner (refactor of cli.cmd_*)
src/ucl/web/
  __init__.py
  app.py        # create_app(services) -> FastAPI; /api routes; static web/dist at /
  services.py   # Services(store, runner, live, players, bus), built once by `ucl web` and owned by the app
  store.py      # DataStore: immutable Snapshot(dataset, results, analysis, loaded_at) swapped atomically
  events.py     # EventBus: boot-scoped ids, ring buffer, bounded subscriber queues, SSE encoding, heartbeats, close()
  pipeline.py   # PipelineRunner: one run at a time; stage registry; counters; state snapshots
  queries.py    # pure: search, profile, trend, compare, summary (JSON-safe output)
  live.py       # build_snapshot (pure over a client) and LiveService (§3.2)
  players.py    # squad tables, PlayerService (cache and inverted index), index job (§3.3)
  jsonsafe.py   # NaN -> null, numpy -> python, ids as strings
web/                    # Vite + React + TypeScript SPA (npm)
  src/main.tsx, App.tsx, api.ts, hash routes #/pipeline #/explore #/compare #/squad
  src/components/  TopNav, TeamSearch, SeasonPicker, StatBars, Sparkline, MirrorBars,
                   Pitch, SquadTable, PlayerPanel, StageRail, EventLog
  src/lib/      pitchLayout.ts, format.ts, theme.css
```

**Clients.** `ucl web` builds two `UefaClient`s:
- **Background client:** the normal retry ladder. Used by the runner, `LiveService` and the index job.
- **Request client:** one attempt with an 8 s timeout. Used for on-demand squad fetches. A failure is remembered for 60 s, so repeated requests get a 503 at once instead of stalling.

**Process (`ucl web`):**
- **Network:** binds to `127.0.0.1` only, with `--port` (default 8787). The POST routes start jobs and LM Studio, and there is no authentication.
- **Missing build:** if `web/dist` is missing, it prints `cd web && npm install && npm run build` and exits 1.
- **Browser:** opens once the server accepts connections. `--no-open` skips this.
- **Shutdown:**
  - The runner thread is a daemon.
  - `EventBus.close()` ends open SSE streams.
  - uvicorn runs with `timeout_graceful_shutdown=2`, so Ctrl-C returns promptly even mid-run.
- **Help:** `ucl --help` mentions `web` in an epilog, because `web` is dispatched before the stage parser.

**Dependencies:**
- **Runtime:** `fastapi`, `uvicorn`.
- **Dev:** `httpx`, for `TestClient`.
- **Frontend (npm):** `react`, `react-dom`, `typescript`, `vite`, `@vitejs/plugin-react`, `vitest`.
- **No chart or animation library:** charts are SVG, and animation uses CSS transitions on `transform` and `fill`.

### 4.1 Changes to existing modules (behaviour-preserving)

All the changes below are optional parameters. Their defaults reproduce today's behaviour.

- **`UefaClient`:**
  - **Constructor:** `UefaClient(on_request=None, retry_delays=HTTP_RETRY_DELAYS_S, timeout=…)`.
  - **Request hook:** `on_request(kind, key, source)` is called once per resource, with `source` either "cache" or "network". Exceptions inside the hook are swallowed.
  - **New result type:** `Fetched(data, stale: bool, fetched_at: datetime | None)`, returned per call. The client keeps no mutable flags.
    - `fetched_at` is the time the cache file was written.
    - `stale` is true when a refetch was due and failed, so the cached copy was served.
  - **New `matches_fresh(season, max_age) -> Fetched`:** refetches when the cache is older than `max_age`. `matches(season)` is unchanged.
  - **Stats parameters:** `team_match_stats(id, cache_missing=True, max_age=None)` and `team_match_stats_many(ids, cache_missing=True, max_age=None)`. With `cache_missing=False`, missing or empty responses are not cached.
  - **New `squad(season, team_id, max_age=None) -> Fetched`:** paged and cached as §3.3 describes.
- **`dataset.fetch_all(..., progress=None, log=print)`:**
  - `progress` reports `{season, matches, phase_matches, missing, failed}` per season.
  - `log` carries the three kinds of lines it prints today: the per-season summary, the "e.g. match …: reason" examples, and the early-stop notice.
- **`model.run(..., progress=None)`:**
  - The main LOSO reports `{"fold": k, "of": 15}`.
  - Ablation reports `{"ablation_set": name, "k": i, "of": 7}`.
  - The `loso` calls inside `ablation` never report as folds.
- **`analyst.run(..., log=...)`:** unchanged. The runner counts log lines to get k/11. It also emits its own log line before `ensure_ready`, which can take minutes.
- **`cli.py`:**
  - **Characterization tests first.** Before any refactor, tests capture each command's stdout and exit code, on success and on failure: fetch, build, model, analyze, report and all. They must pass unchanged after the refactor.
  - The `cmd_*` functions call `stages.*` and keep their printing.
  - The stage functions call the module functions with the exact positional signatures the CLI uses today: `model.run(team_seasons, finals, features)`, `dataset.build(client, seasons)` and `analyst.save(analysis)`. New keyword arguments are passed only when they are not `None`, so the existing patches in `test_cli` keep working.
  - `ucl web` is **not** in `COMMANDS`. `main()` dispatches `web` before the stage parser.
  - `ucl all` is unchanged: fetch, build, model, analyze, report.

### 4.2 `stages.py`

Each stage function returns a result object and never prints. Messages go through `log`.

- `fetch(client, progress=None, log=print) -> FetchResult(failed: dict)`
- `build(client) -> BuildResult(dataset, notes)` (raises `ValidationError`)
- `model(dataset, progress=None) -> ModelResults` (saves `out/`)
- `analyze(dataset, results, llm_model, enabled, log, merge=False) -> Analysis`
- `report(dataset, results, analysis) -> Path`
- `live(live_service) -> LiveState` (forces a refresh with `max_age = 0`)
- `players(player_service, seasons_teams, progress=None) -> PlayerIndexResult`

**Saving `analysis.json`:**
- **With `merge=True` (the web runner):**
  - An `unavailable` result never overwrites the file.
  - An `ok` result where some narratives are `unavailable` is merged per key. Those keys keep their previous text, and the stage ends with `warning`, naming them.
- **With `merge=False` (the CLI):** today's behaviour, which saves every status.
- **Staleness:** `analysis.json` stores the facts each narrative was written from. The profile compares them with the current `facts.build_facts`. On a mismatch, the badge says "written for earlier numbers".

### 4.3 Pipeline runner

**Stages, in canonical order:** `fetch`, `build`, `model`, `analyze`, `report`, plus the optional `live` and `players`.

`POST /api/pipeline/runs` takes the body `{stages?: [...], skip_ai?: bool=false, refresh_live?: bool=false}`:
- **Stages:** the default is the five core stages. The listed stages run in canonical order.
- **`skip_ai`:** marks `analyze` as **skipped**. It doesn't run and `analysis.json` is untouched.
- **`refresh_live`:** appends `live`.
- **Errors:**
  - Unknown stage names: 422.
  - A missing prerequisite input on disk: 422 with the reason. For example, `model` needs `data/processed/dataset.json`.
  - A run already in progress: 409.

**Stage status:** `idle | running | done | warning | skipped | failed`.

| Outcome | Status | Run continues? |
|-|-|-|
| `fetch` with any failed ids | failed | no |
| `build` raises `ValidationError` | failed (the event lists every error) | no |
| any exception | failed | no |
| `analyze` returns `unavailable` | warning (the reason is shown); `analysis.json` kept | yes |
| `analyze` returns `ok` with some narratives unavailable | warning (the keys are named; their old text is kept) | yes |
| `live` with stale data | warning | yes |
| `players` with some squads failed | warning ("N squads unavailable") | yes |
| otherwise | done | yes |

**Concurrency:**
- A `threading.Lock` guards the start check, counters and stage states. `start()` checks and sets atomically.
- The run thread is a daemon and clears `running` in a `finally`.
- `state()` returns a deep-copied snapshot.
- Hooks and event publishing are wrapped so they can never change stage results. Payloads go through `jsonsafe` first.

**Counters (authoritative in `/state`):**

```
requests:   {network, cache}
seasons:    {done, of}        # of = len(config.SEASONS) = 15
validated:  bool | null
folds:      {done, of: 15}
ablation:   {done, of: 7}
narratives: {done, of: 11}
players:    {done, of}
```

Per-request hooks only bump counters. The runner publishes a coalesced `progress` event at most every 250 ms.

**Data reload:**
- After each successful writing stage, the `DataStore` swaps in a fresh immutable snapshot with one atomic assignment:
  - `build` reloads the dataset.
  - `model` reloads the results.
  - `analyze` reloads the analysis.
- The `live` and `players` stages update their own services.
- Each request reads `store.snapshot` once, so it never sees a mixed state.
- **Mixed generations:** after `build`, and before a successful `model` (or if `model` fails), the dataset is newer than the results.
  - `queries` tolerates this: rows missing from the results get `model: null`.
  - `meta.data` exposes `built_at`, `modelled_at` and `analysed_at`, so the UI can say "model outputs are older than the data; run model".

**Event bus:**
- Every event is `{id, run_id, ts, type, stage, data}`.
  - `id` is boot-scoped, `"{boot}-{n}"`. `boot` is fixed for each server process, and `n` increases monotonically.
  - `type` is `stage | progress | log | done | run_failed`. The name avoids `error`, which `EventSource` uses for connection errors.
- **Payloads:**

  | type | data |
  |-|-|
  | `stage` | `{name, status, started, finished, message}` |
  | `progress` | `{counters}` (the whole counters object) |
  | `log` | `{level: "info" \| "warn" \| "error", text}` |
  | `done` | `{status: "done" \| "warning", stages: [{name, status}]}` |
  | `run_failed` | `{stage, errors: [str]}` (every `ValidationError` message) |

- SSE frames carry `id:`, `event:` and `data:`.
- **Backlog:** a ring buffer of 500 events.
  - A new subscriber whose `Last-Event-ID` has the current boot gets the events after it. Otherwise it gets the whole buffer.
  - The backlog is taken under the lock, together with the subscriber's registration.
- **Client:** dedupes by `(boot, n)`. When the boot changes, it clears its log and refetches `/api/pipeline/state`.
- **Queues:** bounded at 2,000; when one is full, the oldest event is dropped.
- **Heartbeat:** a `: ping` comment every 15 s.
- The stream never closes on `done`. `EventBus.close()` ends every stream at shutdown.
- The SSE endpoint is a sync generator served on Starlette's threadpool, and unsubscribes in a `finally` when the client disconnects.

## 5. API (JSON)

**Conventions:**
- Every response is JSON-safe: NaN becomes `null`, numpy types become plain Python, and ids are strings.
- Errors are `{"error": {"code": "...", "message": "..."}}`.
- **Not ready:** while `meta.ready` is false, the data routes (summary, teams, profile, compare) return 503 with `{"error": {"code": "not_ready", "message": "Run the pipeline first: processed data or model outputs are missing."}}`. The meta and pipeline routes always work.
- Live-backed responses carry `stale: bool` and `fetched_at: iso | null`.

| Method & path | Response |
|-|-|
| `GET /api/meta` | `{ready, seasons:[{season, label, live}], live_season, live_status, features:[{feature, label, section, group, decimals, percent, lower_is_better}], player_stats:[PLAYER_STATS…], stages:[{name, label, optional}], data:{built_at, modelled_at, analysed_at}}`. `ready:false` when processed data or outputs are missing or corrupt; the app still starts, so the Pipeline screen can run |
| `GET /api/summary` | `{metrics:{spearman, auc, brier_skill, top4_share}, drivers:[…], synthesis}`. Each metric is `{value, ci:[lo, hi]}`. `drivers` is the top 6, `[{feature, label, group, importance, kind:"robust"\|"conditional"\|"model-dependent", helps:"higher"\|"lower"}]`. `synthesis` is a narrative (below) or `null` |
| `GET /api/teams?q=` | `q` shorter than 2 characters gives `[]`. Otherwise up to 12 matches, `[{team_id, name, seasons:[{season, label, stage_label}]}]`. Matching is case- and accent-insensitive over the display name, the international name and `config.TEAM_SEARCH_ALIASES` (e.g. "PSG", "Man City", "Barca"). Prefix matches rank first |
| `GET /api/teams/{team_id}/seasons/{season}` | Profile (below); 404 if the team didn't play that season |
| `GET /api/compare?a={team}:{season}&b={team}:{season}` | `{a, b, rows:[{feature, label, a_value, b_value, a_z, b_z, a_beats, b_beats, ahead:"a"\|"b"\|"tie"}]}`. `a_z` and `b_z` are **oriented**: the sign is flipped for lower-is-better stats, so positive is always better, and the UI draws them as given. `ahead` compares them; a tie is `\|Δ\| < 0.05`. A malformed parameter gives 422; an unknown team-season gives 404 |
| `GET /api/squads/{team_id}/{season}` | `{players:[{player_id, name, position:"GK"\|"DEF"\|"MID"\|"FWD"\|null, shirt, age, image_url, minutes, stats:{key: number\|null}}], minutes_published: bool, stale, fetched_at}`. A cached table is served at once. An uncached one is fetched on demand with the request client; 503 when UEFA is unreachable (§8) |
| `GET /api/players/{player_id}` | `{player:{player_id, name, position, shirt, age, image_url}, history:[{season, label, team_id, team, minutes, stats}], index:{complete, squads:{done, of}, running}, unavailable_seasons:[…], stale, fetched_at}`, newest first. `stale` and `fetched_at` describe the live season's squad when it is in the history |
| `POST /api/pipeline/runs` | 202 `{run_id}` / 409 / 422 (§4.3) |
| `GET /api/pipeline/state` | `{running, run_id, boot, stages:[{name, status, started, finished, message}], counters:{…}}` |
| `GET /api/pipeline/events` | SSE (§4.3) |

**Profile:**
- **Header fields:** `{team_id, name, season, label, live, matches_played, stage_label, ko_stage, live_available, stale, fetched_at}`.
- **`features`:** `[{feature, label, section, value, z, beats_season, beats_all, provisional}]`. "Beats" is `facts.beats`; a NaN gives `null`. `provisional` is true only for live rows (§3.2).
- **`trend`:** `{feature: [{season, label, value, beats_season, live}]}`.
- **`model`:** `{p_final, rank, ko_size, exp_stage, top_up:[≤4 {feature, label, value, contribution}], top_down:[≤3 …]}`, or `null`.
- **`narrative`:** `{text: markdown, html: report.md_to_html(...), badge:{kind:"good"|"warn"|"stale"|"unavailable", label}}`, or `null`. Narratives exist only for the 10 finalists of 2022–26. `stale` means the narrative was written for earlier numbers (§4.2).

## 6. Screens

**Shared:**
- A top nav: Pipeline · Explore · Compare · Squad.
- Selections live in the URL hash: `#/explore?t=52280&s=2026`, `#/compare?a=52280:2026&b=52747:2026`, `#/squad?t=…&s=…&stat=goals`.
- Typeahead search, keyboard navigable.
- A season scrubber over 16 seasons, with the live one marked. Seasons the team didn't play are disabled.

**Pipeline:**
- A stage rail showing statuses and elapsed times.
- A counter strip, with totals (for example 15/15 seasons, 7/7 ablation sets).
- A filterable event log that follows new events, with a pause toggle.
- **Controls:**
  - Run all, or a single stage.
  - Toggles for "Skip AI" (with the note "needs LM Studio; unchanged inputs replay from cache") and "Refresh live season".
  - "Build player index", which starts a run with `stages: ["players"]`.
  - All run buttons are disabled while a run is in progress.

**Explore:**
- Header: team, season, result. The live season shows "in progress, N of 8 matches", and provisional figures are marked.
- Stats grouped in sections (`config.STAT_SECTIONS`):

  | Section | Features |
  |-|-|
  | Results | points_pg, goal_diff_pg |
  | Attack | shots_pg, shot_accuracy, conversion, attacks_pg |
  | Defence | shots_against_pg, on_target_against_pg |
  | Control | possession_pct, pass_accuracy, passes_pg, long_pass_share |
  | Intensity | distance_km_pg, fouls_pg |
  | Pedigree | coef_log |

  Each stat is a "teams beaten" bar with a sparkline; the live point is hollow.
- A model card for knockout teams.
- An AI report with its badge, for finalists.
- When nothing is selected, an overview card from `/api/summary`: the headline metrics, the top drivers and the AI summary.

**Compare:** two pickers with a swap button. Mirrored z bars, with the centre at 0 and the better side always drawn outward; the leading side is emphasised. Values and "teams beaten" figures appear on both sides.

**Squad:**
- **Pitch:** an SVG pitch, landscape, with four lines (GK by its own goal, then DEF, MID, FWD).
- **Circles:** radius from √minutes; the fill comes from the stat ramp through theme tokens.
- **Animation:** changing team, season or stat animates position, size and colour over 450 ms. Players entering or leaving fade. `prefers-reduced-motion` disables the animation.
- **Interaction:** each circle is focusable, with an `aria-label` such as "No. 7 Bukayo Saka, 1,290 minutes, goals 0.31 per 90". A `SquadTable` twin lists the same data.
- **Player panel:** clicking or pressing Enter opens it.
  - It shows the photo, position, age, this season's stats, the history table across clubs, and a sparkline of the chosen stat.
  - While the index is incomplete, it shows "N of M squads indexed" and the "Build player index" button (§3.3). While the index job runs, it shows its progress.

**Look:**
- Minimal: generous whitespace, tabular figures, hairline rules.
- Light and dark tokens, following the report's dataviz rules: categorical colours for the 3 model groups, a single-hue ramp for magnitude, and neutral section headers.
- Usable down to 768px.

## 7. Pitch layout (pure, tested)

`layoutSquad(players, stat, width, height) -> {circles:[{player_id, x, y, r, value01}], overflow:{GK, DEF, MID, FWD}}`

`stat` is the `PLAYER_STATS` entry (`{key, kind, …}`), so the per-90 rule can apply.

1. Drop players whose minutes are 0. If minutes aren't published, keep everyone and set the radius to a constant.
2. Use each player's `position`. A `null` position goes to MID.
3. Sort each line by minutes, descending. Cap the lines at GK 3, DEF 8, MID 8, FWD 6. The extra players are counted in `overflow` for that line, with no cascading; the UI shows "+N" at the end of the line.
4. Each line sits at a fixed fraction of the pitch length: GK 0.08, DEF 0.30, MID 0.55, FWD 0.80. Players are spread evenly across the width, with margins.
5. `r = 9 + 15·√(minutes / max minutes)`.
6. `value01` is the per-90 or raw stat (§3.3), normalised to the squad's range; `null` when unknown. With no variation it is 0.5. The component maps `value01` to a fill using theme tokens (`color-mix` along the ramp), so the colours follow the theme.

## 8. Errors

- **UEFA unreachable:**
  - **Live data:** served from the last snapshot with `stale: true`. With no snapshot, the live points are omitted and `live_available` is false.
  - **Squads:** a cached table is served. For the live season, `stale: true` is set when a refresh is due and failing. An uncached table gets a 503 `uefa_unreachable` after one 8 s attempt, and the failure is remembered for 60 s.
  - **Historical routes** never touch the network.
- **Pipeline errors:** handled per §4.3. Outputs already written stay in place, and the store has already reloaded after each successful stage.
- **Bad input:**
  - unknown team or season: 404
  - malformed `compare` parameter: 422
  - invalid run body: 422
- **Not ready:** 503 `not_ready` (§5).
- **SSE disconnects** are dropped without affecting the run.

## 9. Testing & verification

**pytest:**
- **CLI characterization (written first):** stdout and exit code of fetch, build, model, analyze, report and all, on success and failure. They are captured before the refactor and stay unchanged after it.
- **Hooks:**
  - `on_request` distinguishes cache from network.
  - A hook that raises changes nothing.
  - `fetch_all(progress, log)` reports once per season, and routes all three kinds of line through `log`.
  - `model.run(progress)` reports 15 folds and 7 sets, with no folds from ablation.
  - `Fetched` is per call: two interleaved calls never share `stale` or `fetched_at`.
- **`stages.analyze`:**
  - `unavailable` never overwrites `analysis.json`.
  - Partial results merge per key.
  - Changed facts are flagged as stale.
- **`live`:**
  - the field includes teams with no finished match
  - features come only from `FINISHED` matches
  - zero finished matches gives a snapshot with all-null features
  - `pct_all` is computed against history with the average-rank rule
  - the provisional flag
  - nothing is written to `data/processed`
  - the stale fallback and the 5-minute backoff, with an injected clock
  - single-flight: two concurrent refreshes make one fetch
  - the 24 h cache rule for live stats
- **`players`:**
  - squad parsing, and paging until a page is empty
  - position mapping, including `null`
  - per-90 values and `minutes_published`
  - plausibility nulls
  - the inverted index built from fake squad tables
  - the index job's progress and partial failure
  - the request client's 503 and 60 s failure memory
- **`queries`:**
  - search: accents, aliases, prefix matches first, short `q`
  - profile shape, and NaN to `null`
  - trend with and without a live point
  - compare: oriented z, ahead and tie
  - mixed generations (a dataset newer than the results)
- **`PipelineRunner` and `EventBus`, with fake stages:**
  - order and canonicalisation
  - 409 when two runs race (threads)
  - failed stops the run
  - warning continues
  - skip
  - the `finally` reset and the daemon thread
  - payload shapes
  - backlog replay with `Last-Event-ID`, and the full replay on a boot mismatch
  - bounded queues
  - coalesced progress
  - `close()` ends streams
- **API through `TestClient`:** every route's shape and error codes, including 503 `not_ready`, with fakes.
- **`ucl web` process:**
  - a missing `web/dist` prints the build commands and exits 1
  - the server config binds `127.0.0.1`

**vitest:**
- `pitchLayout`:
  - caps and overflow, and ordering
  - radius bounds
  - per-90 by `stat.kind`
  - `value01` for constant and unknown values
  - a `null` position goes to MID
  - unpublished minutes
- `format`

**Housekeeping:**
- `.gitignore` gets `web/node_modules/` and `web/dist/`.
- `pyproject.toml` gets the dependencies listed in §4.
- The Vite dev proxy forwards `/api`, with SSE unbuffered.
- The browser preview config stays in the session workspace, not the repo.

**Verification before done.** Run `uv run ucl web` on the real data. In the browser pane:
1. Run all with "Skip AI" and watch the stages and counters live. Then confirm `git diff` shows no changes beyond timestamps.
2. Search "Arsenal", pick 2025-26, compare with PSG 2025-26.
3. Open the Arsenal squad, switch the stat, and watch the animation.
4. Build the player index, then open a player who has history at more than one club.
5. Check the live season's in-progress label.
6. Check both themes and a 768px width.

## 10. Delivery

Two plans, each producing working software on its own:
- **Plan A, pipeline and history:**
  - CLI characterization tests, the hooks, `stages.py`, `DataStore`, `PipelineRunner` and `EventBus`
  - the historical API: meta, summary, teams, profile, compare and the pipeline routes
  - the app shell and the Pipeline, Explore and Compare screens
  - verification steps 1, 2 and 6
- **Plan B, live season and players:**
  - `Fetched`, `LiveService` and the live fields in the API
  - squads, `PlayerService` and the `players` stage
  - the Squad screen and the player panel
  - verification steps 3 to 6
