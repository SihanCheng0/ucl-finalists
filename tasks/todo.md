# UCL Finalists — task list

Plan: docs/plans/2026-10-01-ucl-finalists.md
Spec: docs/specs/2026-10-01-ucl-finalists-design.md

- [x] Chunk 1: scaffold, config, UEFA client
- [x] Chunk 2: labels, features, dataset
- [x] Chunk 3: fetch + build on real data, models, drivers, ablation, finals comparison
- [x] Chunk 4: grounding checks, LM Studio client
- [x] Chunk 5: fact sheets, analyst, analyze on the real data
- [x] Chunk 6: charts and the report
- [x] Chunk 7: end-to-end run, verification, publish

## Data notes (real build, cleaned)

- 488 team-seasons (256 knockout) with **15 features**: the coverage rule dropped `save_pct` (UEFA's 2011-12 feed lacks `saves` for 26 of 192 team-matches, 86.5% < 90%; the feed omits some counters instead of writing 0, so zero-filling would be wrong). `dataset.json` records the active 15 under `features`; downstream stages use that list, not `config.FEATURES`.
- Fix A cleaned the feed: possession in seconds and distance in metres converted, partial tracking (a match distance under 80 km) and placeholder zeros masked, Steaua's match-feed id aliased to FCSB (50065). A plausibility gate fails the build on out-of-range season averages.
- 4 team-matches have no usable stats (2015-16: Wolfsburg-CSKA returns nothing; Sevilla-Mönchengladbach has only distance covered, so it counts as having no stats); 0 values imputed.
- 11 coefficients imputed (clubs absent from the previous ranking: Málaga 2013, Real Sociedad 2014, Monaco 2015, Leicester 2017, Leipzig 2018, Lens and Newcastle 2024, Brest, Girona, Stuttgart and Bologna 2025). None of the 10 finalists is imputed.
- All 10 finalists' league/group points and goal difference match UEFA standings (`scripts/check_standings.py`).

## Model notes (real run, cleaned data)

- 15 leave-one-season-out folds, 256 knockout team-seasons (30 finalists), `config` hyperparameters (not tuned). Intervals are 95% season-bootstrap (1000 resamples, seed 42; the ablation rows use the same, so the "all" row equals the headline). `uv run ucl model` takes about 40 s.
- Spearman 0.45 [0.34, 0.54], AUC 0.73 [0.61, 0.83], Brier skill 0.05 [-0.04, 0.13] (Brier 0.098 vs 0.103 for the base rate: the edge is inconclusive), finalists in the top 4 by expected stage 47% [30%, 60%] against 24% for a random ranking. Modest numbers; the report should say so.
- Ablation (Spearman / AUC): pedigree 0.27 / 0.66, results 0.30 / 0.72, style 0.39 / 0.71, all 0.45 / 0.73. The intervals overlap, so differences between feature sets are tentative.
- Top-6 drivers: attacks_pg (robust), goal_diff_pg (robust), long_pass_share (conditional), passes_pg (robust), fouls_pg (model-dependent), conversion (conditional). Conditional means the direction holds only with the other stats held fixed: long_pass_share and conversion do not point the same way on their own (Spearman with stage -0.28 and 0.07).

## Review

- **Delivered.** `uv run ucl all` runs fetch → build → model → analyze → report end to end. `analyze` replays unchanged requests from `data/llm_cache`, so reruns are byte-identical. 290 tests pass.
- **Metrics on the real data:**
  - Spearman 0.45 [0.34, 0.54]
  - AUC 0.73 [0.61, 0.83]
  - Brier 0.098, against 0.103 for the base rate. The skill, 0.05 [-0.04, 0.13], is inconclusive.
  - Finalists in the top 4: 47% [30%, 60%], against 24% by chance.

  Group-stage play only modestly predicts deep runs.
- **Drivers.**
  - Robust: attacks, goal difference and passes per game.
  - Conditional: long-pass share and shot conversion.
  - Model-dependent: fouls.

  Pedigree adds nothing once play is known (Spearman 0.45 with everything, 0.44 without pedigree), though the intervals overlap.
- **Picks.** 3 of the 10 finalists ranked 1st or 2nd in their season: Liverpool 2021-22, Real Madrid 2023-24 and Arsenal 2025-26, all #2. Inter 2022-23 ranked #14 of 16.
- **Dropped and imputed.** The coverage rule dropped `save_pct`. 4 team-matches have no stats, and 11 coefficients were imputed, none for a finalist. See Data notes.
- **AI write-ups.**
  - All 11 narratives are ok, and 0 figures are missing from the data, so all 12 badges in the report are green. That is 10 cards, plus the synthesis split across two sections.
  - Fixes D to G narrowed the wording:
    - plain-English facts
    - the "teams beaten" framing
    - format names
    - length
    - the final placed inside the first phase
- **Left as is.** These are wording slips only; every figure is correct.
  - Liverpool 2021-22 still says it lost "a 0-1 decider during the group stage". Its one retry repeated the sentence, and its `style` field records it.
  - Dortmund 2023-24 "reached the 2023-24 Champions League final after navigating the group stage, where they ultimately lost to Real Madrid 0-2". The "where" is ambiguous.
  - Real Madrid 2021-22 says "the club's actual probability", meaning the model's, and kept "most significantly" after its one retry.
  - Inter 2022-23 says "12.9% of their total plays" about long passes.
- **Not in this plan.** Publishing is done by hand, not by the pipeline. The interactive dashboard (UCL Lab) has its own spec and plans.

# UCL Lab — local dashboard (spec: docs/specs/2026-10-01-ucl-lab-dashboard-design.md, rev 3)

Plan A: docs/plans/2026-10-01-ucl-lab-plan-a.md (chunks 1–5 reviewed; the frontend and Plan B were
built straight from the spec overnight, as asked).

- [x] CLI characterization tests (tests/golden/cli), progress hooks, stages.py shared by the CLI and the web runner
- [x] DataStore, EventBus (boot-scoped SSE), PipelineRunner, real stage wiring, queries, FastAPI app, `ucl web`
- [x] Live 2026-27 season (in memory, background refresh) and players (squads on demand, player index, histories)
- [x] React SPA: Pipeline, Explore, Compare, Squad (animated pitch, player panel)
- [x] Verification in the browser pane (below)

## Review (UCL Lab)

- Tests: 436 Python (incl. 14 CLI goldens) and 20 vitest. `npm --prefix web run build` type-checks with TypeScript 7.
- Verified in the browser pane:
  - Run all with "Skip AI" and "Refresh live season" from the Pipeline screen: fetch 0.6 s, build 0.9 s, model
    36 s (Spearman 0.45, AUC 0.73), analyze skipped, report, live (36 teams, 18 finished matches). `git status`
    clean afterwards, so the UI run regenerates every committed output byte for byte.
  - Player index: 524 of 524 squads in 88 s, none failed. Gabriel Jesus's history spans Manchester City
    (2017-18 to 2021-22), Arsenal (2023-24 to 2025-26) and Barcelona (2026-27, live).
  - Explore: search "arsenal", live 2026-27 profile ("in progress, 1 match of 8", early-season flags), 2025-26
    with model card and AI report. Compare Arsenal vs PSG 2025-26. Squad Arsenal 2025-26 coloured by goals and
    by distance, player panel.
  - Dark and light themes; no horizontal scroll at 473 px and 768 px.
- Reviews that changed the code: runner hangs on a bad stage result, final event before the run ended, 409 vs 422
  order; event payloads validated at publish; ø/æ in search; Ctrl-C tracebacks and slow shutdown with an open
  event stream; the SPA page served uncached.
- Not done: the live season's model card (the model only scores completed seasons, by design); the live stage's
  requests aren't counted in the run counters (its client is the live service's own).

# Shared GitHub repo for the team (SihanCheng0/ucl-finalists, public, code under MIT)

- [x] Analyze replays saved answers before touching LM Studio, so a clone with the committed cache rebuilds the
      write-ups offline (`analyst.run`: offline replay first, `ensure_ready` only on a cache miss)
- [x] Specs and plans copied into `docs/`; light-mode screenshots in `docs/images/` (`scripts/screenshots.py`)
- [x] README for teammates, `docs/how-it-works.md`, `docs/README.md`, `CONTRIBUTING.md`, `Makefile`
- [x] CI: Python tests, frontend tests and build (`.github/workflows/ci.yml`)
- [x] Data committed: `data/raw` (2,110 UEFA responses, 267 MB), `data/llm_cache` (165 answers), processed data, `out/`

## Review (GitHub repo)

- Replay checked on the real data with LM Studio blocked (no server start, no request): status ok, all 11 write-ups
  identical to `out/analysis.json`.
- Squad screen keeps the selected player in the URL (`&p=`), so a link opens straight to a player.
- `out/report*.html` regenerated from the committed `analysis.json`; `uv run ucl report` reproduces them byte for byte.
- Made public on 2026-10-02 at the user's request. Before that, the history was rewritten so every commit uses the
  GitHub noreply address instead of a personal email, and the code got an MIT license (UEFA's data excluded).


# Predictions: title odds and head to head (2026-10-02)

- [x] `src/ucl/forecast.py`: Elo ratings from every match since 2011-12 (season pull toward the coefficient prior),
      Poisson goals models for the league phase, early and late knockouts, analytic head-to-heads, and a vectorised
      simulation of a season under UEFA's format (league table, play-off and round-of-16 seeding, bracket halves)
- [x] Tuning on 2013-14 to 2018-19, test on 2019-20 to 2025-26 (`scripts/tune_forecast.py`), track record in the UI
- [x] `/api/forecast` and `/api/forecast/h2h`, rebuilt when the dataset or the live season changes
- [x] Predict screen (title odds, head to head, track record) and the "If they met" card on Compare

## Review (predictions)

- Test seasons: likeliest result right in 58% of 997 matches; log loss 0.925 against 1.008 (coefficient only) and 1.040
  (base rates). Two-legged ties: favourites went through 67%, predicted 69%. Title odds when the knockouts began: the
  winner had 11% on average (6% for a random pick); the favourite won 0 of 7.
- One goals model overrated favourites from the quarter-finals on (predicted 65%, actual 54%); separate stage models
  fixed most of it (62% against 57%) and improved out-of-sample log loss. Damping big margins or weighting knockout
  results more changed little.
- Known bias: ratings see only Champions League matches, so group-stage dominance rates highly (Bayern was the
  favourite in 5 of the 7 test seasons). The track record card says so.

# Hosted model and website (OpenRouter, Vercel, a nightly GitHub Actions run)

- [x] llm.py: one cache-first base for LM Studio and OpenRouter (same question, same cache key, so saved answers
      replay on either); OpenRouter keeps reasoning off and reports tokens and cost; `make_llm` picks the provider
- [x] analyst, stages, cli, report: provider-neutral wording; log the tokens and cost of new answers
- [x] checks: an "AI model" group for whichever provider is in use (OpenRouter: key, account, model served)
- [x] forecasts: the head-to-head inputs as data, and a fixture that pins the browser's port of the math
- [x] `ucl publish`: run stages headless, record the run, save every page's data as static JSON for Vercel
- [x] web: a static data source (search, compare, head-to-head in the browser), the nightly run replayed on the
      Pipeline screen, read-only controls; `npm run build:site`
- [x] .github/workflows/nightly.yml: refresh, checks, deploy to Vercel, commit new saved answers
- [x] docs: README (site, nightly run, token costs, setup), CONTRIBUTING, how-it-works
- [x] verify: pytest, vitest, both builds, a local preview of the static site
- [ ] deploy and push, each after the user's OK

## Review

- Token cost, measured from the last full analyze run (the 14 qwen/qwen3.8-27b cache entries, counted with the
  model's own tokenizer): 44,313 input and 4,752 output tokens. On OpenRouter's bf16/fp16/fp8 providers that is
  $0.015 (Ionstream) to $0.051 (Cerebras) for all 11 write-ups. A nightly run asks nothing: the write-ups cover
  finished seasons and replay from the cache. For comparison (Anthropic list prices, 2026-09-25, up to 35% more
  tokens for Claude's tokenizer): Haiku 4.5 $0.07-0.09, Sonnet 5.5 $0.14-0.18, Opus 5.5 $0.30-0.50 per full run.
- The user picked Qwen3.8-27B on OpenRouter: the same model id, so the 14 saved answers still replay.
- The website is a static export, not a Python function: the API's GET routes are saved as JSON (1,155 files,
  28 MB) and the browser does search, comparisons and head-to-heads. Checked in a local preview: the browser's
  head-to-head (Bayern v PSG, neutral) matched the server's to the displayed precision (41/25/34, 56%, 52%, 1-1).
- The nightly job runs on GitHub Actions rather than Vercel Cron: Hobby cron only calls a short function, which
  can't load pandas, wait on UEFA and keep files. `ucl publish --run live,players` took 12 s locally.
- Data times come from file mtimes, which a checkout resets, so the workflow dates the outputs by their last commits.
- Python 525 passed, web 39 passed, both builds pass.
