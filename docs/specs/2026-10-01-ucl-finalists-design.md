# UCL Finalists — What Makes the Best Teams Win

**Date:** 2026-10-01
**Project:** `~/Projects/ucl-finalists`
**Status:** Approved design (approach 1: predictive model + local-AI analyst). Revision 4: refinements from plan review (stats id aliasing, P(final) rescaling, 404/410-only missing markers, `facts.py` split).

## 1. Goal

Gather the 10 finalist sides of the last five UEFA Champions League finals and explain, with a
validated model and a local LLM, why the best teams perform best.

Questions the output answers:

1. Would a model that knows a team's group/league-phase play have picked it as a finalist? ("Would the model have picked them?")
2. Which factors drive deep knockout runs, and is it **pedigree** (who you are), **results** (points, goal difference) or **style** (how you play)?
3. What specifically powered each of the 10 finalists?
4. What, if anything, separated winners from runners-up?

## 2. Scope

**In:** 15 seasons of UEFA data (2011-12 → 2025-26), feature engineering, two validated models with SHAP
explanations, a local-LLM analyst with numeric grounding checks, a static HTML report, a one-command pipeline.

**Out (YAGNI):** xG (UEFA does not publish it historically), player-level data, predictions for the live
2026-27 season, hyperparameter tuning (fixed, documented hyperparameters instead — the dataset is small),
a web app.

## 3. The 10 finalists (expected values — used as build assertions, matched by team id)

| seasonYear | Season | Winner (id) | Runner-up (id) | Final |
|-|-|-|-|-|
| 2022 | 2021-22 | Real Madrid (50051) | Liverpool (7889) | 1-0, Saint-Denis |
| 2023 | 2022-23 | Man City (52919) | Inter (50138) | 1-0, Istanbul |
| 2024 | 2023-24 | Real Madrid (50051) | B. Dortmund (52758) | 2-0, London |
| 2025 | 2024-25 | Paris (52747) | Inter (50138) | 5-0, Munich |
| 2026 | 2025-26 | Paris (52747) | Arsenal (52280) | 1-1, Paris won 4-3 on penalties, Budapest |

Names are UEFA's `internationalName` values ("Paris" = PSG, "B. Dortmund"). Each finalist is a
**team-season** (Real Madrid 2021-22 and 2023-24 are separate entries). Finalists are derived from the data
and asserted against this table **by team id**.

## 4. Data sources (UEFA public JSON APIs — verified reachable 2026-10-01)

| Data | Endpoint | Fields used |
|-|-|-|
| Matches | `https://match.uefa.com/v5/matches?competitionId=1&seasonYear={Y}&phase=TOURNAMENT&limit=500&offset=0&order=ASC` | `id`, `round.metaData.name`, `homeTeam`/`awayTeam` (`id`, `internationalName`), `score.total`, `score.penalty`, `winner.match.team.id`, `stadium.city.translations.name.EN` |
| Per-match team stats | `https://matchstats.uefa.com/v2/team-statistics/{matchId}` | list of 2 entries: `teamId`, `statistics[{name, value}]` |
| Club coefficients | `https://comp.uefa.com/v2/coefficients?coefficientRange=OVERALL&coefficientType=MEN_CLUB&language=EN&page={p}&pagesize=500&seasonYear={Y}` | `member.id`, `overallRanking.totalValue` — paged until a page returns < 500 members |

Facts established by probing:

- `seasonYear` = the calendar year the season ends (2026 = 2025-26).
- Round names vary by season: `Group stage` (2012–2024), `League Phase` (2025–2026),
  `Knockout Phase Play-Offs` (2025) / `Knock-out Play-off` (2026), `Round of 16`, `Quarter-finals`,
  `Semi-finals`, `Final`. 2020 has single-leg QF/SF (4 and 2 matches).
- The final's winner is `winner.match.team.id`. Penalty finals may carry `reason: "DRAW"` (2012, 2016)
  or `WIN_ON_PENALTIES` (2026) — **never infer the winner from the score**. Home/away in a final is nominal
  (2022 is stored as Liverpool 0-1 Real Madrid), so scores are re-oriented to winner/runner-up.
- Per-match stats present in every season (sampled): `goals`, `goals_conceded`, `attempts_on_target`,
  `attempts_off_target`, `ball_possession`, `passes_attempted`, `passes_completed`,
  `passes_long_attempted`, `passes_short_attempted`, `passes_medium_attempted`, `attacks`,
  `distance_covered`, `fouls_committed`, `fouls_suffered`, `saves`. There is **no total-attempts stat in
  every season**, so shots = on target + off target (blocked shots excluded consistently).
- Team ids are mostly shared across feeds, but the stats feed occasionally uses a different id for a club than the match feed (Steaua 2614166 in matches vs 50065 in stats). When exactly one side of a match is unmatched and exactly one stats entry is left over, that entry belongs to the unmatched side.
- Standings (`https://standings.uefa.com/v1/standings?competitionId=1&seasonYear={Y}`, `points`, `goalDifference`) are used only to verify the computed group/league-phase points and goal difference.
- Coefficient ranking `seasonYear=Y` covers seasons Y-4..Y. **Pre-season strength for season Y uses
  ranking Y-1.** Team ids are shared with the match API. Clubs with no UEFA matches in the window
  (debutants such as Girona, Brest, Bologna in 2024-25) are absent from the ranking.

## 5. Dataset construction

### 5.1 Populations

- **Field:** every team in the group/league phase — 32 per season (2012–2024), 36 (2025–2026); 488 team-seasons.
  Used as the reference for within-season normalisation and percentiles.
- **Knockout population (training set):** teams that played in the knockouts (R16 or KO play-off) —
  16 per season (2012–2024), 24 (2025–2026); 256 team-seasons, all of them members of their season's field.
  Training only on these avoids the tautology "teams that win group games get out of the group".

### 5.2 Label

`ko_stage` — comparable across both formats (every season has exactly 8 / 4 / 2 / 1 teams at levels ≥1 / ≥2 / ≥3 / ≥4):

| ko_stage | Meaning |
|-|-|
| 0 | Out before the quarter-finals (KO play-off or Round of 16) |
| 1 | Lost in quarter-finals |
| 2 | Lost in semi-finals |
| 3 | Runner-up |
| 4 | Winner |

Derived as the furthest round a team appears in; the final's winner from `winner.match.team.id`.
`reached_final = ko_stage >= 3`. An unknown round name raises an error. A display label
(`stage_label`, e.g. "Round of 16", "Knockout play-off", "Group stage") is kept for every field team.

### 5.3 Features (from group/league-phase matches only)

For each team-match, the team's own stats plus the **opponent's row from the same match** (for "against" stats).

| Group | Feature | Definition |
|-|-|-|
| Pedigree | `coef_log` | log of pre-season 5-year club coefficient (ranking Y-1) |
| Results | `points_pg` | 3/1/0 per match, from `score.total` |
| Results | `goal_diff_pg` | (goals − goals conceded) per match, from `score.total` |
| Style – attack | `shots_pg` | (on target + off target) per match |
| Style – attack | `shot_accuracy` | Σ on target / Σ shots |
| Style – attack | `conversion` | Σ goals / Σ shots |
| Style – attack | `attacks_pg` | UEFA `attacks` per match |
| Style – defence | `shots_against_pg` | opponent shots per match |
| Style – defence | `on_target_against_pg` | opponent shots on target per match |
| Style – defence | `save_pct` | Σ saves / Σ opponent shots on target |
| Style – control | `possession_pct` | mean match possession |
| Style – control | `pass_accuracy` | Σ passes completed / Σ passes attempted |
| Style – control | `passes_pg` | passes attempted per match |
| Style – control | `long_pass_share` | Σ long passes attempted / Σ passes attempted |
| Style – intensity | `distance_km_pg` | distance covered per match |
| Style – intensity | `fouls_pg` | fouls committed per match |

**Missing-stat rules:**

- `points_pg` and `goal_diff_pg` use the match list's scores over **all** group/league matches, so they are always complete.
- A per-match average divides by the number of matches in which that stat is present for the team (and,
  for "against" stats, for the opponent).
- A ratio sums its numerator and denominator over the **same** matches. `conversion` uses goals from the
  stats feed, not the score, and only matches where both shots and goals are present.

**Normalisation:** model inputs are within-season z-scores computed against the **field** of that season.
This uses feature values only, never labels. Display values are the raw values plus two percentiles:

- `pct_season`: within the season's field.
- `pct_all`: among all 488 field team-seasons, 2012–2026.

**Coverage rule:** a feature is kept only if its source stats are present in ≥ 90% of team-matches in every
season; otherwise it is dropped globally and the build logs it. Zero denominators → NaN → imputed with the
season median (count logged).

**Coefficient join:** by team id. A club absent from ranking Y-1 is expected (debutant). It gets the minimum
coefficient among that season's matched participants, and its count is logged. The build fails only if fewer
than 75% of a season's field match, which guards against id bugs.

### 5.4 Finals table

`finals` has one row per season with `season`, `winner_id`, `winner`, `runner_up_id`, `runner_up`,
`winner_goals`, `runner_up_goals`, `winner_pens`, `runner_up_pens` (null if no shootout) and `city`.
The scores are oriented to the winner.

### 5.5 Build validations (the build fails loudly if any is violated)

1. Field size: 32 (2012–2024) or 36 (2025–2026) teams per season.
2. Matches per field team: 6 (2012–2024) or 8 (2025–2026) group/league-phase matches in the match list.
3. Knockout population: 16 (2012–2024) or 24 (2025–2026) teams per season, all members of the field.
4. Exactly 8 / 4 / 2 / 1 knockout teams at ko_stage ≥ 1 / ≥ 2 / ≥ 3 / ≥ 4 per season; exactly one winner and
   one runner-up; exactly two teams in each final.
5. The 10 finalist slots in §3 match the data by team id and result.
6. **Completeness:** a team-season with stats for fewer than 4 of its group/league matches is marked
   `complete=False`.
   - It is **left out of model fitting**, but is still predicted, counted in validations and shown in outputs, flagged.
   - If any of the 10 finalists is incomplete, the build fails.
7. No NaN in the model feature matrix after imputation.

## 6. Modeling

All validation is **leave-one-season-out (LOSO)**. There are 15 folds; each trains on the complete knockout
teams of the other 14 seasons and predicts every knockout team of the held-out season. Every reported
prediction, SHAP value and metric is out-of-fold. Each finalist is scored by a model that never saw its season.

The training seasons include later ones, so this is "would the model have picked them", **not** a forecast made at the time.

| Model | Purpose | Spec |
|-|-|-|
| **A — Gradient boosting** | Expected `ko_stage`; SHAP drivers | `sklearn.ensemble.GradientBoostingRegressor(n_estimators=250, learning_rate=0.03, max_depth=2, min_samples_leaf=8, subsample=0.8, random_state=42)`; SHAP via `shap.TreeExplainer` (exact TreeSHAP) on the held-out fold |
| **B — Logistic regression** | P(reach final); direction cross-check | `sklearn.linear_model.LogisticRegression(C=0.3, max_iter=2000)` on z-scored features, target `reached_final`; each held-out season's probabilities are rescaled to sum to 2 (two finalists), none above 1, because training is dominated by 16-team seasons and raw probabilities run high in 24-team seasons |

Rationale for scikit-learn: LightGBM and XGBoost wheels need Homebrew `libomp`, which is not installed;
scikit-learn + `shap` 0.52 work on Python 3.13 with no system changes (verified).

**Metrics** (all out-of-fold):

- Model A: mean within-season Spearman ρ (predicted vs actual `ko_stage`); share of actual finalists ranked
  in the model's top 4 of their season.
- Model B: pooled ROC AUC and Brier score for `reached_final`; each actual finalist's rank by P(final)
  within its season's knockout population.
- References: random ranking (ρ = 0, AUC = 0.5) and the base rate (2/16 or 2/24).

**Ablation** (pedigree vs results vs style): rerun both models under LOSO with seven feature sets:

- the single groups `pedigree` = {coef_log}, `results` = {points_pg, goal_diff_pg} and `style` = the 13 style features
- `all`
- `all − pedigree`, `all − results` and `all − style`, which measure each group's unique contribution

Report ρ (model A) and AUC (model B) for each set.

**Drivers:**

- **Importance:** mean |SHAP| over all out-of-fold rows (model A, `all` features).
- **Direction:** the sign of the Spearman correlation between a feature's z-score and its SHAP value.
- **Labels:** only the **top 6** features by importance are labelled. A top-6 feature is **robust** if model
  B's coefficient has the same sign as its SHAP direction in ≥ 12 of the 15 LOSO fits; otherwise it is
  **model-dependent**. Features outside the top 6 get no label.

**Winners vs runners-up** (all 15 finals; the 5 target finals are also shown individually):

- For each feature, take the z-score difference (winner − runner-up).
- Count finals where the winner was higher, lower or tied.
- Run a two-sided sign test on the non-tied finals (binomial, n = higher + lower), Holm-adjusted across the 16 features.
- Always present this with the caveat that 15 pairs cannot establish much.

## 7. Local AI analyst

- **Runtime:** LM Studio OpenAI-compatible server at `http://localhost:1234/v1`, model key
  `qwen/qwen3.5-35b-a3b` (official build). Configurable via `--llm-model`; every step uses the configured key.
- **`ensure_ready(model_key)`:**
  1. `GET /v1/models`. If the server is down, run `lms server start` and poll for up to 30 s.
  2. If `lms ps` does not list `model_key`, run `lms load {model_key} --context-length 16384 -y` (up to 200 s). If it exits
     non-zero, return False at once; otherwise wait up to 30 s for `lms ps` to list the model.
  3. Return False on any failure; it never raises.
- **Request:**
  - `POST /v1/chat/completions` with `reasoning_effort: "none"`, temperature 0.3, max_tokens 8000, timeout 300 s.
  - **Verified 2026-10-01 against LM Studio + qwen3.5-35b-a3b:**
    - With the default settings, `/no_think`, `chat_template_kwargs.enable_thinking=false` or `reasoning_effort: "low"`, the model reasons for 2,400–4,000 tokens on a two-sentence task. It often hits the limit with empty content.
    - With `reasoning_effort: "none"`, it uses 0 reasoning tokens and answers in about 2 s.
    - The 8000-token cap is only a safety net for other `--llm-model` choices: about 104 s at the measured ~77 tok/s, well under the timeout.
  - The answer is `message.content` with any `<think>…</think>` block removed (`reasoning_content` is ignored).
- **Output validity:** a response is invalid if:
  - `finish_reason == "length"`
  - the text is empty after stripping
  - an unclosed `<think>` remains
  - any required heading is missing

  An invalid response is never badged.
- **Call budget and retries** (at most 3 calls per narrative):
  1. Make the initial call.
  2. If it is invalid, retry. The retry appends the failed answer, truncated to its first 1,500 characters
     so the context stays far below 16,384 tokens, plus a user message naming the failure. The request
     therefore differs from the first. If it is still invalid, the narrative is unavailable.
  3. Once a valid text exists, if the grounding check finds unsupported numbers, make one grounding retry.
     It appends the answer plus a user message listing those numbers. Keep whichever valid version has
     fewer unsupported numbers (the earlier one on a tie, or if the retry is invalid) and flag the rest.
     If the grounding retry fails with a timeout or HTTP error, also keep the earlier valid text with its flags.
  4. **Every** response, valid or not, is cached under its full request. Retries are distinct requests, so a
     re-run replays exactly the same sequence from cache.
     Failed calls (timeout or HTTP error) are not cached, so such a narrative is re-attempted on the next run.
- **Fact sheet per finalist** (JSON, numbers pre-rounded to their display form):
  - Keys are **plain-English labels with units and meaning**, for example
    `"Model probability of reaching the final (%)"`, not `p_final_pct`.
    - Verified: with code-style keys the model misread `p_final_pct: 31` as "a 31% final pass completion rate".
    - Features where lower is better (shots against, shots on target against, fouls) carry "(lower is better)".
  - club, season, actual result and final score, with penalties (the sentence is phrased from the team's side; the
    report's finals list stays winner-oriented)
  - P(final) % and rank within its season's knockout teams
  - the knockout population size and base-rate % for that season
  - expected `ko_stage`
  - top 4 positive and top 3 negative SHAP drivers (feature label, raw value, `pct_season`, `pct_all`, contribution)
  - all 16 raw features with both percentiles
- **Scouting report prompt:** 180–250 words of markdown under three headings: `How they got there`,
  `Would the model have picked them?` and `Weak spots`. Rules:
  - Use only the fact sheet, and quote numbers exactly as given.
  - Do not introduce players, managers, matches or events that are not in the facts.
- **Synthesis prompt** (one narrative, under the same call budget):
  - Input: the global drivers table, ablation table, metrics, finalist summary table and winners-vs-runners-up table (with Holm-adjusted p).
    Also the context counts: 15 seasons, 15 finals, 30 finalists, 488 field and 256 knockout team-seasons.
  - Output: 350–450 words under two headings, `Why the best teams win` and `Winners vs runners-up`.
  - Same rules as the scouting report, plus: never call a difference significant unless its Holm-adjusted p < 0.05.
- **Grounding check** (pure function):
  - Extract every numeric token from the text. Ignore years 2011–2027, season labels (`2025-26`, `2025–26`,
    `2025—26`, `2025/26`) and integers 0–10.
  - A remaining token `t` with `d` decimal places is supported by a fact value `f` if |`|t|` − `|f|`| ≤ 0.5·10⁻ᵈ + 1e-9, or the same holds with `f` scaled ×100 or ÷100 (percent ↔ fraction).
    The epsilon absorbs float error, e.g. 18.5 vs 18.45.
  - So 18.4 is supported by 18.43, 31% by 0.31, and 0.88 is **not** supported by 0.78.
  - Unsupported numbers trigger the grounding retry above; any that survive are flagged in the report.
  - **Scope of the check:** it confirms that each number appears in the facts, not that it is attached to
    the right feature. Badge wording reflects that: "All figures found in the data" / "N figures not found in
    the data" / "unavailable". The Method section explains the limit.
- **Cache:** `data/llm_cache/{sha256(model+messages+params)}.json`, so re-runs are instant and reproducible.
- **Failure and skip:**
  - If `ensure_ready` is False, `analysis.json` gets `status: "unavailable"`. A per-call failure marks only that narrative unavailable.
  - With `--no-ai`, `status: "skipped"`.
  - In both cases the report shows a notice in place of the missing narratives, and every other output is unaffected.

## 8. Report (`out/report.html`)

A single self-contained HTML file. Charts are HTML/CSS rows generated in Python, with no JS libraries, so text
stays legible at phone width.
It follows the claude.ai artifact page contract: CSS colour tokens with light and dark themes, and a
readable layout at phone width. Sections:

1. **The 10 finalists** — five finals with winner-oriented scores (penalties shown for 2026).
2. **Would the model have picked them?** — dot chart of each finalist's out-of-fold P(final) against its season's
   base rate, with rank within season and expected vs actual stage. A one-line note explains LOSO, including that it trains on later seasons.
3. **What drives deep runs** — mean |SHAP| bar chart coloured by feature group, with robust / model-dependent
   markers on the top 6; ablation chart (ρ and AUC for all seven feature sets).
4. **Team cards ×10** — AI scouting report with a grounding badge ("All figures found in the data" /
   "N figures not found in the data" / "unavailable"), a top-driver diverging bar chart, and key stats with
   `pct_season` and `pct_all`.
5. **Winners vs runners-up** — AI commentary plus a paired-difference chart across 15 finals, with
   higher / lower / tied counts and Holm-adjusted p.
6. **Method & caveats** — data source and span, LOSO metrics, and the caveats in §11. Also what the
   AI number check covers: every figure appears in the data. It does not cover whether the figure is
   attached to the right stat.

After the pipeline runs, the report is published as a private claude.ai artifact. This is done by the
assistant, not by the pipeline.

## 9. Architecture

Python 3.13, uv project, `src/` layout. Dependencies: pandas, numpy, scipy, scikit-learn, shap;
dev: pytest. HTTP via stdlib `urllib`, so no HTTP library dependency.

```
ucl-finalists/
  pyproject.toml            # [project.scripts] ucl = "ucl.cli:main"
  src/ucl/
    config.py    # seasons, expected finalists (ids), round→stage map, feature list & groups, paths, hyperparameters
    uefa.py      # UefaClient: cached, retrying HTTP for matches / match stats / coefficients
    dataset.py   # raw → matches, team_match_stats, team_seasons, finals; labels; features; percentiles; validations
    model.py     # LOSO for models A and B, metrics, SHAP, ablation, drivers, winners-vs-runners-up
    grounding.py # pure output-validity, heading normalisation and grounding checks
    llm.py       # LM Studio client: readiness, request-keyed response cache
    facts.py     # plain-English fact sheets for the LLM
    analyst.py   # prompts, 3-call retry policy, analysis.json persistence
    charts.py    # HTML/CSS chart builders
    assets.py    # page CSS and tooltip JS
    report.py    # report renderers (artifact-ready fragment + standalone document)
    cli.py       # fetch | build | model | analyze | report | all  [--no-ai] [--llm-model KEY]
  tests/         # unit tests + small JSON fixtures
  data/raw/      # API cache, including known-missing markers (git-ignored)
  data/llm_cache/  # (git-ignored)
  data/processed/  matches.csv, team_match_stats.csv, team_seasons.csv, finals.csv, missing_stats.csv
  out/           # predictions.csv, shap.csv, drivers.csv, ablation.csv, metrics.json, finals_compare.csv, analysis.json,
                 # report.html (standalone) and report_page.html (artifact-ready fragment)
```

**Module interfaces** (each depends only on modules above it):

- `uefa.UefaClient(cache_dir, max_workers=4)`:
  - `.matches(season_year) -> list[dict]`
  - `.team_match_stats(match_id) -> list[dict] | None` (None = permanently unavailable)
  - `.coefficients(season_year) -> list[dict]` (all pages)
- `dataset.build(client, seasons) -> Dataset`, with DataFrames `matches`, `team_match_stats`, `team_seasons`
  (raw and z features, `pct_season_*`, `pct_all_*`, labels, flags) and `finals`.
  `dataset.validate(ds) -> None` raises `ValidationError` listing every failed check.
- `model.run(team_seasons, finals, features) -> ModelResults`:
  - `predictions`: per knockout team-season — expected ko_stage, P(final), rank in season
  - `shap`: per knockout team-season — base value + one out-of-fold SHAP column per feature
  - `drivers`, `ablation`, `metrics`, `finals_compare`
- `facts.build_facts(team_seasons, finals, results) -> dict` builds the plain-English fact sheets.
- `analyst.run(team_seasons, finals, results, llm_model, enabled) -> Analysis` (narratives + validity/grounding
  status per item).
  Pure helpers live in `grounding.py`: `check_grounding(text, facts) -> list[str]`,
  `validate_output(text, finish_reason, required_headings) -> str | None` (the reason it is invalid, or None) and
  `normalize_headings(text, headings)`, which rewrites tolerated heading variants (bold, numbered) as `## Heading`.
- `report.render(team_seasons, finals, results, analysis, generated=None) -> str` (pure given `generated`; returns the
  artifact-ready fragment) and `report.standalone(fragment) -> str` (full document).

Each CLI stage reads the previous stage's files from disk, so any stage can be re-run alone. `build`,
`model` and `report` run fully offline once `fetch` has filled the cache.

## 10. Error handling

- **HTTP:** a browser User-Agent, 25 s timeout, 4 retries with exponential backoff (1/2/4/8 s + jitter) on
  network errors, 5xx and 429. Other 4xx responses fail fast for that resource. At most 4 concurrent requests.
- **Caching:**
  - Cache writes are atomic (temp file + rename).
  - Transient errors are never cached.
  - A 404 or 410 on a match-stats request is cached as a known-missing marker, so `build` stays offline and deterministic.
    Other 4xx responses (for example a 403 block) fail fast and are not cached; 408 and 429 are retried.
  - `fetch` is resumable. `team_match_stats_many` returns `{id: reason}` for failures; `fetch` logs one reason per
    season and stops early if every request in a season fails.
  - Responses are shape-checked before caching (a non-empty match list; a coefficient page with `data.members`), and a
    corrupt cache file raises an error naming the file.
- Missing match stats are recorded in `data/processed/missing_stats.csv` and handled per §5.3 / §5.5.6.
- LM Studio unavailable, or a single call failing → §7 failure behaviour.

## 11. Testing & verification

- `tests/test_labels.py`:
  - The round→stage mapping is correct for both formats, including the KO play-off.
  - A 2026-style penalties final and a 2012-style `reason: "DRAW"` final both produce the right winner, with the score oriented to the winner.
  - An unknown round name raises.
- `tests/test_features.py` — on a hand-built two-match fixture:
  - opponent-derived stats, ratio-of-sums, points and goal difference are exact
  - a match with a missing stat is excluded from both numerator and denominator
  - within-season z-scores have mean 0
- `tests/test_model.py`:
  - No LOSO fold trains on its held-out season or on `complete=False` rows.
  - There is one prediction and one SHAP row per knockout team-season.
  - Two runs give identical output (fixed seed).
  - Sign-test counts treat ties correctly.
- `tests/test_grounding.py`:
  - Accepts "31%" against 0.31 or 31, 18.4 against 18.43, and 18.5 against 18.45.
  - Rejects 0.88 against 0.78 and an invented "47%".
  - Ignores "2025-26", "2025–26", "2024" and "3".
- `tests/test_output_validity.py` — rejects `finish_reason="length"`, empty text, an unclosed `<think>` and a missing heading.
- `tests/test_retry.py` — with a fake LLM:
  - an invalid-then-valid sequence makes 2 calls
  - the retry request differs from the first (it carries feedback)
  - a re-run makes 0 calls (all cached)
  - no narrative ever exceeds 3 calls
  - keep-best: a grounding retry that returns invalid text, or more unsupported numbers, keeps the earlier text (3 calls)
  - a timeout on the grounding retry keeps the earlier text
- **Integration:** `uv run ucl all --no-ai` on cached data produces every output file without error.
- **Verification before done:**
  - All §5.5 validations pass.
  - Spot-check the 10 finalists' group/league-phase points and goal difference against UEFA's published standings.
  - Read the generated report end to end, in both themes and at phone width.
  - Confirm every AI narrative's badge.

**Caveats the report must state:**

- Correlation is not causation, and group/league-phase stats depend on the opponents drawn.
- Knockouts are high-variance; the 2026 final was decided on penalties.
- Real Madrid won 6 of the 15 finals, so one club's profile weighs heavily on what "winning" looks like.
- In 2024-25 and 2025-26, a top-8 league finish skips the play-off, which builds in a "results" advantage.
- There is no xG, and shots exclude blocked attempts.
- 15 seasons contain only 30 finalists, so probabilities are rough.
- LOSO trains on seasons after the one being scored.
- UEFA's APIs are undocumented, and their values are used as published.
- The AI text is constrained to the facts and number-checked, but its interpretations are not causal evidence.
