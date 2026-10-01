# UCL Finalists — task list

Plan: ~/docs/superpowers/plans/2026-10-01-ucl-finalists.md
Spec: ~/docs/superpowers/specs/2026-10-01-ucl-finalists-design.md

- [x] Chunk 1: scaffold, config, UEFA client
- [x] Chunk 2: labels, features, dataset
- [x] Chunk 3: fetch + build on real data, models, drivers, ablation, finals comparison
- [x] Chunk 4: grounding checks, LM Studio client
- [x] Chunk 5: fact sheets, analyst, analyze on the real data
- [ ] Chunk 6: charts and the report
- [ ] Chunk 7: end-to-end run, verification, publish

## Data notes (real build, 2026-10-01)

- Built 488 team-seasons (256 knockout) with **15 features**, not 16: the coverage rule dropped `save_pct`.
  UEFA's 2011-12 (season 2012) stats feed lacks `saves` for 26 of 192 team-matches (86.5% < 90%); every other season is >= 97.9%.
  The feed omits some counters instead of writing 0, so the gaps are mixed: of the 26, 13 are provably real (opponent shots on target minus goals conceded is >= 1),
  6 look like true zeros, and 7 can't be checked (the opponent's shot count is absent too). Zero-filling would be wrong, so the feature is dropped, as the coverage rule intends.
  `dataset.json` records the active 15 under `features`; downstream stages must use that list, not `config.FEATURES`.
- Stats missing: Wolfsburg 1-0 CSKA Moskva (2015-16, match 2015667) returns an empty list (HTTP 200), so 2 team-matches have no stats.
  Sevilla 3-0 Mönchengladbach (2015-16, match 2015672) has only `distance_covered` and `top_speed`; it counts as "with stats" but adds NaN to every other feature (averages skip NaN).
- Coefficients imputed: 12 = 11 clubs absent from the previous ranking (Málaga 2013, Real Sociedad 2014, Monaco 2015, Leicester 2017, Leipzig 2018, Lens and Newcastle 2024, Brest, Girona, Stuttgart and Bologna 2025)
  plus Steaua 2014, whose match-feed id 2614166 is FCSB (50065) in the ranking. None of the 10 finalists is imputed.
- All 10 finalists' league/group points and goal difference match UEFA standings (`scripts/check_standings.py`).

## Model notes (real run, 2026-10-01)

- 15 leave-one-season-out folds, 256 knockout team-seasons (30 finalists), 15 features, `config` hyperparameters as planned (not tuned). `uv run ucl model` takes about 10 s and its output is byte-identical on a re-run.
- Spearman 0.46 (per season: 0.08 in 2020 to 0.72 in 2024), AUC 0.72, Brier 0.098 against 0.103 for the base rate (a slim edge), and 47% of finalists land in the model's top 4 by expected stage. These are modest numbers; the report should present them as such.
- Ablation (Spearman / AUC): pedigree 0.28 / 0.66, results 0.30 / 0.72, style 0.41 / 0.69, all 0.46 / 0.72, all minus pedigree 0.47 / 0.72, all minus results 0.40 / 0.69, all minus style 0.37 / 0.73.
  The two results features alone match the full set on AUC; the style features add rank ordering (Spearman), not finalist discrimination.
- Top-6 drivers: attacks_pg, goal_diff_pg, long_pass_share, passes_pg, fouls_pg, conversion. Five are robust (same sign in 15 of 15 logistic folds); fouls_pg is model-dependent (5 of 15).

## Review
(filled in at the end)
