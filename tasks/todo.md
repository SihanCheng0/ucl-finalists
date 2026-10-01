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
(filled in at the end)
