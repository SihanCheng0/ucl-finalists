# UCL Finalists — task list

Plan: ~/docs/superpowers/plans/2026-10-01-ucl-finalists.md
Spec: ~/docs/superpowers/specs/2026-10-01-ucl-finalists-design.md

- [x] Chunk 1: scaffold, config, UEFA client
- [x] Chunk 2: labels, features, dataset
- [ ] Chunk 3: fetch + build on real data, models, drivers, ablation, finals comparison
- [ ] Chunk 4: grounding checks, LM Studio client
- [ ] Chunk 5: fact sheets, analyst, analyze on the real data
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

## Review
(filled in at the end)
