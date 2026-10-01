# UCL Finalists

What do Champions League finalists have in common? This project collects 15 seasons of UEFA data
(2011-12 to 2025-26) and trains leave-one-season-out models of how far knockout teams go. A local LLM
(Qwen 3.5 in LM Studio) then writes a scouting report on each of the 10 finalists of 2022–2026.

## Run it

You need Python 3.13 with uv. The AI write-ups also need LM Studio, with its server on localhost:1234 and
the `lms` CLI. If `qwen/qwen3.5-35b-a3b` isn't loaded, the analyze stage loads it at 16384 context.

```bash
uv sync
uv run ucl all            # fetch → build → model → analyze → report
uv run ucl all --no-ai    # everything except the LLM write-ups
open out/report.html
```

Each stage also runs alone: `uv run ucl fetch | build | model | analyze | report`. Use
`--llm-model KEY` to pick another LM Studio model. `uv run pytest -q` runs the tests, and
`uv run python scripts/check_standings.py` checks the finalists' points and goal difference against UEFA's
standings.

## Stages and outputs

| Stage | Writes |
|-|-|
| fetch | `data/raw/`: UEFA matches, per-match team stats and club coefficients (cached, resumable) |
| build | `data/processed/`: 488 group/league-phase team-seasons with features and labels. Fails on implausible values |
| model | `out/predictions.csv`, `shap.csv`, `drivers.csv`, `ablation.csv`, `finals_compare.csv`, `metrics.json` |
| analyze | `out/analysis.json`: the scouting reports. LLM responses are cached in `data/llm_cache/`, so reruns replay them |
| report | `out/report.html` (open locally) and `out/report_page.html` (artifact-ready fragment) |

## Method

- **Target.** Training uses the 256 team-seasons that reached the knockouts. The target is how far each went:
  - 0: out before the quarter-finals
  - 1: quarter-finals
  - 2: semi-finals
  - 3: lost the final
  - 4: won it
- **Inputs.** 15 per-game stats from group and league-phase matches only, z-scored within each season. They
  fall into three groups:
  - pedigree: the pre-season UEFA club coefficient
  - results
  - style

  Save rate was dropped because the 2011-12 feed lacks saves for too many matches. The coverage rule keeps
  a stat only if at least 90% of matches report it.
- **Cleaning.** For some 2014-16 matches, UEFA's feed gives possession in seconds and distance in metres.
  These are converted. Partial tracking (a match distance under 80 km) and placeholder zeros are masked.
  Steaua's match-feed id is aliased to FCSB. A plausibility check fails the build on out-of-range season
  averages.
- **Models.** Two models, each validated leave-one-season-out over 15 folds:
  - Gradient boosting predicts the stage. Exact TreeSHAP explains each prediction.
  - An L2 logistic regression predicts reaching the final. Its probabilities are scaled within each season
    to sum to two.
- **Uncertainty.** 95% intervals come from resampling seasons (1000 bootstrap draws).
- **Driver labels.** The top 6 stats by mean |SHAP| each get one label:
  - *Robust:* the logistic model agrees on the direction in at least 12 of 15 seasons, and the stat points
    the same way on its own.
  - *Conditional:* the direction holds only with the other stats held fixed.
  - *Model-dependent:* the logistic model disagrees too often.
- **Winners vs runners-up.** Sign tests over the 15 finals, with a Holm adjustment across the 15 stats.
- **Local AI.** The LLM sees only a plain-English fact sheet. Each answer is checked for:
  - its structure
  - numbers that aren't in the facts
  - wording slips: significance words in team reports, length, the wrong name for the first phase, or the
    final placed inside it

  One retry fixes what it can, and the better answer is kept. The report shows a badge for the number check.

## Results

| Measure | Value | 95% interval |
|-|-|-|
| Ranking teams within a season (Spearman) | 0.45 | 0.34 to 0.54 |
| Spotting the finalists (AUC) | 0.73 | 0.61 to 0.83 |
| Brier skill against the base rate | 0.05 | -0.04 to 0.13, so inconclusive |
| Finalists in the predicted top four | 47% | 30% to 60%, against 24% for a random ranking |

Group-stage play only modestly predicts deep runs. Attacks, goal difference and passes per game are the
robust drivers. Pedigree adds little once play is known, though the intervals overlap. 3 of the 10 recent
finalists were the model's top-two pick in their season.

- Design spec: `~/docs/superpowers/specs/2026-10-01-ucl-finalists-design.md`
- Implementation plan: `~/docs/superpowers/plans/2026-10-01-ucl-finalists.md`

## Caveats

These are associations, not causes. Knockouts are noisy, and 15 seasons hold only 30 finalists. UEFA publishes
no xG for these seasons, and its APIs are undocumented. The report lists every caveat.
