"""Pick the forecast's rating settings, then report how they do on seasons they weren't tuned on.

    uv run python scripts/tune_forecast.py

Ratings warm up on 2011-12 and 2012-13, the settings are tuned on 2013-14 to 2018-19 (90-minute win/draw/loss log
loss), and everything from 2019-20 on is the test. The goals models are refitted on the training seasons each time.
"""
import itertools

import numpy as np
import pandas as pd

from ucl import config, dataset, forecast
from ucl.uefa import UefaClient

TRAIN = forecast.TRAIN_SEASONS
TEST = forecast.TEST_SEASONS


def score(fixtures, priors, params, seasons_fit, seasons_eval):
    history = forecast.rate(fixtures, priors, params).history
    models = forecast.fit_models(history[history["season"].isin(seasons_fit)])
    return forecast.match_metrics(history[history["season"].isin(seasons_eval)], models), models


def main() -> None:
    fixtures = forecast.load_fixtures(UefaClient(), config.SEASONS)
    priors = forecast.priors_from(dataset.load().team_seasons)
    grid = itertools.product([15, 20, 25, 30, 40], [40, 50, 60, 70, 80], [0.6, 0.7, 0.8, 0.85, 0.9, 0.95],
                             [25, 35, 50, 75, 100])
    results = []
    for k, home, carry, scale in grid:
        params = forecast.Params(k, home, carry, scale)
        metrics, _ = score(fixtures, priors, params, TRAIN, TRAIN)
        results.append((metrics["log_loss"], params))
    results.sort(key=lambda r: r[0])
    print("best on training seasons:")
    for loss, params in results[:5]:
        print(f"  {loss:.4f}  {params}")
    best = results[0][1]
    test, models = score(fixtures, priors, best, TRAIN, TEST)
    print(f"\ntest seasons with {best}:\n  {test}\n  {models}")

    pedigree = min((score(fixtures, priors, forecast.Params(0, h, 1.0, sc), TRAIN, TRAIN)[0]["log_loss"], h, sc)
                   for h in [40, 60, 70, 80, 100] for sc in [50, 75, 100, 150, 200])
    only, _ = score(fixtures, priors, forecast.Params(0, pedigree[1], 1.0, pedigree[2]), TRAIN, TEST)
    print(f"  coefficient only (home {pedigree[1]}, scale {pedigree[2]}): {only}")

    history = forecast.rate(fixtures, priors, best).history
    train, held = history[history["season"].isin(TRAIN)], history[history["season"].isin(TEST)]
    outcome = lambda h: np.where(h.home_goals > h.away_goals, 0, np.where(h.home_goals == h.away_goals, 1, 2))
    rates = np.bincount(outcome(train), minlength=3) / len(train)
    print(f"  base rates {rates.round(3)}: log loss {-np.mean(np.log(rates[outcome(held)])):.4f}")


if __name__ == "__main__":
    main()
