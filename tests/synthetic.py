"""A synthetic dataset with real field sizes, knockout counts and the expected finals."""
import numpy as np
import pandas as pd

from ucl import config, features
from ucl.dataset import Dataset


def make_synthetic_dataset(seasons=tuple(config.SEASONS), seed: int = 7) -> Dataset:
    rng = np.random.default_rng(seed)
    team_rows, final_matches, finals = [], [], []
    for season in seasons:
        ids = [f"{season}{i:02d}" for i in range(config.field_size(season))]
        expected = config.EXPECTED_FINALS.get(season)
        if expected:
            ids[0], ids[1] = expected["winner_id"], expected["runner_up_id"]
        n_ko = config.ko_size(season)
        stages = [4, 3, 2, 2, 1, 1, 1, 1] + [0] * (n_ko - 8) + [None] * (len(ids) - n_ko)
        for team_id, stage in zip(ids, stages):
            strength = rng.normal() + (0.6 * stage if stage is not None else -0.8)
            row = {
                "season": season, "team_id": team_id, "team": f"Team {team_id}",
                "in_ko": stage is not None, "ko_stage": np.nan if stage is None else float(stage),
                "stage_label": "synthetic", "n_matches": config.phase_matches(season),
                "n_with_stats": config.phase_matches(season), "coef_imputed": False,
            }
            for f in config.FEATURES:
                row[f] = strength * rng.uniform(0.3, 1.0) + rng.normal(scale=0.7)
            row["coef"] = round(abs(row["coef_log"]) * 30, 1)
            team_rows.append(row)
        final_matches.append({"season": season, "match_id": f"F{season}", "round": "Final", "depth": 5,
                              "home_id": ids[1], "away_id": ids[0], "winner_id": ids[0]})
        finals.append({"season": season, "winner_id": ids[0], "winner": f"Team {ids[0]}",
                       "runner_up_id": ids[1], "runner_up": f"Team {ids[1]}", "winner_goals": 2,
                       "runner_up_goals": 1, "winner_pens": np.nan, "runner_up_pens": np.nan,
                       "city": "Testville"})
    ts = pd.DataFrame(team_rows)
    ts = pd.concat([ts, features.zscore_within_season(ts, config.FEATURES)], axis=1)
    ts = features.add_percentiles(ts, config.FEATURES)
    ts["complete"] = True
    ts["reached_final"] = ts["ko_stage"].ge(3)
    ts["is_target"] = ts["season"].isin(config.TARGET_SEASONS) & ts["reached_final"]
    ts["team_display"] = ts["team"]
    return Dataset(
        matches=pd.DataFrame(final_matches),
        team_match_stats=pd.DataFrame({"season": [], "match_id": [], "team_id": [], "has_stats": []}),
        team_seasons=ts,
        finals=pd.DataFrame(finals),
        features=list(config.FEATURES),
        notes={"coef_match_rate": {s: 1.0 for s in seasons}},
    )
