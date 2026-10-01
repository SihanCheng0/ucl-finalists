import numpy as np
import pandas as pd
import pytest
from synthetic import make_synthetic_dataset

from ucl import config, dataset


def test_valid_dataset_passes(synthetic_ds):
    dataset.validate(synthetic_ds)


def test_missing_team_is_reported(synthetic_ds):
    ts = synthetic_ds.team_seasons
    synthetic_ds.team_seasons = ts.drop(ts.index[ts["season"] == 2013][-1])
    with pytest.raises(dataset.ValidationError, match="2013: field has 31 teams"):
        dataset.validate(synthetic_ds)


def test_wrong_final_is_reported(synthetic_ds):
    synthetic_ds.finals.loc[synthetic_ds.finals["season"] == 2026, "winner_id"] = "999"
    with pytest.raises(dataset.ValidationError, match="2026: final"):
        dataset.validate(synthetic_ds)


def test_incomplete_target_finalist_is_reported(synthetic_ds):
    ts = synthetic_ds.team_seasons
    ts.loc[ts["is_target"] & (ts["season"] == 2025), "complete"] = False
    with pytest.raises(dataset.ValidationError, match="target finalists without enough stats"):
        dataset.validate(synthetic_ds)


def test_bad_knockout_levels_are_reported(synthetic_ds):
    ts = synthetic_ds.team_seasons
    ts.loc[ts.index[(ts["season"] == 2020) & (ts["ko_stage"] == 1)][0], "ko_stage"] = 0.0
    with pytest.raises(dataset.ValidationError, match="2020: teams at ko_stage"):
        dataset.validate(synthetic_ds)


def test_low_coefficient_match_rate_is_reported(synthetic_ds):
    synthetic_ds.notes["coef_match_rate"][2019] = 0.5
    with pytest.raises(dataset.ValidationError, match="2019: only 50% of clubs matched"):
        dataset.validate(synthetic_ds)


def test_plausibility_errors_name_the_out_of_range_team_seasons():
    ts = pd.DataFrame({
        "season": [2016, 2016, 2016], "team": ["Barcelona", "Roma", "Porto"],
        "possession_pct": [878.8, 55.0, np.nan],   # seconds read as a percentage
        "distance_km_pg": [112.0, 3.2, 118.0],     # km read as thousands of metres
        "pass_accuracy": [0.85, 0.82, 0.80],
    })
    errors = "\n".join(dataset.plausibility_errors(ts))
    assert "2016 Barcelona" in errors and "possession_pct" in errors and "878.8" in errors
    assert "2016 Roma" in errors and "distance_km_pg" in errors
    assert "Porto" not in errors   # a missing value is not an implausible one


def test_plausibility_errors_accept_the_range_limits_and_absent_columns():
    lows = {f: low for f, (low, _) in config.PLAUSIBLE_RANGES.items()}
    highs = {f: high for f, (_, high) in config.PLAUSIBLE_RANGES.items()}
    ts = pd.DataFrame([{"season": 2016, "team": "Low", **lows}, {"season": 2016, "team": "High", **highs}])
    assert dataset.plausibility_errors(ts) == []
    assert dataset.plausibility_errors(ts[["season", "team", "possession_pct"]]) == []


class FakeClient:
    """A one-season stand-in for UefaClient as build() uses it: two teams, a group match and the final."""

    def __init__(self, possession):
        def team(team_id):
            return {"id": team_id, "internationalName": f"Team {team_id}"}

        def match(match_id, round_name):
            return {"id": match_id, "round": {"metaData": {"name": round_name}}, "homeTeam": team("1"),
                    "awayTeam": team("2"), "score": {"total": {"home": 1, "away": 0}},
                    "winner": {"match": {"team": {"id": "1"}}}}

        self.raw = [match("m1", "Group stage"), match("f1", "Final")]
        self.stats = {"m1": [{"teamId": t, "statistics": [{"name": "goals", "value": "1"},
                                                          {"name": "ball_possession", "value": str(possession)}]}
                             for t in "12"]}

    def matches(self, season):
        return self.raw

    def team_match_stats_many(self, match_ids):
        return {m: self.stats.get(m) for m in match_ids}, {}

    def coefficients(self, season):
        return [{"member": {"id": t}, "overallRanking": {"totalValue": 10.0}} for t in "12"]


def test_build_fails_when_a_team_season_average_is_implausible():
    # 99% possession for both sides is a feed error that no per-match rule can see
    with pytest.raises(dataset.ValidationError) as exc:
        dataset.build(FakeClient(possession=99), seasons=[2012])
    assert "2012 Team 1" in str(exc.value) and "2012 Team 2" in str(exc.value)


def test_build_accepts_plausible_averages():
    ds = dataset.build(FakeClient(possession=55), seasons=[2012])
    assert ds.team_seasons["possession_pct"].tolist() == [55.0, 55.0]


def test_save_and_load_roundtrip_keeps_id_strings(tmp_path):
    ds = make_synthetic_dataset(seasons=(2026,))
    ds.team_match_stats = pd.DataFrame(
        {"season": [2026], "match_id": ["0042"], "team_id": ["007"], "has_stats": [False]}
    )
    dataset.save(ds, tmp_path)
    loaded = dataset.load(tmp_path)
    assert loaded.team_match_stats.loc[0, "team_id"] == "007"
    assert loaded.team_match_stats.loc[0, "match_id"] == "0042"
    assert loaded.features == ds.features
    assert loaded.team_seasons["in_ko"].dtype == bool
    assert (tmp_path / "missing_stats.csv").exists()
