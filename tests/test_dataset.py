import pandas as pd
import pytest
from synthetic import make_synthetic_dataset

from ucl import dataset


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
