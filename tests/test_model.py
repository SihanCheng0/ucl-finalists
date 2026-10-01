import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from synthetic import make_synthetic_dataset

from ucl import model


@pytest.fixture(scope="module")
def small():
    ds = make_synthetic_dataset(seasons=tuple(range(2021, 2027)))
    return ds, model.knockout_population(ds.team_seasons)


def test_one_prediction_and_shap_row_per_knockout_team(small):
    ds, ko = small
    out = model.loso(ko, ds.features)
    assert len(out.predictions) == len(ko) == len(out.shap)
    assert not out.predictions.duplicated(["season", "team_id"]).any()
    assert set(out.predictions["rank_in_season"].groupby(out.predictions["season"]).min()) == {1}
    # two teams reach each final, so each season's probabilities add up to 2
    np.testing.assert_allclose(out.predictions.groupby("season")["p_final"].sum(), 2.0)
    assert out.predictions["p_final"].max() <= 1.0


def test_scale_to_two_sums_to_two_and_caps_at_one():
    assert model.scale_to_two([0.1, 0.3, 0.2, 0.4]) == pytest.approx([0.2, 0.6, 0.4, 0.8])
    assert model.scale_to_two([0.9, 0.05, 0.05]) == pytest.approx([1.0, 0.5, 0.5])


def test_evaluate_matches_hand_computed_values():
    pred = pd.DataFrame({
        "season": [1] * 5 + [2] * 5,
        "ko_stage": [4, 3, 2, 1, 0] * 2,
        "exp_stage": [5, 1, 4, 3, 2, 1, 5, 4, 3, 2],
        "reached_final": [True, True, False, False, False] * 2,
        "p_final": [0.6, 0.1, 0.3, 0.2, 0.05, 0.4, 0.5, 0.45, 0.1, 0.05],
        "base_rate": [0.4] * 10,
    })
    m = model.evaluate(pred)
    # rho: 1 - 6*12/120 = 0.4 and 1 - 6*20/120 = 0.0; one finalist per season ranks 5th by expected stage;
    # AUC = 19.5 / 24; Brier = 1.9275 / 10; base-rate Brier = (4*0.36 + 6*0.16) / 10
    assert m["spearman_by_season"] == {1: pytest.approx(0.4), 2: pytest.approx(0.0, abs=1e-12)}
    assert m["spearman_mean"] == pytest.approx(0.2)
    assert m["finalists_in_top4"] == pytest.approx(0.5)
    assert m["auc"] == pytest.approx(0.8125)
    assert m["brier"] == pytest.approx(0.19275)
    assert m["brier_base_rate"] == pytest.approx(0.24)
    assert (m["n_knockout"], m["n_finalists"], m["n_seasons"]) == (10, 4, 2)


def test_loso_never_trains_on_held_out_season_or_incomplete_rows(small, monkeypatch):
    ds, ko = small
    ko = ko.copy()
    ko.loc[0, "complete"] = False
    seen = []
    real_fit = GradientBoostingRegressor.fit

    def spy(self, X, y, *args, **kwargs):
        seen.append(X.index)
        return real_fit(self, X, y, *args, **kwargs)

    monkeypatch.setattr(GradientBoostingRegressor, "fit", spy)
    out = model.loso(ko, ds.features, with_shap=False)
    assert len(seen) == ko["season"].nunique()  # one fit per fold, so the loop below is not vacuous
    for season, index in zip(sorted(ko["season"].unique()), seen):
        train = ko.loc[index]
        assert season not in set(train["season"])
        assert train["complete"].all()
    assert len(out.predictions) == len(ko)  # the incomplete row is still predicted


def test_runs_are_deterministic(small):
    ds, ko = small
    a, b = model.loso(ko, ds.features), model.loso(ko, ds.features)
    pd.testing.assert_frame_equal(a.predictions, b.predictions)
    pd.testing.assert_frame_equal(a.shap, b.shap)


def test_shap_values_add_up_to_the_prediction(small):
    ds, ko = small
    out = model.loso(ko, ds.features)
    total = out.shap["base_value"] + out.shap[[f"shap_{f}" for f in ds.features]].sum(axis=1)
    np.testing.assert_allclose(total.to_numpy(), out.predictions["exp_stage"].to_numpy(), atol=1e-6)


def test_evaluate_reports_the_spec_metrics(small):
    ds, ko = small
    metrics = model.evaluate(model.loso(ko, ds.features, with_shap=False).predictions)
    assert set(metrics) >= {"spearman_mean", "spearman_by_season", "finalists_in_top4", "auc", "brier",
                            "brier_base_rate", "n_knockout", "n_finalists", "n_seasons"}
    assert metrics["n_finalists"] == 12 and metrics["n_seasons"] == 6
    assert 0.0 <= metrics["auc"] <= 1.0
