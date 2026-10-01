import numpy as np
import pandas as pd
import pytest
from synthetic import make_synthetic_dataset

from ucl import config, model


@pytest.fixture(scope="module")
def results():
    ds = make_synthetic_dataset(seasons=tuple(range(2019, 2027)))
    return ds, model.run(ds.team_seasons, ds.finals, ds.features)


def test_robust_needs_sign_agreement_in_twelve_of_fifteen_folds():
    ko = pd.DataFrame({"season": [1, 1, 2, 2], "team_id": ["a", "b", "c", "d"],
                       "z_shots_pg": [-1.0, 1.0, -0.5, 0.5], "z_possession_pct": [-1.0, 1.0, -0.5, 0.5]})
    shap_df = ko[["season", "team_id"]].assign(
        base_value=0.0, shap_shots_pg=[-0.2, 0.2, -0.1, 0.1], shap_possession_pct=[-0.1, 0.1, -0.05, 0.05])
    coefs = [np.array([1.0 if i < 12 else -1.0, 1.0 if i < 11 else -1.0]) for i in range(15)]
    d = model.drivers(ko, shap_df, coefs, ["shots_pg", "possession_pct"]).set_index("feature")
    assert (d.loc["shots_pg", "sign_agree_folds"], d.loc["shots_pg", "label"]) == (12, "robust")
    assert (d.loc["possession_pct", "sign_agree_folds"], d.loc["possession_pct", "label"]) == (11, "model-dependent")


def test_sign_counts_treat_ties():
    higher, lower, tied, p = model.sign_counts(np.array([0.5, -0.2, 0.0, 0.0, 1.0]))
    assert (higher, lower, tied) == (2, 1, 2)
    assert p == pytest.approx(1.0)  # binomial test on the 3 non-tied finals


def test_holm_adjustment():
    assert model.holm([0.01, 0.04, 0.03]).tolist() == pytest.approx([0.03, 0.06, 0.06])


def test_driver_labels_only_on_the_top_six(results):
    _, res = results
    d = res.drivers
    assert list(d["rank"]) == list(range(1, len(d) + 1))
    assert (d.loc[d["rank"] > config.TOP_DRIVERS, "label"] == "").all()
    assert set(d.loc[d["rank"] <= config.TOP_DRIVERS, "label"]) <= {"robust", "model-dependent"}
    assert d["importance"].is_monotonic_decreasing


def test_ablation_has_the_seven_feature_sets(results):
    _, res = results
    assert list(res.ablation["feature_set"]) == [
        "pedigree", "results", "style", "all", "all minus pedigree", "all minus results", "all minus style",
    ]


def test_finals_compare_counts_every_final(results):
    ds, res = results
    fc = res.finals_compare
    assert len(fc) == len(ds.features)
    assert ((fc["higher"] + fc["lower"] + fc["tied"]) == len(ds.finals)).all()
    assert {f"diff_{s}" for s in config.TARGET_SEASONS} <= set(fc.columns)
    assert (fc["p_holm"] >= fc["p"] - 1e-12).all()


def test_predictions_carry_display_columns(results):
    _, res = results
    assert {"team_display", "is_target", "complete"} <= set(res.predictions.columns)
    assert int(res.predictions["is_target"].sum()) == 10


def test_feature_list_follows_config_order(results):
    ds, res = results
    assert model.feature_list(res) == ds.features


def test_save_and_load_roundtrip(results, tmp_path):
    _, res = results
    model.save(res, tmp_path)
    loaded = model.load(tmp_path)
    assert loaded.metrics["n_seasons"] == res.metrics["n_seasons"]
    assert len(loaded.shap) == len(res.shap)
    assert loaded.predictions["team_id"].dtype != np.int64
