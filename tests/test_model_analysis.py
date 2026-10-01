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
    ko = pd.DataFrame({"season": [1, 1, 2, 2], "team_id": ["a", "b", "c", "d"], "ko_stage": [0, 3, 1, 2],
                       "z_shots_pg": [-1.0, 1.0, -0.5, 0.5], "z_possession_pct": [-1.0, 1.0, -0.5, 0.5]})
    shap_df = ko[["season", "team_id"]].assign(
        base_value=0.0, shap_shots_pg=[-0.2, 0.2, -0.1, 0.1], shap_possession_pct=[-0.1, 0.1, -0.05, 0.05])
    coefs = [np.array([1.0 if i < 12 else -1.0, 1.0 if i < 11 else -1.0]) for i in range(15)]
    d = model.drivers(ko, shap_df, coefs, ["shots_pg", "possession_pct"]).set_index("feature")
    # both stats also rise with ko_stage on their own (rho = 1), so only the fold agreement tells them apart
    assert d.loc["shots_pg", "marginal_rho"] == pytest.approx(1.0)
    assert (d.loc["shots_pg", "sign_agree_folds"], d.loc["shots_pg", "label"]) == (12, "robust")
    assert (d.loc["possession_pct", "sign_agree_folds"], d.loc["possession_pct", "label"]) == (11, "model-dependent")


def test_a_stat_that_points_the_other_way_on_its_own_is_conditional():
    ko = pd.DataFrame({
        "season": [1, 1, 2, 2], "team_id": ["a", "b", "c", "d"], "ko_stage": [3, 0, 2, 1],
        "z_long_pass_share": [-1.0, 1.0, -0.5, 0.5],  # rho -1 with ko_stage, though the model reads it as helping
        "z_conversion": [-0.2, 0.3, 0.9, -1.0],       # ranks 2, 3, 4, 1: rho exactly 0 with ko_stage
        "z_fouls_pg": [-1.0, 1.0, -0.5, 0.5],         # as long_pass_share, but the folds disagree on it too
    })
    shap_df = ko[["season", "team_id"]].assign(
        base_value=0.0, shap_long_pass_share=[-0.2, 0.2, -0.1, 0.1], shap_conversion=[-0.1, 0.1, 0.3, -0.3],
        shap_fouls_pg=[-0.1, 0.1, -0.05, 0.05])
    coefs = [np.array([1.0, 1.0, 1.0 if i < 11 else -1.0]) for i in range(15)]
    d = model.drivers(ko, shap_df, coefs, ["long_pass_share", "conversion", "fouls_pg"]).set_index("feature")
    assert list(d["direction"]) == [1, 1, 1]  # every z-score rises with its SHAP values
    assert d["marginal_rho"].to_dict() == {
        "long_pass_share": pytest.approx(-1.0), "conversion": pytest.approx(0.0, abs=1e-12),
        "fouls_pg": pytest.approx(-1.0)}
    assert d["sign_agree_folds"].to_dict() == {"long_pass_share": 15, "conversion": 15, "fouls_pg": 11}
    # the fold agreement passes for the first two but their direction does not hold on its own, so not robust;
    # a failed fold agreement is model-dependent whatever the marginal check says
    assert d["label"].to_dict() == {
        "long_pass_share": "conditional", "conversion": "conditional", "fouls_pg": "model-dependent"}


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
    assert set(d.loc[d["rank"] <= config.TOP_DRIVERS, "label"]) <= {"robust", "conditional", "model-dependent"}
    assert d["importance"].is_monotonic_decreasing


def test_ablation_has_the_seven_feature_sets(results):
    _, res = results
    assert list(res.ablation["feature_set"]) == [
        "pedigree", "results", "style", "all", "all minus pedigree", "all minus results", "all minus style",
    ]


def test_ablation_rows_carry_bootstrap_intervals(results):
    _, res = results
    assert {"spearman_lo", "spearman_hi", "auc_lo", "auc_hi"} <= set(res.ablation.columns)
    # the point estimate is not guaranteed to sit inside its own percentile interval, so only the order is checked
    assert (res.ablation["spearman_lo"] <= res.ablation["spearman_hi"]).all()
    assert (res.ablation["auc_lo"] <= res.ablation["auc_hi"]).all()


def test_run_reports_intervals_for_the_headline_metrics(results):
    _, res = results
    ci = res.metrics["ci"]
    assert set(ci) == {"spearman_mean", "auc", "brier_skill", "finalists_in_top4"}
    assert all(len(bounds) == 2 and bounds[0] <= bounds[1] for bounds in ci.values())


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
    assert loaded.metrics == res.metrics  # JSON turns the int season keys into strings; load turns them back
    assert all(isinstance(season, int) for season in loaded.metrics["spearman_by_season"])
    assert len(loaded.shap) == len(res.shap)
    assert loaded.predictions["team_id"].dtype != np.int64
