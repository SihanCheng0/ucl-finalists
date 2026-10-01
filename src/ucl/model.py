"""Leave-one-season-out models, SHAP drivers, ablation and winners-vs-runners-up (spec §6)."""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.stats import binomtest, spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from . import config


@dataclass
class LosoOutput:
    predictions: pd.DataFrame
    shap: pd.DataFrame | None
    fold_coefs: list[np.ndarray]


@dataclass
class ModelResults:
    predictions: pd.DataFrame
    shap: pd.DataFrame
    drivers: pd.DataFrame
    ablation: pd.DataFrame
    metrics: dict
    finals_compare: pd.DataFrame


def knockout_population(team_seasons: pd.DataFrame) -> pd.DataFrame:
    ko = team_seasons.loc[team_seasons["in_ko"].astype(bool)].copy()
    ko["ko_stage"] = ko["ko_stage"].astype(int)
    ko["reached_final"] = ko["reached_final"].astype(bool)
    return ko.sort_values(["season", "team_id"]).reset_index(drop=True)


def scale_to_two(p) -> np.ndarray:
    """Rescale one season's P(final) to sum to 2 (two finalists) with none above 1.

    Model B trains mostly on 16-team seasons, so its raw probabilities run high in 24-team seasons.
    """
    p = np.asarray(p, dtype=float)
    capped = np.zeros(len(p), dtype=bool)
    while True:
        scaled = np.where(capped, 1.0, p * (2.0 - capped.sum()) / p[~capped].sum())
        newly = (scaled > 1.0) & ~capped
        if not newly.any():
            return scaled
        capped |= newly


def loso(ko: pd.DataFrame, features: list[str], with_shap: bool = True) -> LosoOutput:
    """Each season predicted by models trained on the complete rows of the other seasons."""
    cols = [f"z_{f}" for f in features]
    predictions, shap_frames, coefs = [], [], []
    for season in sorted(ko["season"].unique()):
        train = ko.loc[(ko["season"] != season) & ko["complete"].astype(bool)]
        test = ko.loc[ko["season"] == season]
        gbr = GradientBoostingRegressor(**config.GBR_PARAMS).fit(train[cols], train["ko_stage"])
        logit = LogisticRegression(**config.LOGIT_PARAMS).fit(train[cols], train["reached_final"].astype(int))
        coefs.append(logit.coef_[0].copy())
        fold = pd.DataFrame({
            "season": test["season"].to_numpy(),
            "team_id": test["team_id"].to_numpy(),
            "ko_stage": test["ko_stage"].to_numpy(),
            "reached_final": test["reached_final"].to_numpy(),
            "exp_stage": gbr.predict(test[cols]),
            "p_final": scale_to_two(logit.predict_proba(test[cols])[:, 1]),
        })
        fold["rank_in_season"] = fold["p_final"].rank(ascending=False, method="min").astype(int)
        fold["ko_size"] = len(test)
        fold["base_rate"] = 2 / len(test)
        predictions.append(fold)
        if with_shap:
            explainer = shap.TreeExplainer(gbr)
            values = pd.DataFrame(explainer.shap_values(test[cols]), columns=[f"shap_{f}" for f in features])
            values.insert(0, "base_value", float(np.ravel(explainer.expected_value)[0]))
            values.insert(0, "team_id", test["team_id"].to_numpy())
            values.insert(0, "season", test["season"].to_numpy())
            shap_frames.append(values)
    return LosoOutput(
        predictions=pd.concat(predictions, ignore_index=True),
        shap=pd.concat(shap_frames, ignore_index=True) if with_shap else None,
        fold_coefs=coefs,
    )


def evaluate(pred: pd.DataFrame) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # constant-input warnings in degenerate seasons
        rho = {int(s): float(spearmanr(g["exp_stage"], g["ko_stage"]).statistic) for s, g in pred.groupby("season")}
    finalists = pred["reached_final"].astype(bool)
    stage_rank = pred.groupby("season")["exp_stage"].rank(ascending=False, method="min")
    y = finalists.astype(int)
    return {
        "spearman_mean": float(np.nanmean(list(rho.values()))),
        "spearman_by_season": rho,
        "finalists_in_top4": float((stage_rank[finalists] <= 4).mean()),
        "auc": float(roc_auc_score(y, pred["p_final"])),
        "brier": float(brier_score_loss(y, pred["p_final"])),
        "brier_base_rate": float(brier_score_loss(y, pred["base_rate"])),
        "n_knockout": int(len(pred)),
        "n_finalists": int(y.sum()),
        "n_seasons": int(pred["season"].nunique()),
    }
