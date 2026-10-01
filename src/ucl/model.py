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


def feature_sets(features: list[str]) -> dict[str, list[str]]:
    groups = {g: [f for f in fs if f in features] for g, fs in config.FEATURE_GROUPS.items()}
    sets: dict[str, list[str]] = dict(groups)
    sets["all"] = list(features)
    for name, fs in groups.items():
        sets[f"all minus {name}"] = [f for f in features if f not in fs]
    return {name: fs for name, fs in sets.items() if fs}


def ablation(ko: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for name, fs in feature_sets(features).items():
        metrics = evaluate(loso(ko, fs, with_shap=False).predictions)
        rows.append({"feature_set": name, "n_features": len(fs),
                     "spearman": metrics["spearman_mean"], "auc": metrics["auc"]})
    return pd.DataFrame(rows)


def drivers(ko: pd.DataFrame, shap_df: pd.DataFrame, fold_coefs: list[np.ndarray],
            features: list[str]) -> pd.DataFrame:
    merged = shap_df.merge(ko[["season", "team_id", *[f"z_{f}" for f in features]]], on=["season", "team_id"])
    coefs = np.vstack(fold_coefs)
    rows = []
    for i, f in enumerate(features):
        values = merged[f"shap_{f}"].to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = spearmanr(merged[f"z_{f}"].to_numpy(), values).statistic
        direction = 0 if np.isnan(corr) or corr == 0 else int(np.sign(corr))
        rows.append({
            "feature": f,
            "group": config.FEATURE_GROUP[f],
            "label_text": config.FEATURE_META[f][0],
            "importance": float(np.mean(np.abs(values))),
            "direction": direction,
            "logit_coef_mean": float(coefs[:, i].mean()),
            "sign_agree_folds": int((np.sign(coefs[:, i]) == direction).sum()) if direction else 0,
        })
    out = pd.DataFrame(rows).sort_values("importance", ascending=False, kind="stable").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    top = out["rank"] <= config.TOP_DRIVERS
    robust = out["sign_agree_folds"] >= math.ceil(config.ROBUST_SHARE * len(fold_coefs))
    out["label"] = np.where(top & robust, "robust", np.where(top, "model-dependent", ""))
    return out


def holm(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def sign_counts(diffs: np.ndarray, tol: float = 1e-9) -> tuple[int, int, int, float]:
    """(higher, lower, tied, two-sided sign-test p on the non-tied finals)."""
    higher, lower = int((diffs > tol).sum()), int((diffs < -tol).sum())
    n = higher + lower
    p = float(binomtest(higher, n, 0.5).pvalue) if n else 1.0
    return higher, lower, len(diffs) - n, p


def finals_compare(team_seasons: pd.DataFrame, finals: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Winner minus runner-up z-scores across every final, with sign tests and Holm adjustment."""
    by_key = team_seasons.set_index(["season", "team_id"])
    diffs: dict[str, dict[int, float]] = {f: {} for f in features}
    for fr in finals.itertuples(index=False):
        winner = by_key.loc[(fr.season, fr.winner_id)]
        runner_up = by_key.loc[(fr.season, fr.runner_up_id)]
        for f in features:
            diffs[f][int(fr.season)] = float(winner[f"z_{f}"] - runner_up[f"z_{f}"])
    rows = []
    for f in features:
        values = np.array(list(diffs[f].values()))
        higher, lower, tied, p = sign_counts(values)
        row = {"feature": f, "label_text": config.FEATURE_META[f][0], "higher": higher, "lower": lower,
               "tied": tied, "mean_diff": float(values.mean()), "p": p}
        row.update({f"diff_{s}": diffs[f].get(s, np.nan) for s in config.TARGET_SEASONS})
        rows.append(row)
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"])
    return out


def run(team_seasons: pd.DataFrame, finals: pd.DataFrame, features: list[str]) -> ModelResults:
    ko = knockout_population(team_seasons)
    main = loso(ko, features)
    display = team_seasons[["season", "team_id", "team_display", "is_target", "complete"]]
    return ModelResults(
        predictions=main.predictions.merge(display, on=["season", "team_id"], how="left"),
        shap=main.shap,
        drivers=drivers(ko, main.shap, main.fold_coefs, features),
        ablation=ablation(ko, features),
        metrics=evaluate(main.predictions),
        finals_compare=finals_compare(team_seasons, finals, features),
    )


def feature_list(results: ModelResults) -> list[str]:
    """Active features in config order (drivers are sorted by importance)."""
    active = set(results.drivers["feature"])
    return [f for f in config.FEATURES if f in active]


FRAMES = {
    "predictions": "predictions.csv",
    "shap": "shap.csv",
    "drivers": "drivers.csv",
    "ablation": "ablation.csv",
    "finals_compare": "finals_compare.csv",
}


def save(results: ModelResults, directory: Path = config.OUT_DIR) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for attr, name in FRAMES.items():
        getattr(results, attr).to_csv(directory / name, index=False)
    (directory / "metrics.json").write_text(json.dumps(results.metrics, indent=2))


def load(directory: Path = config.OUT_DIR) -> ModelResults:
    from .dataset import read_csv

    directory = Path(directory)
    frames = {attr: read_csv(directory / name) for attr, name in FRAMES.items()}
    frames["drivers"]["label"] = frames["drivers"]["label"].fillna("")
    return ModelResults(metrics=json.loads((directory / "metrics.json").read_text()), **frames)
