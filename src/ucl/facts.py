"""Plain-English fact sheets the local LLM writes from (spec §7)."""
from __future__ import annotations

import math

import pandas as pd

from . import config
from .features import display_value
from .model import ModelResults, feature_list

PCT_SEASON = "Percentile vs that season's teams"
PCT_ALL = "Percentile vs all Champions League teams since 2011-12"
SIGNIFICANCE = 0.05
DIRECTION = {1: "higher helps", -1: "lower helps", 0: "no clear direction"}


def stat_label(feature: str) -> str:
    label = config.FEATURE_META[feature][0]
    return f"{label} (lower is better)" if feature in config.LOWER_IS_BETTER else label


def _name(team_id: str, fallback: str) -> str:
    return config.DISPLAY_NAMES.get(team_id, fallback)


def result_text(team_id: str, final) -> str:
    """How a finalist's final went, phrased from that team's side."""
    wg, rg = int(final["winner_goals"]), int(final["runner_up_goals"])
    winner = _name(final["winner_id"], final["winner"])
    runner_up = _name(final["runner_up_id"], final["runner_up"])
    if pd.notna(final["winner_pens"]):
        wp, rp = int(final["winner_pens"]), int(final["runner_up_pens"])
        if team_id == final["winner_id"]:
            return f"Winner: beat {runner_up} on penalties in the final ({wg}-{rg}, {wp}-{rp} on penalties)"
        return f"Runner-up: lost the final to {winner} on penalties ({rg}-{wg}, {rp}-{wp} on penalties)"
    if team_id == final["winner_id"]:
        return f"Winner: beat {runner_up} {wg}-{rg} in the final"
    return f"Runner-up: lost the final to {winner} {rg}-{wg}"


def _stat_fact(row: pd.Series, feature: str) -> dict:
    return {
        "value": display_value(row, feature),
        PCT_SEASON: int(row[f"pct_season_{feature}"]),
        PCT_ALL: int(row[f"pct_all_{feature}"]),
    }


def team_fact_sheet(row: pd.Series, pred: pd.Series, shap_row: pd.Series, final, features: list[str]) -> dict:
    contributions = sorted(((f, float(shap_row[f"shap_{f}"])) for f in features), key=lambda kv: kv[1], reverse=True)

    def driver(f: str, c: float) -> dict:
        return {"stat": stat_label(f), **_stat_fact(row, f), "SHAP contribution (knockout stages)": round(c, 2)}

    return {
        "Club": row["team_display"],
        "Season": config.season_label(int(row["season"])),
        "Actual result": result_text(row["team_id"], final),
        "Knockout teams that season": int(pred["ko_size"]),
        "Base rate: chance a random knockout team reaches the final (%)": round(100 * float(pred["base_rate"]), 1),
        "Model probability of reaching the final (%)": round(100 * float(pred["p_final"]), 1),
        "Rank by that probability among the season's knockout teams": int(pred["rank_in_season"]),
        "Model's expected knockout stage (0 = out before quarter-finals, 1 = quarter-finals, "
        "2 = semi-finals, 3 = lost final, 4 = won final)": round(float(pred["exp_stage"]), 2),
        "Stats that pushed the prediction up most": [driver(f, c) for f, c in contributions if c > 0][:4],
        "Stats that pushed the prediction down most": [driver(f, c) for f, c in reversed(contributions) if c < 0][:3],
        "All league/group-phase stats": {stat_label(f): _stat_fact(row, f) for f in features},
    }


def synthesis_facts(team_seasons: pd.DataFrame, results: ModelResults) -> dict:
    m = results.metrics
    needed = math.ceil(config.ROBUST_SHARE * m["n_seasons"])
    finalists = results.predictions.loc[results.predictions["is_target"].astype(bool)].sort_values(
        ["season", "ko_stage"], ascending=[True, False]
    )
    return {
        "Context": {
            "Seasons analysed": m["n_seasons"],
            "Finals": m["n_seasons"],
            "Finalists": m["n_finalists"],
            "Group/league-phase team-seasons": int(len(team_seasons)),
            "Knockout team-seasons": m["n_knockout"],
        },
        "Model quality (each season scored by a model trained on the other seasons)": {
            "Mean within-season Spearman correlation, predicted vs actual stage": round(m["spearman_mean"], 2),
            "Share of actual finalists in the model's top 4 of their season (%)": round(100 * m["finalists_in_top4"], 1),
            "AUC for reaching the final": round(m["auc"], 2),
            "Brier score": round(m["brier"], 3),
            "Brier score of the base-rate guess": round(m["brier_base_rate"], 3),
        },
        "Robustness rule": f"robust = the logistic model agrees on the direction in at least {needed} of "
                           f"{m['n_seasons']} seasons; only the top {config.TOP_DRIVERS} stats get a robustness label",
        "What drives deep runs, most important first": [
            {"stat": stat_label(r.feature), "group": r.group,
             "importance (mean |SHAP|, knockout stages)": round(r.importance, 3),
             "direction": DIRECTION[int(r.direction)], "robustness label": r.label or f"not in top {config.TOP_DRIVERS}"}
            for r in results.drivers.head(8).itertuples()
        ],
        "Feature-set comparison (Spearman / AUC)": [
            {"feature set": r.feature_set, "number of stats": int(r.n_features),
             "Spearman": round(r.spearman, 2), "AUC": round(r.auc, 2)}
            for r in results.ablation.itertuples()
        ],
        "The 10 finalists": [
            {"club": r.team_display, "season": config.season_label(int(r.season)),
             "result": "Winner" if int(r.ko_stage) == 4 else "Runner-up",
             "model probability of reaching the final (%)": round(100 * r.p_final, 1),
             "rank among the season's knockout teams": int(r.rank_in_season), "knockout teams": int(r.ko_size)}
            for r in finalists.itertuples()
        ],
        "Significance threshold for Holm-adjusted p": SIGNIFICANCE,
        "Winners vs runners-up across all finals (z-score difference, winner minus runner-up)": [
            {"stat": stat_label(r.feature), "finals where the winner was higher": int(r.higher),
             "finals where the winner was lower": int(r.lower), "ties": int(r.tied),
             "mean difference (z)": round(r.mean_diff, 2), "Holm-adjusted p": round(r.p_holm, 3)}
            for r in results.finals_compare.sort_values("p_holm", kind="stable").itertuples()
        ],
    }


def build_facts(team_seasons: pd.DataFrame, finals: pd.DataFrame, results: ModelResults) -> dict[str, dict]:
    """One sheet per target finalist (keyed '{season}-{team_id}') plus 'synthesis'."""
    features = feature_list(results)
    rows = team_seasons.set_index(["season", "team_id"], drop=False)  # sheets read row["season"]
    preds = results.predictions.set_index(["season", "team_id"])
    shap_rows = results.shap.set_index(["season", "team_id"])
    finals_by_season = finals.set_index("season")
    sheets: dict[str, dict] = {}
    for season, team_id in preds.index[preds["is_target"].astype(bool)]:
        sheets[f"{season}-{team_id}"] = team_fact_sheet(
            rows.loc[(season, team_id)], preds.loc[(season, team_id)], shap_rows.loc[(season, team_id)],
            finals_by_season.loc[season], features,
        )
    sheets["synthesis"] = synthesis_facts(team_seasons, results)
    return sheets
