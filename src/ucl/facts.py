"""Plain-English fact sheets the local LLM writes from (spec §7)."""
from __future__ import annotations

import math

import pandas as pd

from . import config
from .features import display_value
from .model import ModelResults, feature_list

BEATS_SEASON = "Teams beaten in that season's first phase (%)"
BEATS_ALL = "Teams beaten among all Champions League teams since 2011-12 (%)"
STAGES = ["out before quarter-finals", "quarter-finals", "semi-finals", "lost final", "won final"]
STRONGEST = "Strongest stats (most teams beaten first)"
WEAKEST = "Weakest stats (those that beat under half of that season's teams, fewest beaten first)"
NO_WEAK_STAT = "none: it beat at least half of that season's teams on every stat"
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


def nearest_stage(expected: float) -> str:
    """Name of the stage on the 0-4 scale closest to the model's expected stage; the LLM rounded this badly."""
    return STAGES[min(max(int(expected + 0.5), 0), len(STAGES) - 1)]


def pick_verdict(rank: int) -> str:
    """Whether the model would have picked a finalist: exactly two teams reach each final, so only its top two count."""
    if rank <= 2:
        return f"Yes: ranked {rank}, inside the model's top two"
    if rank <= 4:
        return f"No: ranked {rank}, a leading contender but outside the model's top two"
    return f"No: ranked {rank}, outside the model's top two"


def phase_format(season: int) -> str:
    """The first phase's name, stated outright: the other keys only say 'first phase', so the local model would
    guess which format a season had."""
    return "league phase" if config.is_league_format(season) else "group stage"


def phase_size(season: int) -> str:
    """The first phase's size, apart from its name: with both in one value the local model pasted the whole
    description into its prose."""
    if config.is_league_format(season):
        return "36 teams, 8 matches each"
    return "32 teams in groups of four, 6 matches each"


def beats(row: pd.Series, feature: str, scope: str) -> int:
    """Share of teams beaten on a stat. Percentiles are flipped where lower is better, because the local
    model kept reading a low percentile on those stats as a weakness."""
    pct = int(row[f"pct_{scope}_{feature}"])
    return 100 - pct if feature in config.LOWER_IS_BETTER else pct


def _stat_fact(row: pd.Series, feature: str) -> dict:
    return {
        "value": display_value(row, feature),
        BEATS_SEASON: beats(row, feature, "season"),
        BEATS_ALL: beats(row, feature, "all"),
    }


def team_fact_sheet(row: pd.Series, pred: pd.Series, shap_row: pd.Series, final, features: list[str]) -> dict:
    contributions = sorted(((f, float(shap_row[f"shap_{f}"])) for f in features), key=lambda kv: kv[1], reverse=True)

    def stat_entry(f: str) -> dict:
        return {"stat": stat_label(f), **_stat_fact(row, f)}

    def driver(f: str, c: float) -> dict:
        return {**stat_entry(f), "SHAP contribution (knockout stages)": round(c, 2)}

    # picked here rather than left to the LLM, which chose them from the SHAP lists instead
    ranked = sorted(features, key=lambda f: beats(row, f, "season"), reverse=True)

    return {
        "Club": row["team_display"],
        "Season": config.season_label(int(row["season"])),
        "First-phase format that season": phase_format(int(row["season"])),
        "First-phase size": phase_size(int(row["season"])),
        "Actual result": result_text(row["team_id"], final),
        "Knockout teams that season": int(pred["ko_size"]),
        "Base rate: chance a random knockout team reaches the final (%)": round(100 * float(pred["base_rate"]), 1),
        "Model probability of reaching the final (%)": round(100 * float(pred["p_final"]), 1),
        "Rank by that probability among the season's knockout teams": int(pred["rank_in_season"]),
        "Would the model have picked them?": pick_verdict(int(pred["rank_in_season"])),
        "Model's expected knockout stage (0 = out before quarter-finals, 1 = quarter-finals, "
        "2 = semi-finals, 3 = lost final, 4 = won final)": round(float(pred["exp_stage"]), 2),
        "Stage on that scale nearest to the model's expectation": nearest_stage(round(float(pred["exp_stage"]), 2)),
        STRONGEST: [stat_entry(f) for f in ranked[:3]],
        WEAKEST: [stat_entry(f) for f in reversed(ranked) if beats(row, f, "season") < 50][:3] or NO_WEAK_STAT,
        "Stats that pushed the prediction up most": [driver(f, c) for f, c in contributions if c > 0][:4],
        "Stats that pushed the prediction down most (the model's view; the team may still rank high on them)": [
            driver(f, c) for f, c in reversed(contributions) if c < 0
        ][:3],
        "All first-phase stats": {stat_label(f): _stat_fact(row, f) for f in features},
    }


def _interval(bounds, digits: int, scale: float = 1.0) -> list[float]:
    """A [lo, hi] pair rounded like the point value it goes with."""
    return [round(scale * b, digits) for b in bounds]


def vs_no_skill(bounds, no_skill: float, edge: str) -> str:
    """Whether an interval reaches its no-skill value, stated outright: the local model kept calling 30.0-60.0
    'inconclusive' against 23.9 when left to compare them."""
    lo, hi = bounds
    if lo <= no_skill <= hi:
        return f"Yes: the {edge} is inconclusive"
    return f"No: the whole interval is {'above' if lo > no_skill else 'below'} it"


def synthesis_facts(team_seasons: pd.DataFrame, results: ModelResults) -> dict:
    m = results.metrics
    ci = m["ci"]
    needed = math.ceil(config.ROBUST_SHARE * m["n_seasons"])
    skill_ci, top4_ci = _interval(ci["brier_skill"], 2), _interval(ci["finalists_in_top4"], 1, 100)
    top4_chance = round(100 * m["finalists_in_top4_chance"], 1)
    finalists = results.predictions.loc[results.predictions["is_target"].astype(bool)].sort_values(
        ["season", "ko_stage"], ascending=[True, False]
    )
    return {
        "Context": {
            "Seasons analysed": m["n_seasons"],
            "Finals": m["n_seasons"],
            "Finalists": m["n_finalists"],
            "Group/league-phase team-seasons": int(len(team_seasons)),
            "Knockout team-seasons (what the model learns from, using each team's group/league-phase stats)":
                m["n_knockout"],
        },
        "Model quality (each season scored by a model trained on the other seasons)": {
            "Intervals": "95% intervals from resampling seasons",
            "Mean within-season Spearman correlation, predicted vs actual stage": round(m["spearman_mean"], 2),
            "95% interval for the mean Spearman correlation": _interval(ci["spearman_mean"], 2),
            "Share of actual finalists in the model's top 4 of their season (%)": round(100 * m["finalists_in_top4"], 1),
            "95% interval for the share of finalists in the model's top 4 (%)": top4_ci,
            "Share of finalists a random ranking would put in the top 4 (%)": top4_chance,
            "Does the top-4 share interval include the random-ranking share?":
                vs_no_skill(top4_ci, top4_chance, "edge over a random ranking"),
            "AUC for reaching the final": round(m["auc"], 2),
            "95% interval for the AUC": _interval(ci["auc"], 2),
            "Brier score": round(m["brier"], 3),
            "Brier score of the base-rate guess": round(m["brier_base_rate"], 3),
            "Brier skill vs the base-rate guess (0 = no better, 1 = perfect)": round(m["brier_skill"], 2),
            "95% interval for the Brier skill": skill_ci,
            "Does the Brier skill interval include 0 (no skill)?":
                vs_no_skill(skill_ci, 0, "edge over the base-rate guess"),
        },
        "Robustness rule": f"Only the top {config.TOP_DRIVERS} stats are labelled. "
                           f"robust = the logistic model agrees on the direction in at least {needed} of "
                           f"{m['n_seasons']} seasons and the stat points that way on its own too; "
                           f"conditional = the logistic model agrees in at least {needed} of {m['n_seasons']} "
                           "seasons, but on its own the stat points the other way or barely at all, so the "
                           "direction holds only with the other stats held fixed; "
                           f"model-dependent = the logistic model agrees in fewer than {needed} of "
                           f"{m['n_seasons']} seasons, so the direction depends on the model used",
        "What drives deep runs, most important first": [
            {"stat": stat_label(r.feature), "group": r.group,
             "importance (mean |SHAP|, knockout stages)": round(r.importance, 3),
             "direction": DIRECTION[int(r.direction)],
             "direction on its own (Spearman with knockout stage)": round(r.marginal_rho, 2),
             "robustness label": r.label or f"not in top {config.TOP_DRIVERS}"}
            for r in results.drivers.head(8).itertuples()
        ],
        "Feature-set comparison (Spearman / AUC)": [
            {"feature set": r.feature_set, "number of stats": int(r.n_features),
             "Spearman": round(r.spearman, 2), "Spearman 95% interval": _interval((r.spearman_lo, r.spearman_hi), 2),
             "AUC": round(r.auc, 2), "AUC 95% interval": _interval((r.auc_lo, r.auc_hi), 2)}
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
