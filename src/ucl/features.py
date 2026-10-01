"""Per-match rows → per-season features, normalisation and percentiles (spec §5.3)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _stat_values(entry: dict) -> dict[str, float]:
    values: dict[str, float] = {}
    for stat in entry.get("statistics", []):
        try:
            values[stat["name"]] = float(stat["value"])
        except (KeyError, TypeError, ValueError):
            continue
    return values


def team_match_rows(matches: pd.DataFrame, stats: dict[str, list[dict] | None]) -> pd.DataFrame:
    """One row per team per group/league-phase match: result, own stats, opponent shot stats."""
    rows = []
    for m in matches.loc[matches["depth"] == 0].itertuples(index=False):
        by_team = {str(e["teamId"]): _stat_values(e) for e in (stats.get(m.match_id) or [])}
        # The stats feed occasionally uses another id for a club than the match feed (Steaua 2614166
        # vs FCSB 50065). With one side unmatched and one entry left over, that entry is the unmatched side's.
        unmatched = [t for t in (m.home_id, m.away_id) if t not in by_team]
        leftover = [t for t in by_team if t not in (m.home_id, m.away_id)]
        if len(unmatched) == 1 and len(leftover) == 1:
            by_team[unmatched[0]] = by_team.pop(leftover[0])
        for side, other in (("home", "away"), ("away", "home")):
            team_id, opp_id = getattr(m, f"{side}_id"), getattr(m, f"{other}_id")
            gf, ga = int(getattr(m, f"{side}_goals")), int(getattr(m, f"{other}_goals"))
            own, opp = by_team.get(team_id), by_team.get(opp_id)
            row = {
                "season": int(m.season), "match_id": m.match_id, "team_id": team_id, "opp_id": opp_id,
                "gf": gf, "ga": ga, "points": 3 if gf > ga else 1 if gf == ga else 0,
                "has_stats": bool(own),
            }
            for name in config.OWN_STATS:
                row[name] = own.get(name, np.nan) if own else np.nan
            for name in config.OPP_STATS:
                row[f"opp_{name}"] = opp.get(name, np.nan) if opp else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _ratio(numerator: pd.Series, denominator: pd.Series) -> float:
    """Ratio of sums over the matches where both are present."""
    both = numerator.notna() & denominator.notna()
    total = denominator[both].sum()
    return float(numerator[both].sum() / total) if total > 0 else np.nan


def _season_row(g: pd.DataFrame) -> dict[str, float]:
    shots = g["attempts_on_target"] + g["attempts_off_target"]
    opp_shots = g["opp_attempts_on_target"] + g["opp_attempts_off_target"]
    return {
        "n_matches": len(g),
        "n_with_stats": int(g["has_stats"].sum()),
        "points_pg": g["points"].mean(),
        "goal_diff_pg": (g["gf"] - g["ga"]).mean(),
        "shots_pg": shots.mean(),
        "shot_accuracy": _ratio(g["attempts_on_target"], shots),
        "conversion": _ratio(g["goals"], shots),
        "attacks_pg": g["attacks"].mean(),
        "shots_against_pg": opp_shots.mean(),
        "on_target_against_pg": g["opp_attempts_on_target"].mean(),
        "save_pct": _ratio(g["saves"], g["opp_attempts_on_target"]),
        "possession_pct": g["ball_possession"].mean(),
        "pass_accuracy": _ratio(g["passes_completed"], g["passes_attempted"]),
        "passes_pg": g["passes_attempted"].mean(),
        "long_pass_share": _ratio(g["passes_long_attempted"], g["passes_attempted"]),
        "distance_km_pg": g["distance_covered"].mean(),
        "fouls_pg": g["fouls_committed"].mean(),
    }


def season_features(team_matches: pd.DataFrame) -> pd.DataFrame:
    """Per-match averages skip matches where the stat is missing (pandas mean skips NaN)."""
    rows = [
        {"season": int(season), "team_id": team_id, **_season_row(g)}
        for (season, team_id), g in team_matches.groupby(["season", "team_id"], sort=True)
    ]
    return pd.DataFrame(rows)


def covered_features(team_matches: pd.DataFrame, features: list[str]) -> tuple[list[str], dict[str, float]]:
    """Keep features whose source stats are present in >= COVERAGE_MIN of team-matches in every season."""
    kept: list[str] = []
    worst: dict[str, float] = {}
    for f in features:
        sources = config.FEATURE_SOURCES[f]
        if not sources:
            kept.append(f)
            continue
        present = team_matches[sources].notna().all(axis=1)
        worst[f] = float(present.groupby(team_matches["season"]).mean().min())
        if worst[f] >= config.COVERAGE_MIN:
            kept.append(f)
    return kept, worst


def coefficient_table(rankings: dict[int, list[dict]]) -> pd.DataFrame:
    """Ranking year Y applies to season Y + 1 (pre-season strength, spec §4)."""
    rows = [
        {"season": year + 1, "team_id": str(m["member"]["id"]), "coef": float(m["overallRanking"]["totalValue"])}
        for year, members in rankings.items()
        for m in members
    ]
    return pd.DataFrame(rows, columns=["season", "team_id", "coef"]).drop_duplicates(["season", "team_id"])


def attach_coefficients(team_seasons: pd.DataFrame, coefs: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, float]]:
    """Join by team id. Debutants absent from ranking Y-1 get the season minimum."""
    out = team_seasons.merge(coefs, on=["season", "team_id"], how="left")
    match_rate = {int(s): float(r) for s, r in out["coef"].notna().groupby(out["season"]).mean().items()}
    out["coef_imputed"] = out["coef"].isna()
    out["coef"] = out["coef"].fillna(out.groupby("season")["coef"].transform("min"))
    out["coef_log"] = np.log1p(out["coef"])
    return out, match_rate


def impute_season_median(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    n_missing = int(out[features].isna().sum().sum())
    out[features] = out[features].fillna(out.groupby("season")[features].transform("median"))
    return out, n_missing


def zscore_within_season(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Population z-scores against each season's field. Uses feature values only, never labels."""
    deviation = df[features] - df.groupby("season")[features].transform("mean")
    std = (deviation**2).groupby(df["season"]).transform("mean") ** 0.5
    return (deviation / std).add_prefix("z_")


def add_percentiles(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df.copy()
    for f in features:
        out[f"pct_season_{f}"] = (out.groupby("season")[f].rank(pct=True) * 100).round()
        out[f"pct_all_{f}"] = (out[f].rank(pct=True) * 100).round()
    return out


def display_value(row: pd.Series, feature: str) -> float | int:
    """Human-facing value: the raw coefficient for coef_log, percentages for 0-1 fractions."""
    _, decimals, is_fraction = config.FEATURE_META[feature]
    value = float(row["coef"] if feature == "coef_log" else row[feature])
    if is_fraction:
        value *= 100
    return int(round(value)) if decimals == 0 else round(value, decimals)
