"""Build, validate, save and load the analysis dataset (spec §5)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from . import config
from . import features as feat
from . import labels
from .uefa import UefaClient

ID_COLUMNS = ["team_id", "opp_id", "match_id", "home_id", "away_id", "winner_id", "runner_up_id"]


class ValidationError(Exception):
    """The built dataset breaks one of the spec §5.5 rules."""


@dataclass
class Dataset:
    matches: pd.DataFrame
    team_match_stats: pd.DataFrame
    team_seasons: pd.DataFrame
    finals: pd.DataFrame
    features: list[str]
    notes: dict = field(default_factory=dict)


def fetch_all(client: UefaClient, seasons: list[int] = config.SEASONS,
              log: Callable[[str], None] = print) -> dict[str, str]:
    """Fill the raw cache. Returns {match id: reason} for stats that failed after retries."""
    failed_all: dict[str, str] = {}
    for season in seasons:
        matches = labels.match_rows(client.matches(season), season)
        client.coefficients(season - 1)
        phase_ids = matches.loc[matches["depth"] == 0, "match_id"].tolist()
        stats, failed = client.team_match_stats_many(phase_ids)
        missing = sum(1 for v in stats.values() if v is None)
        log(f"{season}: {len(matches)} matches, {len(phase_ids)} phase matches, "
            f"stats missing {missing}, failed {len(failed)}")
        if failed:
            match_id, reason = next(iter(failed.items()))
            log(f"  e.g. match {match_id}: {reason}")
        failed_all.update(failed)
        if phase_ids and len(failed) == len(phase_ids):
            log("  every request failed for this season; stopping. Fix the cause above, then run `uv run ucl fetch` again.")
            break
    return failed_all


def plausibility_errors(team_seasons: pd.DataFrame) -> list[str]:
    """Team-seasons whose raw average is outside config.PLAUSIBLE_RANGES: a unit error that slipped through."""
    errors = []
    for feature, (low, high) in config.PLAUSIBLE_RANGES.items():
        if feature not in team_seasons:
            continue
        values = team_seasons[feature]
        bad = team_seasons[values.notna() & ~values.between(low, high)]
        errors += [f"{season} {team}: {feature} = {value:g} is outside {low:g}-{high:g}"
                   for season, team, value in bad[["season", "team", feature]].itertuples(index=False)]
    return errors


def build(client: UefaClient, seasons: list[int] = config.SEASONS) -> Dataset:
    matches = pd.concat([labels.match_rows(client.matches(s), s) for s in seasons], ignore_index=True)
    stats, failed = client.team_match_stats_many(matches.loc[matches["depth"] == 0, "match_id"].tolist())
    if failed:
        match_id, reason = next(iter(failed.items()))
        raise RuntimeError(f"{len(failed)} match-stat requests failed (e.g. {match_id}: {reason}); "
                           "run `uv run ucl fetch` again first")
    tm = feat.team_match_rows(matches, stats)
    kept, coverage = feat.covered_features(tm, config.FEATURES)

    ts = labels.stages(matches).merge(feat.season_features(tm), on=["season", "team_id"], how="left")
    errors = plausibility_errors(ts)  # observed averages only, so run before imputation fills the gaps
    if errors:
        raise ValidationError("\n".join(errors))
    coefs = feat.coefficient_table({s - 1: client.coefficients(s - 1) for s in seasons})
    ts, coef_rates = feat.attach_coefficients(ts, coefs)
    ts, n_imputed = feat.impute_season_median(ts, kept)
    ts = pd.concat([ts, feat.zscore_within_season(ts, kept)], axis=1)
    ts = feat.add_percentiles(ts, kept)
    ts["complete"] = ts["n_with_stats"] >= config.MIN_MATCHES_WITH_STATS
    ts["reached_final"] = ts["ko_stage"].ge(3)
    ts["is_target"] = ts["season"].isin(config.TARGET_SEASONS) & ts["reached_final"]
    ts["team_display"] = ts["team_id"].map(config.DISPLAY_NAMES).fillna(ts["team"])
    ts = ts.sort_values(["season", "team_id"]).reset_index(drop=True)

    notes = {
        "coverage": coverage,
        "dropped_features": [f for f in config.FEATURES if f not in kept],
        "imputed_values": n_imputed,
        "coef_match_rate": coef_rates,
        "coef_imputed": int(ts["coef_imputed"].sum()),
        "team_matches_without_stats": int((~tm["has_stats"]).sum()),
    }
    return Dataset(matches, tm, ts, labels.finals_table(matches), kept, notes)


def validate(ds: Dataset) -> None:
    """Raise ValidationError listing every broken spec §5.5 rule."""
    errors: list[str] = []
    ts, matches, finals = ds.team_seasons, ds.matches, ds.finals
    for season, g in ts.groupby("season"):
        season = int(season)
        if len(g) != config.field_size(season):
            errors.append(f"{season}: field has {len(g)} teams, expected {config.field_size(season)}")
        wrong = g[g["n_matches"] != config.phase_matches(season)]
        if len(wrong):
            errors.append(f"{season}: {len(wrong)} teams lack {config.phase_matches(season)} group/league matches")
        ko = g[g["in_ko"]]
        if len(ko) != config.ko_size(season):
            errors.append(f"{season}: knockout population is {len(ko)}, expected {config.ko_size(season)}")
        levels = [int((ko["ko_stage"] >= k).sum()) for k in (1, 2, 3, 4)]
        if levels != [8, 4, 2, 1]:
            errors.append(f"{season}: teams at ko_stage >= 1..4 are {levels}, expected [8, 4, 2, 1]")
    final_counts = matches.loc[matches["round"] == "Final"].groupby("season").size()
    for season in sorted(ts["season"].unique()):
        if int(final_counts.get(season, 0)) != 1:
            errors.append(f"{season}: expected exactly one final match")
    for season, expected in config.EXPECTED_FINALS.items():
        row = finals.loc[finals["season"] == season]
        if season not in set(ts["season"]):
            continue
        if row.empty or (row.iloc[0]["winner_id"], row.iloc[0]["runner_up_id"]) != (
            expected["winner_id"], expected["runner_up_id"]
        ):
            errors.append(f"{season}: final does not match the expected winner/runner-up ids")
    targets = ts.loc[ts["is_target"]]
    expected_targets = 2 * len([s for s in config.TARGET_SEASONS if s in set(ts["season"])])
    if len(targets) != expected_targets:
        errors.append(f"found {len(targets)} target finalists, expected {expected_targets}")
    incomplete = targets.loc[~targets["complete"]]
    if len(incomplete):
        names = ", ".join(f"{r.team} {r.season}" for r in incomplete.itertuples())
        errors.append(f"target finalists without enough stats: {names}")
    zcols = [f"z_{f}" for f in ds.features]
    if ts[zcols].isna().any().any():
        errors.append("NaN in model features after imputation")
    for season, rate in ds.notes.get("coef_match_rate", {}).items():
        if rate < config.COEF_MATCH_MIN:
            errors.append(f"{season}: only {rate:.0%} of clubs matched a coefficient")
    if errors:
        raise ValidationError("\n".join(errors))


def save(ds: Dataset, directory: Path = config.PROCESSED_DIR) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ds.matches.to_csv(directory / "matches.csv", index=False)
    ds.team_match_stats.to_csv(directory / "team_match_stats.csv", index=False)
    ds.team_seasons.to_csv(directory / "team_seasons.csv", index=False)
    ds.finals.to_csv(directory / "finals.csv", index=False)
    tm = ds.team_match_stats
    tm.loc[~tm["has_stats"].astype(bool), ["season", "match_id", "team_id"]].to_csv(
        directory / "missing_stats.csv", index=False
    )
    meta = {"features": ds.features, "notes": ds.notes}
    (directory / "dataset.json").write_text(json.dumps(meta, indent=2, default=str))


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV keeping id columns as strings (ids like '007' must survive)."""
    header = pd.read_csv(path, nrows=0).columns
    return pd.read_csv(path, dtype={c: str for c in ID_COLUMNS if c in header})


def load(directory: Path = config.PROCESSED_DIR) -> Dataset:
    directory = Path(directory)
    meta = json.loads((directory / "dataset.json").read_text())
    return Dataset(
        matches=read_csv(directory / "matches.csv"),
        team_match_stats=read_csv(directory / "team_match_stats.csv"),
        team_seasons=read_csv(directory / "team_seasons.csv"),
        finals=read_csv(directory / "finals.csv"),
        features=meta["features"],
        notes=meta["notes"],
    )
