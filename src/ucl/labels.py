"""Round names → depth, how far each team went, and the finals table (spec §5.2, §5.4)."""
from __future__ import annotations

import pandas as pd

from . import config


def _city(match: dict) -> str | None:
    stadium = match.get("stadium") or {}
    return (((stadium.get("city") or {}).get("translations") or {}).get("name") or {}).get("EN")


def match_rows(raw_matches: list[dict], season: int) -> pd.DataFrame:
    """One row per match, keeping only the fields the pipeline uses."""
    rows = []
    for m in raw_matches:
        round_name = m["round"]["metaData"]["name"]
        if round_name not in config.ROUND_DEPTH:
            raise ValueError(f"unknown round name {round_name!r} in season {season}")
        score = m.get("score") or {}
        total = score.get("total") or {}
        pens = score.get("penalty") or {}
        winner = ((m.get("winner") or {}).get("match") or {}).get("team") or {}
        rows.append({
            "season": season,
            "match_id": str(m["id"]),
            "round": round_name,
            "depth": config.ROUND_DEPTH[round_name],
            "home_id": str(m["homeTeam"]["id"]),
            "home": m["homeTeam"]["internationalName"],
            "away_id": str(m["awayTeam"]["id"]),
            "away": m["awayTeam"]["internationalName"],
            "home_goals": total.get("home"),
            "away_goals": total.get("away"),
            "home_pens": pens.get("home"),
            "away_pens": pens.get("away"),
            "winner_id": str(winner["id"]) if winner.get("id") is not None else None,
            "city": _city(m),
        })
    return pd.DataFrame(rows)


def ko_stage(depth: int, won_final: bool) -> int | None:
    """0 out before QF, 1 QF, 2 SF, 3 runner-up, 4 winner; None outside the knockouts."""
    if depth == 0:
        return None
    if depth in (1, 2):
        return 0
    if depth == 5:
        return 4 if won_final else 3
    return depth - 2


def stage_label(depth: int, won_final: bool, season: int) -> str:
    if depth == 0:
        return "League phase" if config.is_league_format(season) else "Group stage"
    if depth == 5:
        return "Winner" if won_final else "Runner-up"
    return {1: "Knockout play-off", 2: "Round of 16", 3: "Quarter-finals", 4: "Semi-finals"}[depth]


def stages(matches: pd.DataFrame) -> pd.DataFrame:
    """Furthest round per (season, team): in_ko, ko_stage (NaN outside the knockouts), stage_label."""
    sides = [
        matches[["season", "depth", f"{side}_id", side]].rename(columns={f"{side}_id": "team_id", side: "team"})
        for side in ("home", "away")
    ]
    furthest = (
        pd.concat(sides, ignore_index=True)
        .sort_values("depth", kind="stable")
        .groupby(["season", "team_id"], as_index=False)
        .last()
    )
    final_winner = matches.loc[matches["round"] == "Final"].set_index("season")["winner_id"].to_dict()
    rows = []
    for r in furthest.itertuples(index=False):
        depth = int(r.depth)
        won = depth == 5 and final_winner.get(r.season) == r.team_id
        rows.append({
            "season": int(r.season),
            "team_id": r.team_id,
            "team": r.team,
            "in_ko": depth >= 1,
            "ko_stage": ko_stage(depth, won),
            "stage_label": stage_label(depth, won, int(r.season)),
        })
    out = pd.DataFrame(rows)
    out["ko_stage"] = out["ko_stage"].astype("float")
    return out


def finals_table(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per season with scores oriented to the winner (home/away in a final is nominal)."""
    rows = []
    for m in matches.loc[matches["round"] == "Final"].itertuples(index=False):
        if m.winner_id not in (m.home_id, m.away_id):
            raise ValueError(f"season {m.season}: final winner {m.winner_id!r} is not one of the finalists")
        w, r = ("home", "away") if m.winner_id == m.home_id else ("away", "home")
        rows.append({
            "season": int(m.season),
            "winner_id": getattr(m, f"{w}_id"),
            "winner": getattr(m, w),
            "runner_up_id": getattr(m, f"{r}_id"),
            "runner_up": getattr(m, r),
            "winner_goals": getattr(m, f"{w}_goals"),
            "runner_up_goals": getattr(m, f"{r}_goals"),
            "winner_pens": getattr(m, f"{w}_pens"),
            "runner_up_pens": getattr(m, f"{r}_pens"),
            "city": m.city,
        })
    return pd.DataFrame(rows).sort_values("season").reset_index(drop=True)
