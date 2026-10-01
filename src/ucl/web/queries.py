"""Pure functions that shape a snapshot into the API's JSON (spec §5). app.py adds HTTP; everything returned
here has been through to_jsonable."""
from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd

from .. import config, facts
from ..features import display_value
from ..report import md_to_html  # escapes the model's text, so the browser can render the HTML as is
from .jsonsafe import to_jsonable
from .pipeline import CORE, ORDER, STAGE_LABELS

SECTION_OF = {feature: section for section, features in config.STAT_SECTIONS.items() for feature in features}
SEARCH_LIMIT = 12
MIN_QUERY = 2
TIE_Z = 0.05  # a smaller gap in oriented z counts as level
HELPS = {1: "higher", -1: "lower"}
# letters NFKD doesn't split into a base letter and an accent
LETTERS = str.maketrans({"ø": "o", "æ": "ae", "œ": "oe", "ł": "l", "đ": "d", "ı": "i", "þ": "th", "ð": "d"})


class NotFound(Exception):
    """An unknown team or season (HTTP 404)."""


def feature_meta(feature: str) -> dict:
    label, decimals, fraction = config.FEATURE_META[feature]
    return {"feature": feature, "label": label, "section": SECTION_OF[feature],
            "group": config.FEATURE_GROUP[feature], "decimals": decimals,
            "percent": fraction or feature == "possession_pct",
            "lower_is_better": feature in config.LOWER_IS_BETTER}


def meta(snapshot, stage_names: list[str] | None = None, live=None, player_stats: list[dict] | None = None) -> dict:
    """What the UI needs before anything else. `live` is the LiveState (None without the live season)."""
    ds = snapshot.dataset
    seasons = sorted(int(s) for s in ds.team_seasons["season"].unique()) if ds is not None else list(config.SEASONS)
    names = stage_names if stage_names is not None else CORE
    listed = [{"season": s, "label": config.season_label(s), "live": False} for s in seasons]
    if live is not None and live.available:
        listed.append({"season": live.snapshot.season, "label": config.season_label(live.snapshot.season),
                       "live": True})
    return to_jsonable({
        "ready": snapshot.ready,
        "problems": list(snapshot.errors.values()),
        "seasons": listed,
        "live_season": config.LIVE_SEASON if live is not None else None,
        "live_status": live.status if live is not None else "unavailable",
        "live_message": live.message if live is not None else "",
        "sections": list(config.STAT_SECTIONS),
        "features": [feature_meta(f) for f in ds.features] if ds is not None else [],
        "player_stats": player_stats or [],
        "stages": [{"name": name, "label": STAGE_LABELS[name], "optional": name not in CORE}
                   for name in ORDER if name in names],
        "data": {"built_at": snapshot.built_at, "modelled_at": snapshot.modelled_at,
                 "analysed_at": snapshot.analysed_at},
    })


def fold(text: str) -> str:
    """Lower case without accents or punctuation, so 'Atlético' matches 'atletico' and 'B. Dortmund' 'b dortmund'."""
    decomposed = unicodedata.normalize("NFKD", str(text).casefold().translate(LETTERS))
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join("".join(c if c.isalnum() else " " for c in plain).split())


def search(team_seasons: pd.DataFrame, q: str, limit: int = SEARCH_LIMIT) -> list[dict]:
    """Teams whose display name, UEFA name or alias contains `q`; names that start with it come first, then the
    most recent season."""
    query = fold(q)
    if len(query) < MIN_QUERY:
        return []
    aliases: dict[str, list[str]] = {}
    for alias, team_id in config.TEAM_SEARCH_ALIASES.items():
        aliases.setdefault(team_id, []).append(alias)
    hits = []
    for team_id, rows in team_seasons.groupby("team_id", sort=False):
        rows = rows.sort_values("season", ascending=False)
        names = {fold(n) for n in (*rows["team_display"], *rows["team"], *aliases.get(team_id, []))}
        if any(n.startswith(query) for n in names):
            rank = 0
        elif any(query in n for n in names):
            rank = 1
        else:
            continue
        latest = rows.iloc[0]
        hits.append((rank, -int(latest["season"]), fold(latest["team_display"]), {
            "team_id": team_id,
            "name": latest["team_display"],
            "seasons": [{"season": int(s), "label": config.season_label(int(s)), "stage_label": label}
                        for s, label in zip(rows["season"], rows["stage_label"])],
        }))
    hits.sort(key=lambda hit: hit[:3])
    return to_jsonable([hit[3] for hit in hits[:limit]])


def _row(team_seasons: pd.DataFrame, team_id: str, season: int) -> pd.Series:
    hit = team_seasons[(team_seasons["team_id"] == team_id) & (team_seasons["season"] == season)]
    if hit.empty:
        raise NotFound(f"no team {team_id} in {config.season_label(season)}")
    return hit.iloc[0]


def _header(row: pd.Series) -> dict:
    season = int(row["season"])
    return {"team_id": row["team_id"], "name": row["team_display"], "season": season,
            "label": config.season_label(season), "stage_label": row["stage_label"]}


def value_of(row: pd.Series, feature: str):
    raw = row["coef"] if feature == "coef_log" else row[feature]
    return None if pd.isna(raw) else display_value(row, feature)


def beats_of(row: pd.Series, feature: str, scope: str) -> int | None:
    return None if pd.isna(row.get(f"pct_{scope}_{feature}")) else facts.beats(row, feature, scope)


def _flag(row: pd.Series, name: str) -> bool:
    """A boolean column that only live rows carry; NaN (a historical row in a mixed frame) means False."""
    value = row.get(name)
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _is_live(row: pd.Series) -> bool:
    return _flag(row, "live")


def _stat(row: pd.Series, feature: str) -> dict:
    return {"feature": feature, "label": config.FEATURE_META[feature][0], "section": SECTION_OF[feature],
            "value": value_of(row, feature), "z": row.get(f"z_{feature}"),
            "beats_season": beats_of(row, feature, "season"), "beats_all": beats_of(row, feature, "all"),
            "provisional": _flag(row, "provisional")}


def _point(row: pd.Series, feature: str) -> dict:
    season = int(row["season"])
    return {"season": season, "label": config.season_label(season), "value": value_of(row, feature),
            "beats_season": beats_of(row, feature, "season"), "live": _is_live(row)}


def _result(finals: pd.DataFrame, team_id: str, season: int) -> str | None:
    final = finals[finals["season"] == season]
    if final.empty or team_id not in (final.iloc[0]["winner_id"], final.iloc[0]["runner_up_id"]):
        return None
    return facts.result_text(team_id, final.iloc[0])


def _model_card(results, row: pd.Series, features: list[str]) -> dict | None:
    if results is None:
        return None
    season, team_id = int(row["season"]), row["team_id"]
    preds = results.predictions
    hit = preds[(preds["season"] == season) & (preds["team_id"] == team_id)]
    if hit.empty:
        return None  # not a knockout team, or model outputs older than the dataset
    pred = hit.iloc[0]
    shap = results.shap
    shap_row = shap[(shap["season"] == season) & (shap["team_id"] == team_id)]
    columns = [f for f in features if f"shap_{f}" in shap.columns]
    contributions = [] if shap_row.empty else sorted(
        ((f, float(shap_row.iloc[0][f"shap_{f}"])) for f in columns), key=lambda kv: kv[1], reverse=True)

    def entry(feature: str, contribution: float) -> dict:
        return {"feature": feature, "label": config.FEATURE_META[feature][0], "value": value_of(row, feature),
                "contribution": round(contribution, 2)}  # as in the facts the AI write-ups quote

    exp_stage = float(pred["exp_stage"])
    return {"p_final": float(pred["p_final"]), "base_rate": float(pred["base_rate"]),
            "rank": int(pred["rank_in_season"]), "ko_size": int(pred["ko_size"]), "exp_stage": exp_stage,
            "nearest_stage": facts.nearest_stage(round(exp_stage, 2)),
            "top_up": [entry(f, c) for f, c in contributions if c > 0][:4],
            "top_down": [entry(f, c) for f, c in reversed(contributions) if c < 0][:3]}


def _narrative(analysis, key: str, stale_keys: frozenset[str]) -> dict | None:
    """The write-up for `key` with its badge; None when there is none (only the finalists and the synthesis)."""
    narrative = analysis.narratives.get(key) if analysis is not None else None
    if narrative is None:
        return None
    if narrative.status != "ok" or not narrative.text:
        return {"text": None, "html": None, "badge": {"kind": "unavailable", "label": "AI write-up unavailable"}}
    if key in stale_keys:
        badge = {"kind": "stale", "label": "Written for earlier numbers"}
    elif narrative.unsupported:
        n = len(narrative.unsupported)
        badge = {"kind": "warn", "label": f"{n} figure{'s' if n != 1 else ''} not found in the data"}
    else:
        badge = {"kind": "good", "label": "All figures found in the data"}
    return {"text": narrative.text, "html": md_to_html(narrative.text, narrative.unsupported), "badge": badge}


def profile(snapshot, team_id: str, season: int, team_seasons: pd.DataFrame | None = None,
            live: dict | None = None) -> dict:
    """One team-season: header, stats with "teams beaten", trends, model card and AI report (spec §5).
    `team_seasons` may add live rows to the history; `live` carries the live snapshot's status fields."""
    ds = snapshot.dataset
    every = ds.team_seasons if team_seasons is None else team_seasons
    row = _row(every, team_id, season)
    played = every[every["team_id"] == team_id].sort_values("season")
    live = live or {}
    return to_jsonable({
        **_header(row),
        "live": _is_live(row),
        "live_available": bool(live.get("available", False)),
        "stale": bool(live.get("stale", False)) if _is_live(row) else False,
        "fetched_at": live.get("fetched_at") if _is_live(row) else None,
        "matches_played": int(row["n_matches"]),
        "phase_matches": config.phase_matches(season),
        "provisional": _flag(row, "provisional"),
        "ko_stage": None if pd.isna(row["ko_stage"]) else int(row["ko_stage"]),
        "result": _result(ds.finals, team_id, season),
        "seasons": [{"season": int(s), "label": config.season_label(int(s))} for s in played["season"]],
        "features": [_stat(row, f) for f in ds.features],
        "trend": {f: [_point(r, f) for _, r in played.iterrows()] for f in ds.features},
        "model": _model_card(snapshot.results, row, ds.features),
        "narrative": _narrative(snapshot.analysis, f"{season}-{team_id}", snapshot.stale_keys),
    })


def parse_pick(text: str) -> tuple[str, int]:
    """'52280:2026' -> ('52280', 2026). Raises ValueError for anything else."""
    team_id, sep, season = (text or "").partition(":")
    if not sep or not team_id or not (season.isdecimal() and season.isascii()):
        raise ValueError(f"expected team:season, like 52280:2026, not {text!r}")
    return team_id, int(season)


def _oriented(row: pd.Series, feature: str) -> float | None:
    z = row.get(f"z_{feature}")
    if z is None or pd.isna(z):
        return None
    return -float(z) if feature in config.LOWER_IS_BETTER else float(z)


def ahead(a_z: float | None, b_z: float | None) -> str:
    if a_z is None or b_z is None or abs(a_z - b_z) < TIE_Z:
        return "tie"
    return "a" if a_z > b_z else "b"


def compare(snapshot, a: tuple[str, int], b: tuple[str, int], team_seasons: pd.DataFrame | None = None) -> dict:
    """Two team-seasons stat by stat. a_z/b_z are oriented: flipped for lower-is-better stats, so positive is
    always better and the UI draws them as given (spec §5)."""
    ds = snapshot.dataset
    every = ds.team_seasons if team_seasons is None else team_seasons
    row_a, row_b = _row(every, *a), _row(every, *b)
    rows = []
    for feature in ds.features:
        a_z, b_z = _oriented(row_a, feature), _oriented(row_b, feature)
        rows.append({"feature": feature, "label": config.FEATURE_META[feature][0], "section": SECTION_OF[feature],
                     "a_value": value_of(row_a, feature), "b_value": value_of(row_b, feature), "a_z": a_z,
                     "b_z": b_z, "a_beats": beats_of(row_a, feature, "season"),
                     "b_beats": beats_of(row_b, feature, "season"), "ahead": ahead(a_z, b_z)})
    return to_jsonable({"a": {**_header(row_a), "live": _is_live(row_a)},
                        "b": {**_header(row_b), "live": _is_live(row_b)}, "rows": rows})


def summary(snapshot) -> dict:
    """The overview: headline metrics with 95% intervals, the top drivers and the AI summary."""
    metrics = snapshot.results.metrics

    def metric(key: str) -> dict:
        return {"value": metrics[key], "ci": list(metrics["ci"][key])}

    drivers = snapshot.results.drivers
    top = drivers[drivers["rank"] <= config.TOP_DRIVERS].sort_values("rank")
    return to_jsonable({
        "metrics": {"spearman": metric("spearman_mean"), "auc": metric("auc"), "brier_skill": metric("brier_skill"),
                    "top4_share": {**metric("finalists_in_top4"), "chance": metrics["finalists_in_top4_chance"]}},
        "drivers": [{"feature": d.feature, "label": d.label_text, "group": d.group, "importance": d.importance,
                     "kind": d.label or None, "helps": HELPS.get(int(d.direction))} for d in top.itertuples()],
        "synthesis": _narrative(snapshot.analysis, "synthesis", snapshot.stale_keys),
    })
