from types import SimpleNamespace

import pandas as pd
import pytest
from test_forecast import league_fixtures, match, frame

from ucl import forecast
from ucl.web import forecasts
from ucl.web.forecasts import ForecastService

TRACK_RECORD = {"seasons": ["2019-20", "2025-26"], "matches": {"log_loss": 0.92}, "ties": {"ties": 1},
                "titles": pd.DataFrame([{"season": 2026, "winner_id": "h0", "p_winner": 0.2, "winner_rank": 1,
                                         "favourite_id": "h0", "p_favourite": 0.2, "teams_left": 24}])}


@pytest.fixture(autouse=True)
def quick_track_record(monkeypatch):
    calls = []
    monkeypatch.setattr(forecast, "track_record", lambda fixtures, priors, n: calls.append(n) or TRACK_RECORD)
    return calls


def history(seasons):
    """Two past seasons between four clubs."""
    rows = []
    for season in (2025, 2026):
        rows += [match(season, 0, "h0", "h1", (2, 0)), match(season, 0, "h2", "h3", (1, 1))]
    return frame(rows)


def snapshot(built_at=1.0):
    team_seasons = pd.DataFrame({"season": [2025, 2025, 2025, 2025, 2026, 2026, 2026, 2026],
                                 "team_id": ["h0", "h1", "h2", "h3"] * 2, "z_coef_log": [1.0, -1.0, 0.5, -0.5] * 2,
                                 "team_display": ["Club 0", "Club 1", "Club 2", "Club 3"] * 2})
    return SimpleNamespace(dataset=SimpleNamespace(team_seasons=team_seasons), built_at=built_at)


def live(built_at=10.0, status="ready", played=None):
    fixtures, ids = league_fixtures(played=played)
    rows = pd.DataFrame({"team_id": ids, "team_display": [f"Live {t}" for t in ids],
                         "coef_log": [4.0 - i / 10 for i in range(len(ids))]})
    snap = SimpleNamespace(season=2027, built_at=built_at, rows=rows, fixtures=fixtures, fetched_at=None)
    return SimpleNamespace(snapshot=snap, status=status, message="")


def service(loads):
    return ForecastService(lambda seasons: loads.append(seasons) or history(seasons), simulations=400,
                           record_simulations=10)


def test_title_odds_need_the_dataset_and_the_live_season():
    loads = []
    with pytest.raises(forecasts.Unavailable, match="dataset"):
        service(loads).title_odds(SimpleNamespace(dataset=None, built_at=None), None)
    with pytest.raises(forecasts.Unavailable, match="live season isn't loaded"):
        service(loads).title_odds(snapshot(), SimpleNamespace(snapshot=None, status="unavailable",
                                                              message="UEFA down"))


def test_title_odds_rank_the_live_field_and_report_the_track_record(quick_track_record):
    loads = []
    svc = service(loads)
    played = {("t00", "t01"): (3, 0)}
    view = svc.title_odds(snapshot(), live(played=played))
    teams = view["teams"]
    assert (view["label"], view["played"], view["league_matches"], view["simulations"]) == ("2026-27", 1, 144, 400)
    assert view["as_of"] == "2026-09-01T20:00:00Z" and len(teams) == 36
    assert sum(t["p_win"] for t in teams) == pytest.approx(1) and teams[0]["name"].startswith("Live t0")
    t00 = next(t for t in teams if t["team_id"] == "t00")
    assert (t00["played"], t00["points"], t00["goal_diff"]) == (1, 3, 3)
    assert view["track_record"]["titles"][0] == {**TRACK_RECORD["titles"].iloc[0].to_dict(), "label": "2025-26",
                                                  "winner": "Club 0", "favourite": "Club 0"}
    svc.title_odds(snapshot(), live(played=played))
    assert len(loads) == 1 and quick_track_record == [10]  # both reused while nothing changed


def test_a_new_live_snapshot_rebuilds_the_odds_but_not_the_history():
    loads = []
    svc = service(loads)
    first = svc.title_odds(snapshot(), live(built_at=10.0))
    second = svc.title_odds(snapshot(), live(built_at=11.0, played={("t35", "t00"): (5, 0)}))
    assert len(loads) == 1 and first["played"] == 0 and second["played"] == 1
    svc.title_odds(snapshot(built_at=2.0), live(built_at=11.0))
    assert len(loads) == 2  # a rebuilt dataset reloads the historical fixtures


def test_head_to_head_rates_any_two_team_seasons():
    svc = service([])
    view = svc.head_to_head(snapshot(), live(), ("h0", 2026), ("t05", 2027), "neutral")
    assert view["a"]["name"] == "Club 0" and view["a"]["title"] is None and not view["a"]["live"]
    assert view["b"]["live"] and 0 < view["b"]["title"] < 1 and view["title_label"] == "2026-27"
    assert view["win"] + view["draw"] + view["loss"] == pytest.approx(1)
    with pytest.raises(forecasts.NotFound, match="no rating for team zz"):
        svc.head_to_head(snapshot(), live(), ("zz", 2026), ("h0", 2026), "neutral")
    with pytest.raises(ValueError, match="venue"):
        svc.head_to_head(snapshot(), live(), ("h0", 2026), ("h1", 2026), "moon")
    offline = svc.head_to_head(snapshot(), None, ("h0", 2025), ("h1", 2026), "a")  # past seasons need no live data
    assert offline["title_label"] is None and offline["venue"] == "a"
