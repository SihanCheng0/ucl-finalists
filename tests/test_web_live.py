import threading

import numpy as np
import pandas as pd
import pytest

from ucl import config
from ucl.uefa import Fetched
from ucl.web.live import LiveService, LiveSnapshot, build_snapshot, live_frame, rank_among

FEATURES = ["points_pg", "goal_diff_pg", "possession_pct", "coef_log"]


def team(team_id):
    return {"id": team_id, "internationalName": f"Team {team_id}"}


def match(match_id, home, away, status="FINISHED", score=(1, 0), full_time="2026-09-16T21:00:00Z"):
    m = {"id": match_id, "round": {"metaData": {"name": "League Phase"}}, "homeTeam": team(home),
         "awayTeam": team(away), "status": status}
    if status == "FINISHED":
        m["score"] = {"total": {"home": score[0], "away": score[1]}}
        m["winner"] = {"match": {"team": {"id": home if score[0] > score[1] else away}}} if score[0] != score[1] else {}
        m["fullTimeAt"] = full_time
    return m


def stats(possession_home, possession_away):
    return [{"teamId": t, "statistics": [{"name": "ball_possession", "value": str(p)}]}
            for t, p in (("1", possession_home), ("2", possession_away))]


RAW = [match("m1", "1", "2", score=(2, 0)), match("m2", "3", "4", status="UPCOMING")]
STATS = {"m1": [{"teamId": "1", "statistics": [{"name": "goals", "value": "2"},
                                                {"name": "ball_possession", "value": "61"}]},
                {"teamId": "2", "statistics": [{"name": "goals", "value": "0"},
                                                {"name": "ball_possession", "value": "39"}]}]}
COEFS = [{"member": {"id": t}, "overallRanking": {"totalValue": v}} for t, v in (("1", 90.0), ("2", 30.0))]
HISTORY = pd.DataFrame({"points_pg": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], "goal_diff_pg": [0.0] * 6,
                        "possession_pct": [40.0, 45.0, 50.0, 55.0, 60.0, 65.0], "coef_log": [3.0] * 6})


def test_rank_among_matches_add_percentiles_average_rank():
    history = pd.Series([1.0, 2.0, 2.0, 3.0, np.nan])
    assert rank_among(history, 2.0) == round(100 * 3 / 5)  # ranks 2, 3, 4 tie: average 3 of 5
    assert rank_among(history, 9.0) == 100.0 and rank_among(history, None) is None


def test_the_field_includes_teams_that_have_not_played_yet():
    rows = live_frame(RAW, STATS, COEFS, HISTORY, FEATURES, season=2027).set_index("team_id")
    assert sorted(rows.index) == ["1", "2", "3", "4"]
    assert (rows.loc["3", "n_matches"], rows.loc["3", "provisional"]) == (0, True)
    assert pd.isna(rows.loc["3", "points_pg"]) and pd.isna(rows.loc["3", "pct_all_points_pg"])
    assert rows["live"].all() and (rows["stage_label"] == "League phase (in progress)").all()


def test_features_come_from_finished_matches_ranked_within_the_season_and_against_history():
    rows = live_frame(RAW, STATS, COEFS, HISTORY, FEATURES, season=2027).set_index("team_id")
    assert (rows.loc["1", "points_pg"], rows.loc["1", "goal_diff_pg"]) == (3.0, 2.0)
    assert rows.loc["1", "pct_season_points_pg"] == 100 and rows.loc["2", "pct_season_points_pg"] == 50
    assert rows.loc["1", "pct_all_points_pg"] == rank_among(HISTORY["points_pg"], 3.0)
    assert rows.loc["1", "z_points_pg"] > 0 > rows.loc["2", "z_points_pg"]
    assert rows.loc["1", "coef"] == 90.0


def test_no_finished_match_gives_an_all_null_field_not_an_error():
    upcoming = [match("m1", "1", "2", status="UPCOMING"), match("m2", "3", "4", status="UPCOMING")]
    rows = live_frame(upcoming, {}, COEFS, HISTORY, FEATURES, season=2027)
    assert len(rows) == 4 and rows["points_pg"].isna().all() and (rows["n_matches"] == 0).all()


class FakeClient:
    def __init__(self, stale=False):
        self.stale, self.ages = stale, {}

    def matches_fresh(self, season, max_age):
        self.match_age = max_age
        return Fetched(RAW, self.stale, None)

    def team_match_stats_fresh(self, match_id, max_age, cache_missing=False):
        self.ages[match_id] = max_age
        return Fetched(STATS[match_id], False, None)

    def coefficients(self, season):
        return COEFS


def test_build_snapshot_keeps_settled_copies_and_refetches_recent_ones():
    client = FakeClient()
    after_a_week = pd.Timestamp("2026-09-23T21:00:00Z").timestamp()
    snapshot = build_snapshot(client, HISTORY, FEATURES, season=2027, clock=lambda: after_a_week)
    # settled six days ago: a copy written before that (while UEFA was still completing it) is fetched again
    assert client.ages == {"m1": 6 * 86400} and snapshot.finished_matches == 1 and not snapshot.stale
    client = FakeClient(stale=True)
    an_hour_later = pd.Timestamp("2026-09-16T22:00:00Z").timestamp()
    snapshot = build_snapshot(client, HISTORY, FEATURES, force=True, season=2027, clock=lambda: an_hour_later)
    assert client.ages == {"m1": 0} and client.match_age == 0 and snapshot.stale


def snap(stale=False, built_at=0.0):
    return LiveSnapshot(2027, pd.DataFrame(), stale, None, 1, built_at)


def test_the_service_starts_loading_and_becomes_ready():
    service = LiveService(lambda force: snap(), clock=lambda: 0.0)
    assert service.current().status == "loading" and not service.current().available
    assert service.refresh().status == "ready" and service.current().available


def test_a_failed_refresh_keeps_the_last_snapshot_and_backs_off():
    now, outcomes = [0.0], [snap(), RuntimeError("UEFA down")]

    def build(force):
        outcome = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    service = LiveService(build, clock=lambda: now[0], max_age=100, retry_after=50)
    service.refresh()
    now[0] = 200  # older than max_age: due
    state = service.refresh()
    assert (state.status, state.available, state.message) == ("stale", True, "RuntimeError: UEFA down")
    now[0] = 220
    assert not service.due()  # inside the 50 s back-off
    now[0] = 260
    assert service.due()


def test_without_any_snapshot_a_failure_is_unavailable():
    def build(force):
        raise RuntimeError("the historical dataset isn't built yet: run build first")

    state = LiveService(build, clock=lambda: 0.0).refresh()
    assert (state.status, state.available) == ("unavailable", False)


def test_refreshes_are_single_flight():
    calls, gate = [], threading.Event()

    def build(force):
        calls.append(force)
        gate.wait(5)
        return snap(built_at=0.0)

    service = LiveService(build, clock=lambda: 0.0)
    assert service.refresh_in_background()
    assert not service.refresh_in_background()  # one is already running
    gate.set()
    service.refresh()  # waits for the running one, then finds the snapshot fresh
    assert calls == [False]
    assert service.refresh(force=True).status == "ready" and calls == [False, True]


def test_a_snapshot_built_from_stale_copies_is_retried_after_the_back_off():
    now = [0.0]
    service = LiveService(lambda force: snap(stale=True, built_at=now[0]), clock=lambda: now[0], max_age=3600,
                          retry_after=300)
    assert service.refresh().status == "stale"
    now[0] = 200
    assert not service.due()
    now[0] = 301
    assert service.due()  # not six hours later


def test_a_match_stats_copy_written_before_settling_is_fetched_once_more(tmp_path):
    import json
    import os

    from ucl.uefa import UefaClient

    url = config.MATCH_STATS_URL.format(match_id="m1")
    calls = []

    def fetch(u, timeout):
        calls.append(u)
        return STATS["m1"]

    cache = tmp_path / "stats"
    cache.mkdir()
    early = cache / "m1.json"
    early.write_text(json.dumps([{"teamId": "1", "statistics": []}]))  # an early, incomplete copy
    full_time = pd.Timestamp("2026-09-16T21:00:00Z").timestamp()
    os.utime(early, (full_time + 3600, full_time + 3600))
    client = UefaClient(cache_dir=tmp_path, fetch=fetch, sleep=lambda _: None)
    later = full_time + 30 * 3600
    assert client.team_match_stats_fresh("m1", later - (full_time + config.LIVE_SETTLED_S)).data == STATS["m1"]
    assert calls == [url]


def test_a_service_without_background_refreshes_never_starts_one():
    """The website export: requests read what the stages fetched and start nothing behind its back."""
    calls = []
    service = LiveService(lambda force: calls.append(force) or snap(built_at=0.0), clock=lambda: 0.0,
                          background=False)
    assert service.refresh_in_background() is False and calls == []
    assert service.refresh().status == "ready" and calls == [False]  # an explicit refresh still works
