import json
import time

import pytest

from ucl import config
from ucl.uefa import Fetched
from ucl.web.players import PlayerService, SquadUnavailable, parse_squad


def row(player_id, name, minutes=None, position="MIDFIELDER", team="Arsenal", **stats):
    values = dict(stats)
    if minutes is not None:
        values["minutes_played_official"] = minutes
    return {"playerId": player_id, "teamId": "52280", "team": {"internationalName": team},
            "player": {"id": player_id, "internationalName": name, "fieldPosition": position,
                       "clubJerseyNumber": "7", "age": "24", "imageUrl": f"https://img/{player_id}.jpg"},
            "statistics": [{"name": k, "value": str(v)} for k, v in values.items()]}


SQUAD = [
    row("1", "Bukayo Saka", 1290, "FORWARD", goals=4, distance_covered=150.2, top_speed=34.1, passes_attempted=400,
        passes_completed=340),
    row("2", "Kai Havertz", 600, goals=0),  # a count the feed reports, as zero
    row("3", "Third Keeper", None, "GOALKEEPER", yellow_cards=0),  # in the squad, never played
    row("4", "Old Style", 900, None, distance_covered=11800.0, top_speed=99.0),  # metres and a bad speed
]


def test_parse_squad_orders_by_minutes_maps_positions_and_fills_omitted_zero_counts():
    players = parse_squad(SQUAD)
    assert [p["name"] for p in players] == ["Bukayo Saka", "Old Style", "Kai Havertz", "Third Keeper"]
    saka, old, havertz, keeper = players
    assert (saka["position"], saka["minutes"], saka["shirt"], saka["age"]) == ("FWD", 1290, "7", 24.0)
    assert havertz["stats"]["goals"] == 0 and old["stats"]["goals"] == 0  # reported for others, so 0, not unknown
    assert saka["stats"]["key_passes"] is None  # never reported in this squad: unknown
    assert keeper["minutes"] == 0 and keeper["stats"]["goals"] is None  # didn't play: nothing filled in
    assert saka["stats"]["passes_accuracy"] == 85.0  # worked out from passes when the feed omits it
    assert old["position"] is None


def test_implausible_values_become_null():
    old = next(p for p in parse_squad(SQUAD) if p["name"] == "Old Style")
    assert old["stats"]["distance_covered"] is None and old["stats"]["top_speed"] is None
    saka = next(p for p in parse_squad(SQUAD) if p["name"] == "Bukayo Saka")
    assert saka["stats"]["distance_covered"] == 150.2  # 10.5 km per 90 is plausible


class FakeClient:
    def __init__(self, cache_dir, squads, fail=()):
        self.cache_dir, self.squads, self.fail, self.calls = cache_dir, squads, set(fail), []

    def squad(self, season, team_id, max_age=None):
        path = self.cache_dir / "players" / str(season) / f"{team_id}.json"
        if max_age is None and path.exists():
            return Fetched(json.loads(path.read_text()), False, None)
        self.calls.append((season, team_id, max_age))
        if (season, team_id) in self.fail:
            raise RuntimeError("UEFA down")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.squads[(season, team_id)]))
        return Fetched(self.squads[(season, team_id)], False, None)


SQUADS = {
    (2026, "52280"): [row("1", "Bukayo Saka", 1290, "FORWARD", goals=4)],
    (2025, "52280"): [row("1", "Bukayo Saka", 900, "FORWARD", goals=2)],
    (2024, "50124"): [row("1", "Bukayo Saka", 300, "FORWARD", team="Atleti", goals=1)],  # another club
    (2027, "52280"): [row("1", "Bukayo Saka", 90, "FORWARD", goals=1)],
}


def service(tmp_path, fail=(), pairs=None, clock=time.time):
    client = FakeClient(tmp_path, SQUADS, fail)
    every = pairs if pairs is not None else list(SQUADS)
    return PlayerService(client, client, lambda: every, cache_dir=tmp_path / "players", clock=clock), client


def test_a_squad_is_fetched_once_then_served_from_the_cache(tmp_path):
    players, client = service(tmp_path)
    first = players.squad("52280", 2026)
    assert first["players"][0]["name"] == "Bukayo Saka" and first["minutes_published"] and not first["stale"]
    players.squad("52280", 2026)
    assert client.calls == [(2026, "52280", None)]


def test_a_failed_fetch_is_remembered_for_a_minute(tmp_path):
    now = [1000.0]
    players, client = service(tmp_path, fail={(2026, "52280")}, clock=lambda: now[0])
    with pytest.raises(SquadUnavailable, match="UEFA couldn't be reached"):
        players.squad("52280", 2026)
    with pytest.raises(SquadUnavailable, match="a moment ago"):
        players.squad("52280", 2026)
    assert len(client.calls) == 1
    now[0] += config.REQUEST_FAILURE_MEMORY_S + 1
    with pytest.raises(SquadUnavailable, match="UEFA couldn't be reached"):
        players.squad("52280", 2026)
    assert len(client.calls) == 2


def test_history_spans_seasons_and_clubs_newest_first(tmp_path):
    players, _ = service(tmp_path)
    for season, team_id in SQUADS:
        players.squad(team_id, season)
    history = players.history("1")
    assert [(h["label"], h["team"]) for h in history["history"]] == [
        ("2026-27", "Arsenal"), ("2025-26", "Arsenal"), ("2024-25", "Arsenal"), ("2023-24", "Atleti")]
    assert history["history"][0]["live"] is True and history["history"][1]["stats"]["goals"] == 4
    assert history["player"]["name"] == "Bukayo Saka" and history["player"]["position"] == "FWD"
    assert players.history("nobody") is None


def test_the_index_is_rebuilt_from_the_cache_at_startup(tmp_path):
    players, _ = service(tmp_path)
    players.squad("50124", 2024)
    again, _ = service(tmp_path)
    assert [h["season"] for h in again.history("1")["history"]] == [2024]


def test_the_index_job_fetches_what_is_missing_reports_progress_and_failures(tmp_path):
    players, client = service(tmp_path, fail={(2025, "52280")})
    players.squad("52280", 2026)
    progress = []
    failures = players.build_index(lambda done, of: progress.append((done, of)), log=lambda line: None, workers=1)
    assert [pair for pair, _ in failures] == [(2025, "52280")]
    assert progress[0] == (1, 4) and progress[-1] == (3, 4)
    assert players.index_status() == {"complete": False, "squads": {"done": 3, "of": 4}, "running": False}
    assert players.history("1")["unavailable_seasons"] == ["2024-25"]
    assert (2027, "52280", 0) in client.calls  # the live season's squad is always refetched
