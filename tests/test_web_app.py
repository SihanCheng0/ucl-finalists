import threading

import pytest
from fastapi.testclient import TestClient

from ucl import analyst, dataset, model
from ucl.analyst import Analysis, Narrative
from ucl.facts import build_facts
from ucl.web.app import create_app
from ucl.web.events import EventBus
from ucl.web.pipeline import CORE, PipelineRunner, StageOutcome
from ucl.web.services import Services
from ucl.web.store import DataStore

def local(app):
    """A client that talks to the app as a browser on this machine does (the app only answers to local hosts)."""
    return TestClient(app, base_url="http://127.0.0.1")


NOT_READY = {"error": {"code": "not_ready",
                       "message": "Run the pipeline first: processed data or model outputs are missing."}}


@pytest.fixture(scope="module")
def outputs(tmp_path_factory, built):
    ds, results = built
    root = tmp_path_factory.mktemp("app")
    processed, out = root / "processed", root / "out"
    dataset.save(ds, processed)
    model.save(results, out)
    loaded = dataset.load(processed)
    sheets = build_facts(loaded.team_seasons, loaded.finals, model.load(out))
    narratives = {k: Narrative(k, "ok", text="## How they got there\nThey beat **everyone**.") for k in sheets}
    analyst.save(Analysis("ok", "m", narratives, sheets), out / "analysis.json")
    return processed, out


def make_services(processed, out, gate=None):
    """Real store and bus; fake stages that wait on `gate`, so a run can be held open."""
    gate = gate or threading.Event()

    def stage(ctx):
        gate.wait(5)
        return StageOutcome("done", "ok")

    def missing(names):
        return "model needs data/processed/dataset.json: run build first" if names == ["model"] else None

    bus = EventBus(boot="b")
    return Services(DataStore(processed, out), bus,
                    PipelineRunner({name: stage for name in CORE}, bus, missing_inputs=missing))


@pytest.fixture
def api(outputs):
    gate = threading.Event()
    services = make_services(*outputs, gate=gate)
    with local(create_app(services, heartbeat=0.05)) as client:
        yield client, services, gate
    gate.set()
    services.runner.wait(5)


def error(response):
    return response.status_code, response.json()["error"]["code"]


def test_meta_reports_the_loaded_data(api):
    client, _, _ = api
    meta = client.get("/api/meta").json()
    assert meta["ready"] is True and meta["seasons"][-1] == {"season": 2026, "label": "2025-26", "live": False}
    assert len(meta["features"]) == 16 and meta["data"]["modelled_at"]


def test_data_routes_answer_503_until_the_pipeline_has_run(tmp_path):
    services = make_services(tmp_path / "processed", tmp_path / "out")
    with local(create_app(services)) as client:
        assert client.get("/api/meta").json()["ready"] is False
        for path in ("/api/summary", "/api/teams?q=team", "/api/teams/52280/seasons/2026",
                     "/api/compare?a=1:2026&b=2:2026"):
            response = client.get(path)
            assert (response.status_code, response.json()) == (503, NOT_READY)
        assert client.get("/api/pipeline/state").status_code == 200


def test_summary_and_search(api):
    client, _, _ = api
    assert client.get("/api/summary").json()["synthesis"]["badge"]["kind"] == "good"
    assert client.get("/api/teams", params={"q": "t"}).json() == []
    hits = client.get("/api/teams", params={"q": "team 5228"}).json()
    assert hits[0]["team_id"] == "52280" and hits[0]["seasons"][0]["season"] == 2026


def test_profile_and_its_errors(api):
    client, _, _ = api
    profile = client.get("/api/teams/52280/seasons/2026").json()
    assert profile["result"].startswith("Runner-up") and "<strong>everyone</strong>" in profile["narrative"]["html"]
    assert error(client.get("/api/teams/52280/seasons/2012")) == (404, "not_found")
    assert error(client.get("/api/teams/52280/seasons/abc")) == (422, "invalid_request")


def test_compare_and_its_errors(api):
    client, _, _ = api
    out = client.get("/api/compare", params={"a": "52280:2026", "b": "52747:2026"}).json()
    assert out["a"]["team_id"] == "52280" and {r["ahead"] for r in out["rows"]} <= {"a", "b", "tie"}
    assert error(client.get("/api/compare", params={"a": "52280", "b": "52747:2026"})) == (422, "invalid_request")
    assert error(client.get("/api/compare", params={"a": "52280:2026", "b": "nope:2026"})) == (404, "not_found")


def test_a_run_starts_once_and_bad_runs_are_refused(api):
    client, services, gate = api
    response = client.post("/api/pipeline/runs", json={"stages": ["fetch"]})
    assert (response.status_code, response.json()) == (202, {"run_id": "b-r1"})
    assert error(client.post("/api/pipeline/runs", json={})) == (409, "run_in_progress")
    state = client.get("/api/pipeline/state").json()
    assert (state["running"], state["run_id"], state["boot"]) == (True, "b-r1", "b")
    gate.set()
    assert services.runner.wait(5)
    assert error(client.post("/api/pipeline/runs", json={"stages": ["model"]})) == (422, "bad_run")
    assert error(client.post("/api/pipeline/runs", json={"stages": ["train"]})) == (422, "bad_run")
    assert error(client.post("/api/pipeline/runs", json={"refresh_live": True})) == (422, "bad_run")  # no live stage
    assert error(client.post("/api/pipeline/runs", json={"skip_ai": "sometimes"})) == (422, "invalid_request")
    assert client.post("/api/pipeline/runs", json={}).status_code == 202  # an empty body: every core stage


def test_events_stream_the_backlog_and_resume_after_the_last_event_id(api):
    client, services, _ = api
    services.bus.publish("log", {"level": "info", "text": "one"})
    services.bus.publish("log", {"level": "info", "text": "two"})
    services.bus.close()  # the stream then ends after the backlog, so each response completes
    response = client.get("/api/pipeline/events")
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert [line for line in response.text.splitlines() if line.startswith("id: ")] == ["id: b-1", "id: b-2"]
    resumed = client.get("/api/pipeline/events", headers={"Last-Event-ID": "b-1"})
    assert [line for line in resumed.text.splitlines() if line.startswith("id: ")] == ["id: b-2"]


def test_shutdown_closes_the_event_bus(outputs):
    services = make_services(*outputs)
    with local(create_app(services)):
        assert not services.bus.closed
    assert services.bus.closed


def test_the_built_spa_is_served_at_the_root(outputs, tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>UCL Lab</title>")
    with local(create_app(make_services(*outputs), dist_dir=dist)) as client:
        page = client.get("/")
        assert "UCL Lab" in page.text and page.headers["cache-control"] == "no-cache"
        assert client.get("/api/meta").json()["ready"] is True  # API routes win over the static mount


# --- the live season, squads and players (Plan B) ---

import socket  # noqa: E402

import pandas as pd  # noqa: E402

from ucl.web.live import LiveSnapshot, LiveState  # noqa: E402
from ucl.web.players import STAT_META, SquadUnavailable  # noqa: E402


class FakeLive:
    def __init__(self, rows, status="ready"):
        self.state = LiveState(LiveSnapshot(2027, rows, status == "stale", None, 1, 0.0), status)
        self.revalidations = 0

    def current(self):
        return self.state

    def refresh_in_background(self):
        self.revalidations += 1
        return False


class FakePlayers:
    def squad(self, team_id, season):
        if season == 2025:
            raise SquadUnavailable("UEFA couldn't be reached: offline")
        return {"players": [{"player_id": "1", "name": "Bukayo Saka", "position": "FWD", "minutes": 90.0,
                             "stats": {"goals": 1.0}}], "minutes_published": True, "stale": False, "fetched_at": None}

    def freshness(self, season, team_id):
        return False, None

    def history(self, player_id):
        if player_id == "nobody":
            return None
        return {"player": {"player_id": "1", "name": "Bukayo Saka"}, "unavailable_seasons": [],
                "index": {"complete": False, "squads": {"done": 3, "of": 524}, "running": False},
                "history": [{"season": 2027, "label": "2026-27", "team_id": "52280", "team": "Arsenal",
                             "minutes": 90.0, "stats": {"goals": 1.0}, "live": True}]}


@pytest.fixture
def live_api(outputs):
    services = make_services(*outputs)
    history = services.store.snapshot.dataset.team_seasons
    row = history[(history["team_id"] == "52280") & (history["season"] == 2026)].copy()
    row["season"], row["n_matches"], row["ko_stage"] = 2027, 1, float("nan")
    row["stage_label"], row["live"], row["provisional"] = "League phase (in progress)", True, True
    services.live, services.players = FakeLive(row.reset_index(drop=True)), FakePlayers()
    with local(create_app(services)) as client:
        yield client, services


def test_meta_lists_the_live_season_and_the_player_stats(live_api):
    client, services = live_api
    meta = client.get("/api/meta").json()
    assert meta["seasons"][-1] == {"season": 2027, "label": "2026-27", "live": True}
    assert (meta["live_season"], meta["live_status"]) == (2027, "ready")
    assert [s["key"] for s in meta["player_stats"]] == [s["key"] for s in STAT_META]
    assert meta["sections"][0] == "Results" and services.live.revalidations == 1


def test_a_live_team_season_is_searchable_profiled_and_comparable(live_api):
    client, _ = live_api
    hit = client.get("/api/teams", params={"q": "team 52280"}).json()[0]
    assert hit["seasons"][0] == {"season": 2027, "label": "2026-27", "stage_label": "League phase (in progress)"}
    p = client.get("/api/teams/52280/seasons/2027").json()
    assert (p["live"], p["provisional"], p["matches_played"], p["phase_matches"]) == (True, True, 1, 8)
    assert p["model"] is None and p["narrative"] is None and p["live_available"] is True
    assert [point["live"] for point in p["trend"]["points_pg"]] == [False, True]
    out = client.get("/api/compare", params={"a": "52280:2027", "b": "52747:2026"}).json()
    assert out["a"]["live"] is True and out["b"]["live"] is False


def test_squads_and_their_errors(live_api):
    client, _ = live_api
    squad = client.get("/api/squads/52280/2027").json()
    assert (squad["name"], squad["label"], squad["live"], squad["minutes_published"]) == (
        "Team 52280", "2026-27", True, True)
    assert squad["players"][0]["name"] == "Bukayo Saka"
    assert error(client.get("/api/squads/52280/2012")) == (404, "not_found")
    assert error(client.get("/api/squads/52747/2025")) == (503, "uefa_unreachable")


def test_player_histories(live_api):
    client, _ = live_api
    player = client.get("/api/players/1").json()
    assert player["history"][0]["team"] == "Arsenal" and player["index"]["squads"] == {"done": 3, "of": 524}
    assert (player["stale"], player["fetched_at"]) == (False, None)
    assert error(client.get("/api/players/nobody")) == (404, "not_found")


def test_every_error_has_the_same_shape(api):
    client, _, _ = api
    assert error(client.get("/api/nope")) == (404, "not_found")
    assert error(client.post("/api/meta")) == (405, "method_not_allowed")
    assert error(client.post("/api/pipeline/runs", json={"stage": ["fetch"]})) == (422, "invalid_request")
    assert error(client.get("/api/squads/52280/2026")) == (503, "players_unavailable")


def test_ctrl_c_with_an_event_stream_open_stops_the_server_at_once(outputs):
    import threading
    import time

    from ucl.web.server import make_server, uvicorn_config

    services = make_services(*outputs)
    server = make_server(uvicorn_config(create_app(services), 0), services.bus)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]
    stream = socket.create_connection(("127.0.0.1", port))
    stream.sendall(b"GET /api/pipeline/events HTTP/1.1\r\nHost: localhost\r\n\r\n")
    assert b"text/event-stream" in stream.recv(4096)
    began = time.monotonic()
    server.should_exit = True
    thread.join(5)
    stream.close()
    assert not thread.is_alive() and time.monotonic() - began < 1.5


def test_another_site_cannot_start_a_run_or_reach_the_api(outputs):
    services = make_services(*outputs)
    services.runner._stages = {}  # nothing may run
    with local(create_app(services)) as client:
        assert error(client.post("/api/pipeline/runs")) == (422, "invalid_request")  # no JSON body, no run
        form = client.post("/api/pipeline/runs", data={"stages": "fetch"}, headers={"Origin": "https://evil.test"})
        assert form.status_code == 422
    with TestClient(create_app(services), base_url="http://evil.test") as rebound:
        assert rebound.get("/api/meta").status_code == 400  # DNS rebinding: a foreign Host header is refused
    assert services.runner.state()["run_id"] is None
