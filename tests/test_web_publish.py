import json

import pytest

from ucl import analyst, config, dataset, model
from ucl.analyst import Analysis, Narrative
from ucl.facts import build_facts
from ucl.web import checks, publish, queries
from ucl.web.events import EventBus
from ucl.web.pipeline import PipelineRunner, StageOutcome
from ucl.web.services import Services
from ucl.web.store import DataStore


@pytest.fixture(scope="module")
def outputs(tmp_path_factory, built):
    ds, results = built
    root = tmp_path_factory.mktemp("publish")
    processed, out = root / "processed", root / "out"
    dataset.save(ds, processed)
    model.save(results, out)
    loaded = dataset.load(processed)
    sheets = build_facts(loaded.team_seasons, loaded.finals, model.load(out))
    narratives = {k: Narrative(k, "ok", text="## How they got there\nThey beat **everyone**.") for k in sheets}
    analyst.save(Analysis("ok", "m", narratives, sheets), out / "analysis.json")
    return processed, out


def services_for(processed, out, stages=None):
    bus = EventBus(boot="b")
    stages = stages or {"live": lambda ctx: (ctx.log("fetched 36 teams"), StageOutcome("done", "36 teams"))[1]}
    return Services(DataStore(processed, out), bus, PipelineRunner(stages, bus))


def test_players_are_sharded_by_the_last_two_characters_of_their_id():
    assert publish.player_shard("250010802") == "02"
    assert publish.player_shard("7") == "07"


def test_a_comparison_is_two_profiles_side_by_side(outputs):
    """The website builds comparisons from two profiles (web/src/lib/compare.ts): z flipped for lower-is-better
    stats, value and beats_season as they are. This pins that contract to queries.compare."""
    snapshot = DataStore(*outputs).snapshot
    rows = snapshot.dataset.team_seasons
    picks = list(zip(rows["team_id"], rows["season"].astype(int)))[:6]
    for a, b in zip(picks, reversed(picks)):
        compared = queries.compare(snapshot, a, b)
        pa, pb = queries.profile(snapshot, *a), queries.profile(snapshot, *b)
        stats_b = {stat["feature"]: stat for stat in pb["features"]}

        def oriented(stat):
            z = stat["z"]
            return None if z is None else -z if stat["feature"] in config.LOWER_IS_BETTER else z

        expected = [{"feature": sa["feature"], "label": sa["label"], "section": sa["section"],
                     "a_value": sa["value"], "b_value": stats_b[sa["feature"]]["value"],
                     "a_z": oriented(sa), "b_z": oriented(stats_b[sa["feature"]]),
                     "a_beats": sa["beats_season"], "b_beats": stats_b[sa["feature"]]["beats_season"],
                     "ahead": queries.ahead(oriented(sa), oriented(stats_b[sa["feature"]]))}
                    for sa in pa["features"]]
        assert compared["rows"] == expected
        side = ("team_id", "name", "season", "label", "stage_label", "live")
        assert compared["a"] == {key: pa[key] for key in side} and compared["b"] == {key: pb[key] for key in side}


def test_run_stages_records_the_run_for_the_pipeline_screen(outputs):
    services = services_for(*outputs)
    lines = []
    record = publish.run_stages(services, ["live"], lines.append)
    assert record["state"]["running"] is False
    types = [event["type"] for event in record["events"]]
    assert "done" in types and types.count("log") >= 1
    assert "  live: fetched 36 teams" in lines and "live: done (36 teams)" in lines


def test_run_stages_turns_an_unknown_stage_into_a_publish_error(outputs):
    with pytest.raises(publish.PublishError):
        publish.run_stages(services_for(*outputs), ["nope"], print)


def test_export_writes_every_view_the_site_reads(outputs, tmp_path, monkeypatch):
    asked = {}
    monkeypatch.setattr(checks, "run_checks", lambda services, model, web_build, public: asked.update(public=public) or {
        "checked_at": "2026-10-04T05:00:00+00:00", "model": model, "provider": "openrouter",
        "summary": {"ok": 1, "warn": 0, "fail": 0, "info": 0}, "checks": []})
    services = services_for(*outputs)
    record = {"state": services.runner.state(), "events": []}
    counts = publish.export(services, tmp_path, record, print)

    every = services.store.snapshot.dataset.team_seasons
    assert counts["profiles"] == len(every) and counts["squads"] == counts["players"] == 0
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["ready"] is True
    team_id, season = str(every.iloc[0]["team_id"]), int(every.iloc[0]["season"])
    profile = json.loads((tmp_path / "profiles" / team_id / f"{season}.json").read_text())
    assert (profile["team_id"], profile["season"]) == (team_id, season)
    teams = json.loads((tmp_path / "teams.json").read_text())
    assert {team["team_id"] for team in teams} == set(every["team_id"].astype(str))
    # no forecast service here: the files carry the error the API would have sent
    assert json.loads((tmp_path / "forecast.json").read_text())["status"] == 503
    assert json.loads((tmp_path / "matchups.json").read_text())["error"]["code"] == "forecast_unavailable"
    assert json.loads((tmp_path / "checks.json").read_text())["provider"] == "openrouter"
    assert asked == {"public": True}  # the website's checks leave out what the key has spent
    pipeline = json.loads((tmp_path / "pipeline.json").read_text())
    assert set(pipeline) == {"published_at", "state", "events"}


def test_export_needs_the_pipelines_outputs(tmp_path):
    services = services_for(tmp_path / "processed", tmp_path / "out")
    with pytest.raises(publish.PublishError, match="run the pipeline first"):
        publish.export(services, tmp_path / "data", {"state": {}, "events": []}, print)


def test_publish_needs_the_website_build(tmp_path):
    with pytest.raises(publish.PublishError, match="npm run build:site"):
        publish.publish(tmp_path / "output", site_build=config.ROOT / "web" / "no-such-build")
