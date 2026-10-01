from types import SimpleNamespace

import pytest

from ucl import analyst, config, dataset, model, report, stages
from ucl.analyst import Analysis, Narrative

DS = SimpleNamespace(team_seasons="ts", finals="finals", features=["x"])


def test_fetch_passes_progress_only_when_given(monkeypatch):
    calls = []
    monkeypatch.setattr(dataset, "fetch_all", lambda client, seasons, **kwargs: calls.append(kwargs) or {"9": "down"})
    assert stages.fetch("client").failed == {"9": "down"}
    stages.fetch("client", progress=print, log=print)
    assert calls == [{"log": print}, {"log": print, "progress": print}]


def test_build_validates_before_saving(monkeypatch):
    saved = []
    monkeypatch.setattr(dataset, "build", lambda client, seasons: "ds")
    monkeypatch.setattr(dataset, "save", saved.append)

    def invalid(ds):
        raise dataset.ValidationError("2016 Roma: distance_km_pg = 3.2 is outside 90-140")

    monkeypatch.setattr(dataset, "validate", invalid)
    with pytest.raises(dataset.ValidationError):
        stages.build("client")
    assert saved == []
    monkeypatch.setattr(dataset, "validate", lambda ds: None)
    assert stages.build("client") == "ds" and saved == ["ds"]


def test_model_calls_run_positionally_and_passes_progress_only_when_given(monkeypatch):
    calls, saved = [], []
    monkeypatch.setattr(model, "run", lambda *args, **kwargs: calls.append((args, kwargs)) or "results")
    monkeypatch.setattr(model, "save", saved.append)
    assert stages.model(DS) == "results"
    stages.model(DS, progress=print)
    assert calls == [(("ts", "finals", ["x"]), {}), (("ts", "finals", ["x"]), {"progress": print})]
    assert saved == ["results", "results"]


def analysis_with(narratives, facts):
    return Analysis("ok", "m", dict(narratives), dict(facts))


def test_analyze_without_merge_saves_every_status_as_the_cli_always_has(monkeypatch):
    for status in ("ok", "unavailable", "skipped"):
        saved = []
        monkeypatch.setattr(analyst, "run", lambda *a, **k: Analysis(status, "m"))
        monkeypatch.setattr(analyst, "save", saved.append)
        result = stages.analyze(DS, "results", "m", True)
        assert saved == [result.analysis] and result.unavailable == []


def test_analyze_with_merge_never_saves_a_run_the_llm_could_not_do(monkeypatch):
    saved = []
    monkeypatch.setattr(analyst, "run", lambda *a, **k: Analysis("unavailable", "m", reason="server down"))
    monkeypatch.setattr(analyst, "save", saved.append)
    result = stages.analyze(DS, "results", "m", True, merge=True)
    assert saved == [] and result.analysis.reason == "server down"


def test_analyze_with_merge_keeps_old_text_and_its_facts_for_narratives_it_could_not_write(monkeypatch):
    old = analysis_with({"a": Narrative("a", "ok", text="old a"), "b": Narrative("b", "ok", text="old b")},
                        {"a": {"x": 1}, "b": {"x": 2}})
    new = analysis_with({"a": Narrative("a", "unavailable", reason="empty answer"),
                         "b": Narrative("b", "ok", text="new b"),
                         "c": Narrative("c", "unavailable", reason="empty answer")},
                        {"a": {"x": 10}, "b": {"x": 20}, "c": {"x": 30}})
    saved = []
    monkeypatch.setattr(analyst, "run", lambda *a, **k: new)
    monkeypatch.setattr(analyst, "load", lambda: old)
    monkeypatch.setattr(analyst, "save", saved.append)
    result = stages.analyze(DS, "results", "m", True, merge=True)
    assert result.unavailable == ["a", "c"]
    merged = saved[0]
    assert (merged.narratives["a"].text, merged.facts["a"]) == ("old a", {"x": 1})  # flagged stale later
    assert (merged.narratives["b"].text, merged.facts["b"]) == ("new b", {"x": 20})
    assert merged.narratives["c"].status == "unavailable"  # nothing older to keep


def test_analyze_with_merge_saves_the_new_run_when_the_old_file_is_unreadable(monkeypatch):
    new = analysis_with({"a": Narrative("a", "unavailable", reason="empty answer")}, {"a": {"x": 10}})
    saved = []

    def unreadable():
        raise ValueError("Expecting value: line 1 column 1 (char 0)")

    monkeypatch.setattr(analyst, "run", lambda *a, **k: new)
    monkeypatch.setattr(analyst, "load", unreadable)
    monkeypatch.setattr(analyst, "save", saved.append)
    assert stages.analyze(DS, "results", "m", True, merge=True).unavailable == ["a"]
    assert saved == [new]


def test_analyze_passes_log_through(monkeypatch):
    seen = []
    monkeypatch.setattr(analyst, "run", lambda *a, **k: seen.append(k) or Analysis("skipped", "m"))
    monkeypatch.setattr(analyst, "save", lambda a: None)
    stages.analyze(DS, "results", "m/key", False, log=print)
    assert seen == [{"llm_model": "m/key", "enabled": False, "log": print}]


def test_report_writes_both_pages_into_the_out_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(report, "render", lambda *args: "<p>page</p>")
    monkeypatch.setattr(report, "standalone", lambda page: f"<html>{page}</html>")
    path = stages.report(DS, "results", "analysis")
    assert path == tmp_path / "out" / "report.html"
    assert path.read_text() == "<html><p>page</p></html>"
    assert (tmp_path / "out" / "report_page.html").read_text() == "<p>page</p>"
