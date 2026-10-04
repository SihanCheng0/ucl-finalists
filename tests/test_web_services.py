from types import SimpleNamespace

import pytest

from ucl import analyst, dataset, model, stages
from ucl.analyst import Analysis
from ucl.web import services
from ucl.web.events import EventBus
from ucl.web.pipeline import BadRun, PipelineRunner


class FakeStore:
    def __init__(self):
        self.reloads = []

    def reload(self, *parts):
        self.reloads.append(parts)


def run_stage(name, **kwargs):
    """Run one real stage through a runner; returns its state, the counters, the bus and the store."""
    store, bus = FakeStore(), EventBus(boot="b")
    runner = PipelineRunner(services.stage_registry(store, **kwargs), bus)
    runner.start([name])
    assert runner.wait(10)
    state = runner.state()
    return next(s for s in state["stages"] if s["name"] == name), state["counters"], bus, store


def events_of(bus, type):
    subscription = bus.subscribe()
    events = []
    while (event := subscription.get(0)) is not None:
        if event.type == type:
            events.append(event.data)
    return events


@pytest.fixture
def loaders(monkeypatch):
    monkeypatch.setattr(dataset, "load", lambda: "ds")
    monkeypatch.setattr(model, "load", lambda: "results")
    monkeypatch.setattr(analyst, "load", lambda: "analysis")


def client_with_hook(hook):
    return SimpleNamespace(hook=hook)


def test_fetch_counts_requests_and_seasons(monkeypatch):
    def fetch(client, progress=None, log=print):
        client.hook("matches", "matches/2012", "network")
        client.hook("stats", "stats/1", "cache")
        progress({"season": 2012, "matches": 4, "phase_matches": 3, "missing": 0, "failed": 0})
        log("  e.g. match 9: down")
        return stages.FetchResult({})

    monkeypatch.setattr(stages, "fetch", fetch)
    stage, counters, bus, _ = run_stage("fetch", client_factory=client_with_hook)
    assert stage["status"] == "done"
    assert counters["requests"] == {"network": 1, "cache": 1} and counters["seasons"]["done"] == 1
    assert {"level": "info", "text": "e.g. match 9: down"} in events_of(bus, "log")


def test_fetch_with_failed_stats_fails_and_lists_them(monkeypatch):
    failed = {str(i): "down" for i in range(22)}
    monkeypatch.setattr(stages, "fetch", lambda client, progress=None, log=print: stages.FetchResult(failed))
    stage, _, bus, _ = run_stage("fetch", client_factory=client_with_hook)
    assert stage["status"] == "failed"
    assert stage["message"] == "22 match-stat requests failed; run fetch again to resume"
    errors = events_of(bus, "run_failed")[0]["errors"]
    assert errors[0] == "0: down" and errors[-1] == "… and 2 more" and len(errors) == 21


def test_build_marks_the_dataset_validated_and_reloads_it(monkeypatch):
    monkeypatch.setattr(stages, "build", lambda client: SimpleNamespace(team_seasons=[1, 2, 3], features=["a", "b"]))
    stage, counters, _, store = run_stage("build", client_factory=client_with_hook)
    assert (stage["status"], stage["message"]) == ("done", "3 team-seasons, 2 features")
    assert counters["validated"] is True and store.reloads == [("dataset",)]


def test_a_build_that_fails_validation_lists_every_error_and_reloads_nothing(monkeypatch):
    def invalid(client):
        raise dataset.ValidationError("2016 Roma: distance_km_pg = 3.2 is outside 90-140\n2017 Ajax: possession")

    monkeypatch.setattr(stages, "build", invalid)
    stage, counters, bus, store = run_stage("build", client_factory=client_with_hook)
    assert stage["status"] == "failed" and counters["validated"] is False and store.reloads == []
    assert events_of(bus, "run_failed") == [{"stage": "build", "errors": [
        "2016 Roma: distance_km_pg = 3.2 is outside 90-140", "2017 Ajax: possession"]}]


def test_model_moves_the_fold_and_ablation_counters_and_reloads_results(monkeypatch, loaders):
    def fit(ds, progress=None):
        progress({"fold": 3, "of": 15})
        progress({"ablation_set": "pedigree", "k": 1, "of": 7})
        return SimpleNamespace(metrics={"spearman_mean": 0.4512, "auc": 0.7349})

    monkeypatch.setattr(stages, "model", fit)
    stage, counters, _, store = run_stage("model")
    assert counters["folds"] == {"done": 3, "of": 15} and counters["ablation"] == {"done": 1, "of": 7}
    assert (stage["status"], stage["message"]) == ("done", "Spearman 0.45, AUC 0.73")
    assert store.reloads == [("results",)]


def fake_analyze(result):
    def analyze(ds, results, llm_model, enabled, log=print, merge=False):
        assert (ds, results, enabled, merge) == ("ds", "results", True, True)
        log("  2026-52280: ok, 2 call(s), 0 unsupported")
        log("  synthesis: unavailable, 3 call(s), 0 unsupported")
        return result
    return analyze


def test_analyze_counts_narratives_and_warns_about_any_it_could_not_write(monkeypatch, loaders):
    monkeypatch.setattr(stages, "analyze", fake_analyze(stages.AnalyzeResult(Analysis("ok", "m"), ["synthesis"])))
    stage, counters, bus, store = run_stage("analyze")
    assert counters["narratives"] == {"done": 2, "of": 11}
    assert stage["status"] == "warning"
    assert stage["message"] == ("1 write-up could not be written this time "
                                "(earlier text kept where there was one): synthesis")
    logs = [e["text"] for e in events_of(bus, "log")]
    assert logs[:2] == ["2026-52280: ok, 2 call(s), 0 unsupported",  # stripped of the CLI's indent
                        "synthesis: unavailable, 3 call(s), 0 unsupported"]
    assert store.reloads == [("analysis",)]


def test_analyze_warns_when_lm_studio_is_not_ready(monkeypatch, loaders):
    unavailable = Analysis("unavailable", "m", reason="server did not start: lms not found")
    monkeypatch.setattr(stages, "analyze", fake_analyze(stages.AnalyzeResult(unavailable)))
    stage, _, _, _ = run_stage("analyze")
    assert stage["status"] == "warning"
    assert stage["message"] == "AI model not ready (server did not start: lms not found); earlier write-ups kept"


def test_analyze_uses_the_chosen_llm_model(monkeypatch, loaders):
    seen = []

    def analyze(ds, results, llm_model, enabled, log=print, merge=False):
        seen.append(llm_model)
        return stages.AnalyzeResult(Analysis("ok", llm_model))

    monkeypatch.setattr(stages, "analyze", analyze)
    stage, _, _, _ = run_stage("analyze", llm_model="other/model")
    assert seen == ["other/model"] and (stage["status"], stage["message"]) == ("done", "0 write-ups")


def test_report_writes_the_pages(monkeypatch, loaders, tmp_path):
    monkeypatch.setattr(stages, "report", lambda ds, results, analysis: tmp_path / "report.html")
    stage, _, _, _ = run_stage("report")
    assert (stage["status"], stage["message"]) == ("done", "wrote report.html and report_page.html")


def test_missing_inputs_name_the_stage_to_run_first(tmp_path):
    processed, out = tmp_path / "processed", tmp_path / "out"
    assert services.missing_inputs(["model"], processed, out) == (
        f"model needs {processed / 'dataset.json'}: run build first")
    assert services.missing_inputs(["build", "model"], processed, out) is None  # the run makes it
    assert services.missing_inputs(["fetch"], processed, out) is None
    processed.mkdir()
    (processed / "dataset.json").write_text("{}")
    assert services.missing_inputs(["report"], processed, out) == (
        f"report needs {out / 'metrics.json'}: run model first")
    assert services.missing_inputs(["model", "report"], processed, out) is None


def test_build_services_shares_one_bus_and_checks_inputs(tmp_path):
    svc = services.build_services(tmp_path / "processed", tmp_path / "out", start_live=False)
    assert not svc.store.snapshot.ready and svc.runner.bus is svc.bus
    with pytest.raises(BadRun, match="run build first"):
        svc.runner.start(["model"])
