"""The CLI's exact output, recorded before stages.py took over the stage bodies (spec §4.1).

Run once with UPDATE_GOLDEN=1 to record tests/golden/cli/*.txt from the current code. Every later run must
reproduce them byte for byte. model_ok and all_ok hold real scikit-learn/shap numbers from the synthetic `built`
fixture, so a dependency upgrade can change them: re-record those two deliberately when that happens."""
import dataclasses
import os
from pathlib import Path

import pandas as pd
import pytest

from ucl import analyst, cli, config, dataset, labels, model, report, uefa
from ucl.analyst import Analysis, Narrative

GOLDEN = Path(__file__).parent / "golden" / "cli"


def check(name: str, code: int, out: str, err: str) -> None:
    text = f"exit {code}\n--- stdout\n{out}--- stderr\n{err}"
    path = GOLDEN / f"{name}.txt"
    if os.environ.get("UPDATE_GOLDEN"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    assert path.exists(), f"no golden file {path.name}: run once with UPDATE_GOLDEN=1"
    assert text == path.read_text()


def run(capsys, *argv):
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


class FetchClient:
    """What fetch_all asks of UefaClient: 4 matches a season (3 in the first phase), coefficients and stats."""

    def __init__(self, fail=()):
        self.fail = set(fail)

    def matches(self, season):
        return [{"match_id": f"{season}-{i}", "depth": 0 if i < 3 else 5} for i in range(4)]

    def coefficients(self, season):
        return []

    def team_match_stats_many(self, match_ids):
        ids = list(match_ids)
        stats = {m: (None if m.endswith("-2") else [{}]) for m in ids if m not in self.fail}
        return stats, {m: "RuntimeError: giving up after 5 attempts" for m in ids if m in self.fail}


def patch_fetch(monkeypatch, fail=()):
    monkeypatch.setattr(config, "SEASONS", [2012, 2013])
    monkeypatch.setattr(labels, "match_rows", lambda raw, season: pd.DataFrame(raw, columns=["match_id", "depth"]))
    monkeypatch.setattr(uefa, "UefaClient", lambda: FetchClient(fail))


def patch_build(monkeypatch, ds, error=None, on_validate=False):
    calls = []

    def build(client, seasons):
        calls.append("build")
        if error and not on_validate:
            raise dataset.ValidationError(error)
        return ds

    def validate(built):
        calls.append("validate")
        if error and on_validate:
            raise dataset.ValidationError(error)

    monkeypatch.setattr(uefa, "UefaClient", lambda: FetchClient())
    monkeypatch.setattr(dataset, "build", build)
    monkeypatch.setattr(dataset, "validate", validate)
    monkeypatch.setattr(dataset, "save", lambda built: calls.append("save"))
    return calls


def with_real_notes(ds):
    """A copy with the notes a real build prints, leaving the shared fixtures untouched."""
    return dataclasses.replace(ds, notes={"dropped_features": ["save_pct"], "imputed_values": 0, "coef_imputed": 11,
                                          "team_matches_without_stats": 4})


def patch_model(monkeypatch, built):
    ds, results = built
    monkeypatch.setattr(dataset, "load", lambda: ds)
    monkeypatch.setattr(model, "run", lambda *args: results)
    monkeypatch.setattr(model, "save", lambda r: None)
    monkeypatch.setattr(model, "load", lambda: results)


def patch_analyze(monkeypatch, analysis):
    calls = []

    def run_llm(*args, **kwargs):
        calls.append({k: kwargs[k] for k in ("llm_model", "enabled")})
        log = kwargs.get("log", print)  # analyst.run's own default, which the CLI relies on today
        for key, n in analysis.narratives.items():
            log(f"  {key}: {n.status}, {n.calls} call(s), {len(n.unsupported)} unsupported")
        return analysis

    monkeypatch.setattr(analyst, "run", run_llm)
    monkeypatch.setattr(analyst, "save", lambda a: calls.append("save"))
    monkeypatch.setattr(analyst, "load", lambda: analysis)
    return calls


def patch_report(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(report, "render", lambda *args: "<p>page</p>")
    monkeypatch.setattr(report, "standalone", lambda page: f"<html>{page}</html>")


def mixed_analysis():
    return Analysis("ok", "m/key", {
        "2026-1": Narrative("2026-1", "ok", text="t", unsupported=["7.5"], calls=2),
        "2026-2": Narrative("2026-2", "unavailable", calls=3, reason="empty answer"),
    })


def test_fetch_ok(monkeypatch, capsys):
    patch_fetch(monkeypatch)
    check("fetch_ok", *run(capsys, "fetch"))


def test_fetch_some_failed(monkeypatch, capsys):
    patch_fetch(monkeypatch, fail={"2012-0"})
    check("fetch_some_failed", *run(capsys, "fetch"))


def test_fetch_stops_when_a_whole_season_fails(monkeypatch, capsys):
    patch_fetch(monkeypatch, fail={"2012-0", "2012-1", "2012-2"})
    check("fetch_stops", *run(capsys, "fetch"))


def test_build_ok(monkeypatch, capsys, synthetic_ds):
    calls = patch_build(monkeypatch, with_real_notes(synthetic_ds))
    check("build_ok", *run(capsys, "build"))
    assert calls == ["build", "validate", "save"]


def test_build_fails_validation_while_building(monkeypatch, capsys, synthetic_ds):
    calls = patch_build(monkeypatch, synthetic_ds, error="2016 Roma: distance_km_pg = 3.2 is outside 90-140")
    check("build_invalid", *run(capsys, "build"))
    assert "save" not in calls


def test_build_fails_validation_after_building(monkeypatch, capsys, synthetic_ds):
    calls = patch_build(monkeypatch, synthetic_ds, error="finals: 2026 expected 52747", on_validate=True)
    check("build_invalid_on_validate", *run(capsys, "build"))
    assert calls == ["build", "validate"]


def test_model_ok(monkeypatch, capsys, built):
    patch_model(monkeypatch, built)
    check("model_ok", *run(capsys, "model"))


def test_analyze_ok_with_a_flagged_and_an_unavailable_narrative(monkeypatch, capsys, built):
    patch_model(monkeypatch, built)
    calls = patch_analyze(monkeypatch, mixed_analysis())
    check("analyze_ok", *run(capsys, "analyze", "--llm-model", "m/key"))
    assert calls == [{"llm_model": "m/key", "enabled": True}, "save"]


def test_analyze_unavailable(monkeypatch, capsys, built):
    patch_model(monkeypatch, built)
    patch_analyze(monkeypatch, Analysis("unavailable", "m/key", reason="server did not start: lms not found"))
    check("analyze_unavailable", *run(capsys, "analyze", "--llm-model", "m/key"))


def test_analyze_skipped_with_no_ai(monkeypatch, capsys, built):
    patch_model(monkeypatch, built)
    calls = patch_analyze(monkeypatch, Analysis("skipped", "m/key"))
    check("analyze_skipped", *run(capsys, "analyze", "--no-ai", "--llm-model", "m/key"))
    assert calls == [{"llm_model": "m/key", "enabled": False}, "save"]


def test_report_ok(monkeypatch, capsys, tmp_path, built):
    patch_model(monkeypatch, built)
    patch_analyze(monkeypatch, mixed_analysis())
    patch_report(monkeypatch, tmp_path)
    code, out, err = run(capsys, "report")
    check("report_ok", code, out.replace(str(tmp_path), "<TMP>"), err)
    assert (tmp_path / "out" / "report_page.html").read_text() == "<p>page</p>"
    assert (tmp_path / "out" / "report.html").read_text() == "<html><p>page</p></html>"


def test_all_ok(monkeypatch, capsys, tmp_path, built):
    patch_fetch(monkeypatch)
    patch_build(monkeypatch, with_real_notes(built[0]))
    patch_model(monkeypatch, built)
    patch_analyze(monkeypatch, mixed_analysis())
    patch_report(monkeypatch, tmp_path)
    code, out, err = run(capsys, "all", "--llm-model", "m/key")
    check("all_ok", code, out.replace(str(tmp_path), "<TMP>"), err)


def test_all_stops_at_the_first_failing_stage(monkeypatch, capsys, built):
    patch_fetch(monkeypatch)
    patch_build(monkeypatch, built[0], error="2016 Roma: distance_km_pg = 3.2 is outside 90-140")
    later = []
    monkeypatch.setattr(model, "run", lambda *args: later.append("model"))
    check("all_stops_at_build", *run(capsys, "all"))
    assert later == []


def test_build_lets_a_failed_stats_request_raise(monkeypatch, synthetic_ds):
    patch_build(monkeypatch, synthetic_ds)

    def failing(client, seasons):
        raise RuntimeError("3 match-stat requests failed (e.g. 2012-0: down); run `uv run ucl fetch` again first")

    monkeypatch.setattr(dataset, "build", failing)
    with pytest.raises(RuntimeError, match="run `uv run ucl fetch` again first"):
        cli.main(["build"])
