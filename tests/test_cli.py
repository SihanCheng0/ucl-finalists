from types import SimpleNamespace

import pytest

from ucl import analyst, cli, dataset, model


def test_unknown_command_exits_with_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["nope"])
    assert exc.value.code == 2


def test_commands_run_in_pipeline_order():
    assert list(cli.COMMANDS) == ["fetch", "build", "model", "analyze", "report"]


def test_build_reports_a_failed_plausibility_gate(monkeypatch, capsys):
    def implausible(client, seasons):
        raise dataset.ValidationError("2016 Roma: distance_km_pg = 3.2 is outside 90-140")

    monkeypatch.setattr(dataset, "build", implausible)
    assert cli.main(["build"]) == 2
    err = capsys.readouterr().err
    assert "dataset validation failed" in err and "2016 Roma" in err


def test_model_prints_intervals_chance_level_and_marginal_rho(built, monkeypatch, capsys):
    ds, results = built
    monkeypatch.setattr(dataset, "load", lambda: ds)
    monkeypatch.setattr(model, "run", lambda *args: results)
    monkeypatch.setattr(model, "save", lambda r: None)  # leave the real out/ alone
    assert cli.main(["model"]) == 0
    out = capsys.readouterr().out
    m = results.metrics
    lo, hi = m["ci"]["auc"]
    assert f"AUC {m['auc']:.2f} [{lo:.2f}, {hi:.2f}]" in out
    assert f"chance {m['finalists_in_top4_chance']:.0%}" in out
    assert "marginal_rho" in out and "sign_agree_folds" in out


def run_analyze(monkeypatch, analysis):
    monkeypatch.setattr(dataset, "load", lambda: SimpleNamespace(team_seasons=None, finals=None))
    monkeypatch.setattr(model, "load", lambda: None)
    monkeypatch.setattr(analyst, "run", lambda *args, **kwargs: analysis)
    monkeypatch.setattr(analyst, "save", lambda result: None)  # leave the real out/ alone
    return cli.main(["analyze", "--llm-model", "m/key"])


def test_analyze_says_why_the_llm_was_unavailable(monkeypatch, capsys):
    analysis = analyst.Analysis("unavailable", "m/key", reason="server did not start: lms not found")
    assert run_analyze(monkeypatch, analysis) == 0
    out = capsys.readouterr().out
    assert "analysis unavailable" in out and "server did not start: lms not found" in out


@pytest.mark.parametrize("status", ["ok", "skipped"])
def test_analyze_prints_no_reason_unless_the_llm_was_unavailable(monkeypatch, capsys, status):
    assert run_analyze(monkeypatch, analyst.Analysis(status, "m/key")) == 0
    assert "not ready" not in capsys.readouterr().out
