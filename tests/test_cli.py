import pytest

from ucl import cli, dataset, model


def test_unknown_command_exits_with_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["nope"])
    assert exc.value.code == 2


def test_commands_run_in_pipeline_order():
    assert list(cli.COMMANDS)[:4] == ["fetch", "build", "model", "analyze"]


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
