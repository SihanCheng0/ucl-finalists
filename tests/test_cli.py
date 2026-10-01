import pytest

from ucl import cli, dataset


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
