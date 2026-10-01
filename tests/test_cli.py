import pytest

from ucl import cli


def test_unknown_command_exits_with_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["nope"])
    assert exc.value.code == 2


def test_commands_run_in_pipeline_order():
    assert list(cli.COMMANDS)[:3] == ["fetch", "build", "model"]
