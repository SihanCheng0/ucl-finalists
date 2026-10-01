import threading
from types import SimpleNamespace

import pytest

from ucl import cli, config
from ucl.web import server


def test_web_is_dispatched_before_the_stage_parser(monkeypatch):
    seen = []
    monkeypatch.setattr(server, "main", lambda argv: seen.append(argv) or 0)
    assert cli.main(["web", "--no-open", "--port", "9000"]) == 0
    assert seen == [["--no-open", "--port", "9000"]]


def test_help_mentions_the_dashboard(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    assert "ucl web" in capsys.readouterr().out


def test_a_missing_build_prints_the_commands_and_exits_1(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(config, "WEB_DIR", tmp_path / "web")
    assert server.main(["--no-open"]) == 1
    err = capsys.readouterr().err
    assert "npm install && npm run build" in err and str(tmp_path / "web") in err


def test_the_server_listens_on_localhost_only_and_stops_quickly():
    settings = server.uvicorn_config(object(), 9000)
    assert (settings.host, settings.port, settings.timeout_graceful_shutdown) == ("127.0.0.1", 9000, 2)


def test_the_browser_opens_once_the_server_is_up(monkeypatch):
    opened = []
    monkeypatch.setattr(server.webbrowser, "open", opened.append)
    fake = SimpleNamespace(started=False, should_exit=False)
    threading.Timer(0.05, lambda: setattr(fake, "started", True)).start()
    assert server.open_when_up(fake, "http://127.0.0.1:8787/", poll=0.01, limit=2)
    assert opened == ["http://127.0.0.1:8787/"]


def test_the_browser_stays_shut_if_the_server_never_starts(monkeypatch):
    opened = []
    monkeypatch.setattr(server.webbrowser, "open", opened.append)
    assert not server.open_when_up(SimpleNamespace(started=False, should_exit=True), "http://x/", poll=0.01)
    assert opened == []
