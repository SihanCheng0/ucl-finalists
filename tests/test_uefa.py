import json
import urllib.error

import pytest

from ucl import config
from ucl.uefa import MISSING_MARKER, UefaClient


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://example.test", code, "error", hdrs=None, fp=None)


class FakeFetch:
    """Scripted responses per URL. Each call consumes one item; the last item repeats."""

    def __init__(self, responses: dict[str, list]):
        self.responses = {url: list(items) for url, items in responses.items()}
        self.calls: list[str] = []

    def __call__(self, url: str, timeout: float):
        self.calls.append(url)
        queue = self.responses[url]
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item


def make_client(tmp_path, responses):
    fetch = FakeFetch(responses)
    return UefaClient(cache_dir=tmp_path, fetch=fetch, sleep=lambda _: None), fetch


def test_matches_are_cached(tmp_path):
    url = config.MATCHES_URL.format(season=2026)
    client, fetch = make_client(tmp_path, {url: [[{"id": "1"}]]})
    assert client.matches(2026) == [{"id": "1"}]
    assert client.matches(2026) == [{"id": "1"}]
    assert fetch.calls == [url]


def test_transient_errors_are_retried(tmp_path):
    url = config.MATCHES_URL.format(season=2026)
    script = [http_error(503), urllib.error.URLError("reset"), TimeoutError(), [{"id": "1"}]]
    client, fetch = make_client(tmp_path, {url: script})
    assert client.matches(2026) == [{"id": "1"}]
    assert len(fetch.calls) == 4


def test_gives_up_after_five_attempts_and_caches_nothing(tmp_path):
    url = config.MATCHES_URL.format(season=2026)
    client, fetch = make_client(tmp_path, {url: [http_error(500)]})
    with pytest.raises(RuntimeError, match="giving up"):
        client.matches(2026)
    assert len(fetch.calls) == 5
    assert not list(tmp_path.rglob("*.json"))


def test_permanent_4xx_on_stats_is_cached_as_missing(tmp_path):
    url = config.MATCH_STATS_URL.format(match_id="42")
    client, fetch = make_client(tmp_path, {url: [http_error(404)]})
    assert client.team_match_stats("42") is None
    assert client.team_match_stats("42") is None
    assert len(fetch.calls) == 1
    assert json.loads((tmp_path / "stats" / "42.json").read_text()) == MISSING_MARKER


def test_forbidden_on_stats_fails_fast_and_is_not_cached(tmp_path):
    # a 403 may be a block, not a missing resource: never cache it as "missing"
    url = config.MATCH_STATS_URL.format(match_id="43")
    client, fetch = make_client(tmp_path, {url: [http_error(403)]})
    with pytest.raises(Exception, match="403"):
        client.team_match_stats("43")
    assert len(fetch.calls) == 1
    assert not list(tmp_path.rglob("*.json"))


def test_permanent_4xx_on_matches_raises(tmp_path):
    url = config.MATCHES_URL.format(season=2026)
    client, _ = make_client(tmp_path, {url: [http_error(404)]})
    with pytest.raises(Exception, match="404"):
        client.matches(2026)


def test_empty_stats_list_means_missing(tmp_path):
    url = config.MATCH_STATS_URL.format(match_id="7")
    client, _ = make_client(tmp_path, {url: [[]]})
    assert client.team_match_stats("7") is None


def test_coefficients_follow_pages(tmp_path):
    def page(n):
        members = [{"member": {"id": str(i)}, "overallRanking": {"totalValue": 1.0}} for i in range(n)]
        return {"data": {"members": members}}

    p1 = config.COEF_URL.format(season=2025, page=1)
    p2 = config.COEF_URL.format(season=2025, page=2)
    client, fetch = make_client(tmp_path, {p1: [page(config.COEF_PAGE_SIZE)], p2: [page(3)]})
    assert len(client.coefficients(2025)) == config.COEF_PAGE_SIZE + 3
    assert fetch.calls == [p1, p2]


def test_many_reports_failures_without_stopping(tmp_path):
    ok = config.MATCH_STATS_URL.format(match_id="1")
    bad = config.MATCH_STATS_URL.format(match_id="2")
    entry = [{"teamId": "9", "statistics": []}]
    client, _ = make_client(tmp_path, {ok: [entry], bad: [http_error(503)]})
    results, failed = client.team_match_stats_many(["1", "2"])
    assert results == {"1": entry}
    assert failed == ["2"]
