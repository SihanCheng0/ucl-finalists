import http.client
import json
import threading
import time
import urllib.error

import pytest

from ucl import config
from ucl.uefa import MISSING_MARKER, PermanentHTTPError, UefaClient


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


@pytest.mark.parametrize("error", [
    http_error(408), http_error(429), http_error(503), urllib.error.URLError("reset"), TimeoutError(),
    json.JSONDecodeError("bad", "", 0), http.client.IncompleteRead(b""),
])
def test_transient_errors_are_retried(tmp_path, error):
    url = config.MATCHES_URL.format(season=2026)
    client, fetch = make_client(tmp_path, {url: [error, [{"id": "1"}]]})
    assert client.matches(2026) == [{"id": "1"}]
    assert len(fetch.calls) == 2


def test_backoff_is_one_two_four_eight_seconds_plus_jitter(tmp_path):
    url = config.MATCHES_URL.format(season=2026)
    sleeps: list[float] = []
    client = UefaClient(cache_dir=tmp_path, fetch=FakeFetch({url: [http_error(500)]}), sleep=sleeps.append)
    with pytest.raises(RuntimeError, match="giving up"):
        client.matches(2026)
    assert len(sleeps) == 4
    assert all(d <= s <= d + 0.5 for s, d in zip(sleeps, (1, 2, 4, 8)))


def test_gives_up_after_five_attempts_and_caches_nothing(tmp_path):
    url = config.MATCHES_URL.format(season=2026)
    client, fetch = make_client(tmp_path, {url: [http_error(500)]})
    with pytest.raises(RuntimeError, match="giving up"):
        client.matches(2026)
    assert len(fetch.calls) == 5
    assert not list(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("code", [404, 410])
def test_not_found_on_stats_is_cached_as_missing(tmp_path, code):
    url = config.MATCH_STATS_URL.format(match_id="42")
    client, fetch = make_client(tmp_path, {url: [http_error(code)]})
    assert client.team_match_stats("42") is None
    assert client.team_match_stats("42") is None
    assert len(fetch.calls) == 1
    assert json.loads((tmp_path / "stats" / "42.json").read_text()) == MISSING_MARKER


def test_forbidden_on_stats_fails_fast_and_is_not_cached(tmp_path):
    # a 403 may be a block, not a missing resource: never cache it as "missing"
    url = config.MATCH_STATS_URL.format(match_id="43")
    client, fetch = make_client(tmp_path, {url: [http_error(403)]})
    with pytest.raises(PermanentHTTPError) as exc:
        client.team_match_stats("43")
    assert exc.value.code == 403
    assert len(fetch.calls) == 1
    assert not list(tmp_path.rglob("*.json"))


def test_permanent_4xx_on_matches_raises_after_one_call(tmp_path):
    url = config.MATCHES_URL.format(season=2026)
    client, fetch = make_client(tmp_path, {url: [http_error(404)]})
    with pytest.raises(PermanentHTTPError) as exc:
        client.matches(2026)
    assert exc.value.code == 404 and len(fetch.calls) == 1


def test_empty_stats_list_means_missing(tmp_path):
    url = config.MATCH_STATS_URL.format(match_id="7")
    client, _ = make_client(tmp_path, {url: [[]]})
    assert client.team_match_stats("7") is None


def test_unexpected_response_shapes_are_not_cached(tmp_path):
    url = config.MATCHES_URL.format(season=2026)
    client, _ = make_client(tmp_path, {url: [[]]})
    with pytest.raises(RuntimeError, match="unexpected response"):
        client.matches(2026)
    assert not list(tmp_path.rglob("*.json"))


def test_corrupt_cache_file_names_the_file(tmp_path):
    (tmp_path / "matches").mkdir()
    (tmp_path / "matches" / "2026.json").write_text("{truncated")
    client, fetch = make_client(tmp_path, {})
    with pytest.raises(RuntimeError, match="2026.json"):
        client.matches(2026)
    assert fetch.calls == []


def test_coefficients_follow_pages(tmp_path):
    def page(n):
        members = [{"member": {"id": str(i)}, "overallRanking": {"totalValue": 1.0}} for i in range(n)]
        return {"data": {"members": members}}

    p1 = config.COEF_URL.format(season=2025, page=1)
    p2 = config.COEF_URL.format(season=2025, page=2)
    client, fetch = make_client(tmp_path, {p1: [page(config.COEF_PAGE_SIZE)], p2: [page(3)]})
    assert len(client.coefficients(2025)) == config.COEF_PAGE_SIZE + 3
    assert fetch.calls == [p1, p2]


def test_many_reports_failures_with_reasons(tmp_path):
    ok = config.MATCH_STATS_URL.format(match_id="1")
    bad = config.MATCH_STATS_URL.format(match_id="2")
    entry = [{"teamId": "9", "statistics": []}]
    client, _ = make_client(tmp_path, {ok: [entry], bad: [http_error(503)]})
    results, failed = client.team_match_stats_many(["1", "2"])
    assert results == {"1": entry}
    assert list(failed) == ["2"] and "giving up" in failed["2"]


def test_many_fetches_each_unique_id_once_with_at_most_four_in_flight(tmp_path):
    lock, state, calls = threading.Lock(), {"now": 0, "peak": 0}, []

    def fetch(url, timeout):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
            calls.append(url)
        time.sleep(0.01)
        with lock:
            state["now"] -= 1
        return [{"teamId": "1", "statistics": []}]

    client = UefaClient(cache_dir=tmp_path, fetch=fetch, sleep=lambda _: None)
    results, failed = client.team_match_stats_many([str(i) for i in range(30)] * 2)
    assert len(results) == 30 and not failed
    assert len(calls) == 30 and state["peak"] <= 4
    assert not list(tmp_path.rglob("*.tmp"))
