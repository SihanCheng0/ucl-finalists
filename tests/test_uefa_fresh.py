import time
import urllib.error

import pytest

from ucl import config
from ucl.uefa import UefaClient


def http_error(code):
    return urllib.error.HTTPError("https://example.test", code, "error", hdrs=None, fp=None)


class Script:
    """Responses per URL, consumed in order; the last one repeats. Records every call."""

    def __init__(self, responses):
        self.responses = {url: list(items) for url, items in responses.items()}
        self.calls = []

    def __call__(self, url, timeout):
        self.calls.append((url, timeout))
        queue = self.responses[url]
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item


def client(tmp_path, responses, shift, **kwargs):
    script = Script(responses)
    return UefaClient(cache_dir=tmp_path, fetch=script, sleep=lambda _: None,
                      clock=lambda: time.time() + shift[0], **kwargs), script


MATCHES = config.MATCHES_URL.format(season=2027)
HOUR = 3600


def test_a_fresh_copy_comes_from_the_cache_and_an_old_one_is_refetched(tmp_path):
    shift = [0]
    c, script = client(tmp_path, {MATCHES: [[{"id": "1"}], [{"id": "1"}, {"id": "2"}]]}, shift)
    first = c.matches_fresh(2027, max_age=6 * HOUR)
    assert (first.data, first.stale) == ([{"id": "1"}], False) and first.fetched_at is not None
    assert c.matches_fresh(2027, max_age=6 * HOUR).data == [{"id": "1"}] and len(script.calls) == 1
    shift[0] = 7 * HOUR
    assert c.matches_fresh(2027, max_age=6 * HOUR).data == [{"id": "1"}, {"id": "2"}] and len(script.calls) == 2


def test_a_failed_refetch_serves_the_old_copy_as_stale(tmp_path):
    shift = [0]
    c, _ = client(tmp_path, {MATCHES: [[{"id": "1"}], OSError("offline")]}, shift, retry_delays=())
    fresh = c.matches_fresh(2027, max_age=HOUR)
    shift[0] = 2 * HOUR
    stale = c.matches_fresh(2027, max_age=HOUR)
    assert (stale.data, stale.stale, stale.fetched_at) == ([{"id": "1"}], True, fresh.fetched_at)


def test_with_no_copy_a_failure_raises(tmp_path):
    c, script = client(tmp_path, {MATCHES: [OSError("offline")]}, [0], retry_delays=(), timeout=8)
    with pytest.raises(RuntimeError, match="after 1 attempts"):
        c.matches_fresh(2027, max_age=HOUR)
    assert script.calls == [(MATCHES, 8)]  # the request client makes one quick attempt


def test_missing_live_stats_are_not_cached_so_they_are_asked_for_again(tmp_path):
    url = config.MATCH_STATS_URL.format(match_id="9")
    c, script = client(tmp_path, {url: [http_error(404), [], [{"teamId": "1"}]]}, [0])
    assert c.team_match_stats_fresh("9", max_age=None).data is None
    assert c.team_match_stats_fresh("9", max_age=None).data == []
    assert c.team_match_stats_fresh("9", max_age=None).data == [{"teamId": "1"}]
    assert c.team_match_stats_fresh("9", max_age=None).data == [{"teamId": "1"}] and len(script.calls) == 3


def squad_url(offset):
    return config.PLAYERS_URL.format(season=2026, team_id="52280", limit=config.PLAYER_PAGE_SIZE, offset=offset)


def test_a_squad_is_read_page_by_page_until_a_page_is_empty_and_cached(tmp_path):
    page0 = [{"playerId": str(i)} for i in range(3)]
    c, script = client(tmp_path, {squad_url(0): [page0], squad_url(100): [[]]}, [0])
    assert c.squad(2026, "52280").data == page0
    assert c.squad(2026, "52280").data == page0 and len(script.calls) == 2
    assert (tmp_path / "players" / "2026" / "52280.json").exists()


def test_a_feed_that_ignores_the_offset_does_not_loop(tmp_path):
    page0 = [{"playerId": "1"}]
    c, script = client(tmp_path, {squad_url(0): [page0], squad_url(100): [page0]}, [0])
    assert c.squad(2026, "52280").data == page0 and len(script.calls) == 2


def test_a_live_squad_falls_back_to_its_stale_copy(tmp_path):
    shift = [0]
    page0 = [{"playerId": "1"}]
    c, _ = client(tmp_path, {squad_url(0): [page0, OSError("offline")], squad_url(100): [[]]}, shift,
                  retry_delays=())
    c.squad(2026, "52280", max_age=HOUR)
    shift[0] = 2 * HOUR
    again = c.squad(2026, "52280", max_age=HOUR)
    assert (again.data, again.stale) == (page0, True)
