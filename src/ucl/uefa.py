"""Cached, retrying client for UEFA's public JSON APIs (spec §4, §10)."""
from __future__ import annotations

import http.client
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable

from . import config

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
# Cached in place of a 404/410 so builds stay offline and deterministic.
MISSING_MARKER = {"__missing__": True}
MISSING_CODES = {404, 410}
RETRY_CODES = {408, 429}
MAX_COEF_PAGES = 20
# OSError covers URLError, TimeoutError, ConnectionError and ssl.SSLError.
TRANSIENT_ERRORS = (OSError, http.client.HTTPException, json.JSONDecodeError)

Fetch = Callable[[str, float], Any]
Check = Callable[[Any], bool]


class PermanentHTTPError(Exception):
    """A 4xx that retrying will not fix."""

    def __init__(self, message: str, code: int):
        super().__init__(message)
        self.code = code


def http_get_json(url: str, timeout: float) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


def _is_match_list(data: Any) -> bool:
    return isinstance(data, list) and len(data) > 0


def _is_coefficient_page(data: Any) -> bool:
    return isinstance(data, dict) and isinstance((data.get("data") or {}).get("members"), list)


class UefaClient:
    def __init__(
        self,
        cache_dir: Path = config.RAW_DIR,
        max_workers: int = 4,
        fetch: Fetch = http_get_json,
        sleep: Callable[[float], None] = time.sleep,
        on_request: Callable[[str, str, str], None] | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.max_workers = max_workers
        self._fetch = fetch
        self._sleep = sleep
        self._on_request = on_request

    def matches(self, season: int) -> list[dict]:
        return self._cached(f"matches/{season}", config.MATCHES_URL.format(season=season), check=_is_match_list)

    def team_match_stats(self, match_id: str) -> list[dict] | None:
        """The two teams' stats for a match, or None if UEFA has none."""
        data = self._cached(
            f"stats/{match_id}", config.MATCH_STATS_URL.format(match_id=match_id), missing_ok=True
        )
        return data or None

    def team_match_stats_many(
        self, match_ids: Iterable[str]
    ) -> tuple[dict[str, list[dict] | None], dict[str, str]]:
        """Fetch concurrently. Returns (results, {failed id: reason})."""

        def one(match_id: str):
            try:
                return match_id, self.team_match_stats(match_id), None
            except Exception as exc:  # noqa: BLE001 - reported to the caller with its reason
                return match_id, None, exc

        results: dict[str, list[dict] | None] = {}
        failed: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for match_id, data, error in pool.map(one, list(dict.fromkeys(match_ids))):
                if error is None:
                    results[match_id] = data
                else:
                    failed[match_id] = f"{type(error).__name__}: {error}"
        return results, failed

    def coefficients(self, season: int) -> list[dict]:
        """All members of the 5-year club ranking for `season`, across pages."""
        members: list[dict] = []
        for page in range(1, MAX_COEF_PAGES + 1):
            data = self._cached(
                f"coefficients/{season}_p{page}",
                config.COEF_URL.format(season=season, page=page),
                check=_is_coefficient_page,
            )
            batch = data["data"]["members"]
            members.extend(batch)
            if len(batch) < config.COEF_PAGE_SIZE:
                return members
        raise RuntimeError(f"coefficient ranking {season} did not end within {MAX_COEF_PAGES} pages")

    def _cached(self, key: str, url: str, missing_ok: bool = False, check: Check | None = None) -> Any:
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"corrupt cache file {path}: delete it and run `uv run ucl fetch` again") from exc
            source = "cache"
        else:
            try:
                data = self._get(url)
            except PermanentHTTPError as exc:
                # Only "not found" means missing; a 403 or 400 may be a block and must not poison the cache.
                if not (missing_ok and exc.code in MISSING_CODES):
                    raise
                data = MISSING_MARKER
            if check is not None and data != MISSING_MARKER and not check(data):
                raise RuntimeError(f"unexpected response shape from {url}; nothing was cached")
            _atomic_write_json(path, data)
            source = "network"
        self._notify(key, source)
        return None if data == MISSING_MARKER else data

    def _notify(self, key: str, source: str) -> None:
        """Tell the on_request hook about one resolved resource (once, however many attempts it took). It runs on
        team_match_stats_many's worker threads, so the hook must be thread-safe; a broken hook never breaks a fetch."""
        if self._on_request is None:
            return
        try:
            self._on_request(key.split("/", 1)[0], key, source)
        except Exception:  # noqa: BLE001 - the hook is a progress display, not part of the data path
            pass

    def _get(self, url: str) -> Any:
        delays = config.HTTP_RETRY_DELAYS_S
        last: Exception | None = None
        for attempt in range(len(delays) + 1):
            try:
                return self._fetch(url, config.HTTP_TIMEOUT_S)
            except urllib.error.HTTPError as exc:
                if 400 <= exc.code < 500 and exc.code not in RETRY_CODES:
                    raise PermanentHTTPError(f"HTTP {exc.code} for {url}", exc.code) from exc
                last = exc
            except TRANSIENT_ERRORS as exc:
                last = exc
            if attempt < len(delays):
                self._sleep(delays[attempt] + random.uniform(0, 0.5))
        raise RuntimeError(f"giving up on {url} after {len(delays) + 1} attempts: {last}") from last
