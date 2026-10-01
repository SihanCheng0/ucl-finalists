"""Squad tables and player histories (spec §3.3). UEFA ignores the playerId filter, so a player's history comes
from the squad tables already cached, through an in-memory index player_id -> {(season, team_id)}."""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import config

POSITIONS = {"GOALKEEPER": "GK", "DEFENDER": "DEF", "MIDFIELDER": "MID", "FORWARD": "FWD"}
STAT_META = [{"key": key, "label": label, "kind": kind, "decimals": decimals, "colourable": colourable}
             for key, label, kind, decimals, colourable in config.PLAYER_STATS]
KIND = {meta["key"]: meta["kind"] for meta in STAT_META}


class SquadUnavailable(Exception):
    """UEFA couldn't be reached and there is no cached copy (HTTP 503)."""


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # NaN


def _plausible(stats: dict, minutes: float) -> dict:
    """Values outside PLAYER_STAT_RANGES become None: distance is checked per 90 minutes, which catches a feed
    that reports metres; speed and pass accuracy are checked as they are."""
    out = dict(stats)
    distance = out.get("distance_covered")
    if distance is not None and minutes >= 90:
        low, high = config.PLAYER_STAT_RANGES["distance_covered"]
        if not low <= distance / minutes * 90 <= high:
            out["distance_covered"] = None
    for key in ("top_speed", "passes_accuracy"):
        low, high = config.PLAYER_STAT_RANGES[key]
        if out.get(key) is not None and not low <= out[key] <= high:
            out[key] = None
    return out


def parse_squad(rows: list[dict]) -> list[dict]:
    """One dict per player, most minutes first. Older feeds leave zero counts out, so a count the feed reports
    for anyone in the squad is 0, not unknown, for a player who played without it."""
    reported = {s.get("name") for r in rows for s in r.get("statistics") or []}
    players = []
    for r in rows:
        values = {s.get("name"): _number(s.get("value")) for s in r.get("statistics") or []}
        minutes = values.get("minutes_played_official") or 0.0
        stats = {}
        for meta in STAT_META:
            value = values.get(meta["key"])
            if value is None and meta["kind"] == "count" and meta["key"] in reported and minutes > 0:
                value = 0.0
            stats[meta["key"]] = value
        if stats["passes_accuracy"] is None and stats.get("passes_attempted"):
            stats["passes_accuracy"] = round(100 * (stats.get("passes_completed") or 0) / stats["passes_attempted"], 1)
        info = r.get("player") or {}
        team = r.get("team") or {}
        players.append({
            "player_id": str(r.get("playerId") or info.get("id")),
            "name": info.get("internationalName") or info.get("clubShirtName") or "Unknown player",
            "position": POSITIONS.get(info.get("fieldPosition")),
            "shirt": info.get("clubJerseyNumber"),
            "age": _number(info.get("age")),
            "country": info.get("countryCode") or info.get("nationalityCountryCode"),
            "image_url": info.get("imageUrl"),
            "team_name": team.get("internationalName"),
            "minutes": minutes,
            "stats": _plausible(stats, minutes),
        })
    return sorted(players, key=lambda p: (-p["minutes"], p["name"]))


class PlayerService:
    """Squads on demand and the player index. `request_client` serves browsers (one quick attempt); the index
    job uses `background_client` (the full retry ladder). `pairs()` lists every (season, team_id) there is."""

    def __init__(self, request_client, background_client, pairs: Callable[[], list[tuple[int, str]]],
                 cache_dir: Path = config.PLAYERS_DIR, clock: Callable[[], float] = time.time):
        self._request, self._background, self._pairs = request_client, background_client, pairs
        self._cache_dir, self._clock = Path(cache_dir), clock
        self._lock = threading.Lock()
        self._index: dict[str, set[tuple[int, str]]] = {}
        self._failures: dict[tuple[int, str], float] = {}  # request-time failures, remembered briefly
        self._refreshing: set[tuple[int, str]] = set()
        self.unavailable: set[tuple[int, str]] = set()  # squads the last index job couldn't fetch
        self.running = False
        for season, team_id in self.cached():
            self._add(season, team_id, self._read(season, team_id))

    def _path(self, season: int, team_id: str) -> Path:
        return self._cache_dir / str(season) / f"{team_id}.json"

    def cached(self) -> list[tuple[int, str]]:
        if not self._cache_dir.exists():
            return []
        return [(int(path.parent.name), path.stem) for path in self._cache_dir.glob("*/*.json")
                if path.parent.name.isdecimal()]

    def _read(self, season: int, team_id: str) -> list[dict]:
        try:
            return json.loads(self._path(season, team_id).read_text())
        except (OSError, json.JSONDecodeError):
            return []

    def _add(self, season: int, team_id: str, rows: list[dict]) -> None:
        with self._lock:
            for r in rows:
                player_id = str(r.get("playerId") or (r.get("player") or {}).get("id"))
                self._index.setdefault(player_id, set()).add((season, team_id))

    def squad(self, team_id: str, season: int) -> dict:
        """A team-season's squad. A cached copy is served at once; for the live season an old copy is also
        refreshed in the background. Without a copy, one quick fetch; a failure is remembered for a minute."""
        key = (season, team_id)
        path = self._path(season, team_id)
        if path.exists():
            fetched = self._request.squad(season, team_id, max_age=None)  # the cached copy, no network
            age = self._clock() - path.stat().st_mtime
            stale = season == config.LIVE_SEASON and age > config.LIVE_MAX_AGE_S
            if stale:
                self._refresh_later(season, team_id)
            return self._payload(fetched.data, stale, fetched.fetched_at)
        failed_at = self._failures.get(key)
        if failed_at is not None and self._clock() - failed_at < config.REQUEST_FAILURE_MEMORY_S:
            raise SquadUnavailable("UEFA couldn't be reached a moment ago; try again in a minute")
        try:
            fetched = self._request.squad(season, team_id)
        except Exception as exc:  # noqa: BLE001
            self._failures[key] = self._clock()
            raise SquadUnavailable(f"UEFA couldn't be reached: {exc}") from exc
        self._add(season, team_id, fetched.data)
        return self._payload(fetched.data, fetched.stale, fetched.fetched_at)

    def _refresh_later(self, season: int, team_id: str) -> None:
        key = (season, team_id)
        with self._lock:
            if key in self._refreshing:
                return
            self._refreshing.add(key)

        def refresh() -> None:
            try:
                fetched = self._background.squad(season, team_id, max_age=0)
                self._add(season, team_id, fetched.data)
            except Exception:  # noqa: BLE001 - the old copy stays
                pass
            finally:
                with self._lock:
                    self._refreshing.discard(key)

        threading.Thread(target=refresh, daemon=True, name=f"squad-{season}-{team_id}").start()

    @staticmethod
    def _payload(rows: list[dict], stale: bool, fetched_at) -> dict:
        players = parse_squad(rows)
        return {"players": players, "minutes_published": any(p["minutes"] > 0 for p in players),
                "stale": stale, "fetched_at": fetched_at}

    def index_status(self) -> dict:
        expected = set(self._pairs())
        done = len(expected & set(self.cached()))
        return {"complete": bool(expected) and done == len(expected), "squads": {"done": done, "of": len(expected)},
                "running": self.running}

    def history(self, player_id: str) -> dict | None:
        """Every cached season of a player, newest first, across clubs; None for a player the index doesn't
        know."""
        with self._lock:
            pairs = sorted(self._index.get(str(player_id), ()), reverse=True)
        entries, info = [], None
        for season, team_id in pairs:
            players = {p["player_id"]: p for p in parse_squad(self._read(season, team_id))}
            player = players.get(str(player_id))
            if player is None:
                continue
            info = info or player
            entries.append({"season": season, "label": config.season_label(season), "team_id": team_id,
                            "team": config.DISPLAY_NAMES.get(team_id, player["team_name"] or team_id),
                            "position": player["position"], "minutes": player["minutes"], "stats": player["stats"],
                            "live": season == config.LIVE_SEASON})
        if info is None:
            return None
        return {
            "player": {key: info[key] for key in ("player_id", "name", "position", "shirt", "age", "country",
                                                  "image_url")},
            "history": entries,
            "index": self.index_status(),
            "unavailable_seasons": sorted({config.season_label(season) for season, _ in self.unavailable}),
        }

    def build_index(self, progress: Callable[[int, int], None], log: Callable[[str], None] = print,
                    workers: int = 4, client=None) -> list[tuple[tuple[int, str], str]]:
        """Fetch every squad not cached yet (and the live season's, which change), four at a time, with `client`
        (default: the background client). Returns the failures as ((season, team_id), reason)."""
        client = client or self._background
        pairs = sorted(set(self._pairs()))
        have = set(self.cached())
        todo = [p for p in pairs if p not in have or p[0] == config.LIVE_SEASON]
        done = len(pairs) - len(todo)
        progress(done, len(pairs))
        log(f"{len(pairs)} squads: {done} cached, {len(todo)} to fetch")
        failures: list[tuple[tuple[int, str], str]] = []

        def fetch(pair: tuple[int, str]):
            season, team_id = pair
            try:
                max_age = 0 if season == config.LIVE_SEASON else None
                self._add(season, team_id, client.squad(season, team_id, max_age=max_age).data)
                return pair, None
            except Exception as exc:  # noqa: BLE001 - reported, the rest go on
                return pair, f"{type(exc).__name__}: {exc}"

        self.running = True
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for pair, error in pool.map(fetch, todo):
                    if error is None:
                        done += 1
                    else:
                        failures.append((pair, error))
                    progress(done, len(pairs))
        finally:
            self.running = False
        self.unavailable = {pair for pair, _ in failures}
        return failures
