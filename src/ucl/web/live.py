"""The live season (spec §3.2): the current Champions League season, built in memory from UEFA's feeds in the
team_seasons schema. It is never written to data/processed, validated or modelled."""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from .. import config, labels
from .. import features as feat

STAGE_LABEL = "League phase (in progress)"


@dataclass(frozen=True)
class LiveSnapshot:
    season: int
    rows: pd.DataFrame  # team_seasons columns plus `live` and `provisional`
    stale: bool  # a refresh it needed failed, so a cached copy was used
    fetched_at: datetime | None  # when the match list was fetched
    finished_matches: int
    built_at: float


@dataclass(frozen=True)
class LiveState:
    snapshot: LiveSnapshot | None
    status: str  # "loading" | "ready" | "stale" | "unavailable"
    message: str = ""

    @property
    def available(self) -> bool:
        return self.snapshot is not None

    @property
    def fetched_at(self) -> datetime | None:
        return self.snapshot.fetched_at if self.snapshot is not None else None


def rank_among(history: pd.Series, value: float) -> float | None:
    """The percentile features.add_percentiles would give `value` among the historical values plus itself
    (average rank for ties), so a live team's 'teams beaten since 2011-12' means the same as a historical one's."""
    if value is None or pd.isna(value):
        return None
    known = history.dropna().to_numpy(dtype=float)
    below, equal = int((known < value).sum()), int((known == value).sum()) + 1
    return float(round(100 * (below + (equal + 1) / 2) / (len(known) + 1)))


def live_frame(raw: list[dict], stats: dict[str, list | None], coefficients: list[dict], history: pd.DataFrame,
               features: list[str], season: int = config.LIVE_SEASON) -> pd.DataFrame:
    """Every team in the first phase (taken from all fixtures, whatever their status), with features from the
    finished matches only. pct_season ranks within the live field; pct_all ranks against all of history."""
    every = labels.match_rows(raw, season)
    every = every[every["depth"] == 0]
    field = labels.stages(every)[["season", "team_id", "team"]]
    finished_raw = [m for m in raw if m.get("status") == "FINISHED"]
    per_team = pd.DataFrame(columns=["season", "team_id", "n_matches", "n_with_stats"])
    if finished_raw:
        finished = labels.match_rows(finished_raw, season)
        finished = finished[finished["depth"] == 0]
        if len(finished):
            per_team = feat.season_features(feat.team_match_rows(finished, stats))
    rows = field.merge(per_team, on=["season", "team_id"], how="left")
    for column in ("n_matches", "n_with_stats"):
        rows[column] = pd.to_numeric(rows[column]).fillna(0).astype(int)
    for f in features:
        if f not in rows:
            rows[f] = np.nan
    rows, _ = feat.attach_coefficients(rows, feat.coefficient_table({season - 1: coefficients}))
    played = rows["n_matches"] > 0
    data = rows.loc[played].copy()
    if len(data):
        data, _ = feat.impute_season_median(data, features)
        data = pd.concat([data, feat.zscore_within_season(data, features)], axis=1)
        for f in features:
            data[f"pct_season_{f}"] = (data[f].rank(pct=True) * 100).round()
            data[f"pct_all_{f}"] = [rank_among(history[f], v) if f in history else None for v in data[f]]
    rows = pd.concat([data, rows.loc[~played]], ignore_index=True)
    rows["in_ko"] = False
    rows["ko_stage"] = np.nan
    rows["stage_label"] = STAGE_LABEL
    rows["team_display"] = rows["team_id"].map(config.DISPLAY_NAMES).fillna(rows["team"])
    rows["live"] = True
    rows["provisional"] = rows["n_matches"] < config.phase_matches(season)
    rows["complete"] = False
    rows["reached_final"] = False
    rows["is_target"] = False
    return rows.sort_values("team_id").reset_index(drop=True)


def _age_s(iso: str | None, now: float) -> float:
    if not iso:
        return 0.0
    try:
        return now - datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def build_snapshot(client, history: pd.DataFrame, features: list[str], force: bool = False,
                   season: int = config.LIVE_SEASON, clock: Callable[[], float] = time.time) -> LiveSnapshot:
    """Fetch the live season and build its snapshot. A match's stats settle a day after full time: a copy written
    after that is kept for good, and one written earlier is fetched once more. Until then they are refetched when
    older than LIVE_MAX_AGE_S (or always, with `force`)."""
    max_age = 0 if force else config.LIVE_MAX_AGE_S
    matches = client.matches_fresh(season, max_age)
    raw = matches.data or []
    finished = [m for m in raw if m.get("status") == "FINISHED"]
    phase_ids, full_time = [], {}
    if finished:
        rows = labels.match_rows(finished, season)
        phase_ids = rows.loc[rows["depth"] == 0, "match_id"].tolist()
        full_time = {str(m.get("id")): m.get("fullTimeAt") for m in finished}
    now = clock()

    def one(match_id: str):
        settled_for = _age_s(full_time.get(match_id), now) - config.LIVE_SETTLED_S
        # once settled, a copy younger than the time since settling was written after it, so it is final
        age_limit = settled_for if settled_for > 0 else max_age
        try:
            return match_id, client.team_match_stats_fresh(match_id, age_limit)
        except Exception:  # noqa: BLE001 - one unreachable match leaves a gap, not a failed snapshot
            return match_id, None

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = dict(pool.map(one, phase_ids))
    stats = {match_id: (r.data if r is not None else None) for match_id, r in results.items()}
    stale = matches.stale or any(r is None or r.stale for r in results.values())
    frame = live_frame(raw, stats, client.coefficients(season - 1), history, features, season)
    return LiveSnapshot(season, frame, stale, matches.fetched_at, len(phase_ids), clock())


class LiveService:
    """Owns the live snapshot. current() never blocks and never touches the network. Refreshes are single-flight:
    requests start them in the background (stale-while-revalidate), and the pipeline's live stage runs one and
    waits for it. After a failure the next automatic attempt waits LIVE_RETRY_S."""

    def __init__(self, build: Callable[[bool], LiveSnapshot], clock: Callable[[], float] = time.time,
                 max_age: float = config.LIVE_MAX_AGE_S, retry_after: float = config.LIVE_RETRY_S):
        self._build, self._clock = build, clock
        self._max_age, self._retry_after = max_age, retry_after
        self._lock = threading.Lock()  # guards the fields below
        self._flight = threading.Lock()  # one refresh at a time
        self._snapshot: LiveSnapshot | None = None
        self._status, self._message = "loading", ""
        self._failed_at: float | None = None
        self._pending = False  # a background refresh has been started and hasn't finished

    def current(self) -> LiveState:
        with self._lock:
            return LiveState(self._snapshot, self._status, self._message)

    def due(self) -> bool:
        with self._lock:
            snapshot, failed_at = self._snapshot, self._failed_at
        now = self._clock()
        if failed_at is not None and now - failed_at < self._retry_after:
            return False
        return snapshot is None or snapshot.stale or now - snapshot.built_at > self._max_age

    @property
    def refreshing(self) -> bool:
        with self._lock:
            return self._pending or self._flight.locked()

    def refresh(self, force: bool = False) -> LiveState:
        """Build a snapshot now and wait for it. A refresh already running is waited for, and then not repeated
        unless `force` (the live stage) asks for a fresh one anyway."""
        with self._flight:
            if not force and not self.due():
                return self.current()
            try:
                snapshot = self._build(force)
            except Exception as exc:  # noqa: BLE001 - keep serving what we have
                with self._lock:
                    self._failed_at = self._clock()
                    self._status = "stale" if self._snapshot is not None else "unavailable"
                    self._message = f"{type(exc).__name__}: {exc}"
                return self.current()
            with self._lock:
                self._snapshot = snapshot
                # a snapshot built from cached copies because UEFA was unreachable is retried like a failure
                self._failed_at = self._clock() if snapshot.stale else None
                self._status, self._message = ("stale" if snapshot.stale else "ready"), ""
            return self.current()

    def refresh_in_background(self) -> bool:
        """Start a refresh unless one is running, the snapshot is fresh, or the last failure was too recent."""
        if self._flight.locked() or not self.due():
            return False
        with self._lock:
            if self._pending:
                return False
            self._pending = True  # set before the thread starts, so a second call can't slip in

        def run() -> None:
            try:
                self.refresh()
            finally:
                with self._lock:
                    self._pending = False

        threading.Thread(target=run, daemon=True, name="live-refresh").start()
        return True
