"""Title odds for the live season and head-to-heads between any two team-seasons (forecast.py), rebuilt when the
dataset or the live season changes. Builds take about a second and are shared by every request until then."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .. import config, forecast
from .jsonsafe import to_jsonable

SIMULATIONS = 20000
VENUES = ("neutral", "a", "b")


class Unavailable(Exception):
    """The forecast can't be made yet: no dataset, or no live season for title odds."""


class NotFound(Exception):
    pass


@dataclass(frozen=True)
class Built:
    ratings: forecast.Ratings
    models: forecast.Models
    names: dict[str, str]
    odds: pd.DataFrame | None  # the live season's chances per club, with its league-phase table
    live_fixtures: pd.DataFrame | None


class ForecastService:
    def __init__(self, load_fixtures: Callable[[list[int]], pd.DataFrame], simulations: int = SIMULATIONS,
                 record_simulations: int = 10000):
        self._load_fixtures = load_fixtures
        self._simulations, self._record_simulations = simulations, record_simulations
        self._lock = threading.Lock()
        self._history: tuple[object, pd.DataFrame] | None = None
        self._built: tuple[tuple, Built] | None = None
        self._record: tuple[object, dict] | None = None

    # --- building ---

    def _historical(self, snapshot) -> pd.DataFrame:
        if self._history is None or self._history[0] != snapshot.built_at:
            self._history = (snapshot.built_at, self._load_fixtures(config.SEASONS))
        return self._history[1]

    def _build(self, snapshot, live_snapshot) -> Built:
        if snapshot.dataset is None:
            raise Unavailable("The dataset isn't built yet: run the pipeline first")
        key = (snapshot.built_at, None if live_snapshot is None else live_snapshot.built_at)
        with self._lock:
            if self._built is not None and self._built[0] == key:
                return self._built[1]
            fixtures = self._historical(snapshot)
            priors = forecast.priors_from(snapshot.dataset.team_seasons)
            names = dict(zip(snapshot.dataset.team_seasons["team_id"].astype(str),
                             snapshot.dataset.team_seasons["team_display"]))
            live_fx = None
            if live_snapshot is not None and getattr(live_snapshot, "fixtures", None) is not None \
                    and len(live_snapshot.fixtures):
                live_fx = live_snapshot.fixtures
                fixtures = pd.concat([fixtures, live_fx], ignore_index=True)
                priors.update(forecast.field_priors(live_snapshot.rows, live_snapshot.season))
                names.update(zip(live_snapshot.rows["team_id"].astype(str), live_snapshot.rows["team_display"]))
            ratings = forecast.rate(fixtures, priors)
            models = forecast.fit_models(ratings.history)
            odds = None
            if live_fx is not None:
                season = forecast.Season(live_fx, ratings.current, models)
                odds = season.simulate(self._simulations, seed=0).merge(
                    forecast.league_table(live_fx), on="team_id", how="left")
            built = Built(ratings, models, names, odds, live_fx)
            self._built = (key, built)
            return built

    def _track_record(self, snapshot) -> dict:
        with self._lock:
            if self._record is None or self._record[0] != snapshot.built_at:
                record = forecast.track_record(self._historical(snapshot),
                                               forecast.priors_from(snapshot.dataset.team_seasons),
                                               n=self._record_simulations)
                self._record = (snapshot.built_at, record)
            return self._record[1]

    # --- views ---

    def title_odds(self, snapshot, live_state) -> dict:
        live_snapshot = live_state.snapshot if live_state is not None else None
        built = self._build(snapshot, live_snapshot)
        if built.odds is None:
            message = live_state.message if live_state is not None and live_state.message else ""
            raise Unavailable("The live season isn't loaded yet" + (f": {message}" if message else ""))
        fx = built.live_fixtures
        league = fx[fx["depth"] == 0]
        played = league[league["finished"]]
        teams = [{
            "team_id": r.team_id, "name": built.names.get(r.team_id, r.team_id), "rating": round(float(r.rating)),
            "played": int(r.played), "points": int(r.points), "goal_diff": int(r.goal_diff),
            **{f"p_{stage}": float(getattr(r, f"p_{stage}")) for stage in forecast.STAGES},
        } for r in built.odds.itertuples(index=False)]
        record = self._track_record(snapshot)
        names = built.names
        titles = [{**row, "label": config.season_label(int(row["season"])),
                   "winner": names.get(row["winner_id"], row["winner_id"]),
                   "favourite": names.get(row["favourite_id"], row["favourite_id"])}
                  for row in record["titles"].to_dict("records")]
        return to_jsonable({
            "season": int(live_snapshot.season), "label": config.season_label(int(live_snapshot.season)),
            "played": int(len(played)), "league_matches": int(len(league)),
            "as_of": played["kickoff"].max() if len(played) else None,
            "fetched_at": live_snapshot.fetched_at, "stale": live_state.status == "stale",
            "simulations": self._simulations, "teams": teams,
            "track_record": {**{k: v for k, v in record.items() if k != "titles"}, "titles": titles},
        })

    @staticmethod
    def _title(built: Built) -> dict[str, float]:
        return {} if built.odds is None else dict(zip(built.odds["team_id"], built.odds["p_win"]))

    @staticmethod
    def _side(built: Built, title: dict[str, float], team_id: str, season: int) -> dict:
        rating = built.ratings.season_end.get((season, team_id))
        if rating is None:
            raise NotFound(f"no rating for team {team_id} in {config.season_label(season)}")
        return {"team_id": team_id, "name": built.names.get(team_id, team_id), "season": season,
                "label": config.season_label(season), "live": season == config.LIVE_SEASON,
                "rating": round(rating), "title": title.get(team_id)}

    @staticmethod
    def _title_label(live_snapshot) -> str | None:
        return None if live_snapshot is None else config.season_label(int(live_snapshot.season))

    def head_to_head(self, snapshot, live_state, a: tuple[str, int], b: tuple[str, int], venue: str) -> dict:
        if venue not in VENUES:
            raise ValueError(f"venue must be one of {', '.join(VENUES)}, not {venue!r}")
        live_snapshot = live_state.snapshot if live_state is not None else None
        built = self._build(snapshot, live_snapshot)
        title = self._title(built)
        side_a, side_b = self._side(built, title, *a), self._side(built, title, *b)
        result = forecast.head_to_head(built.ratings.season_end[(a[1], a[0])], built.ratings.season_end[(b[1], b[0])],
                                       built.models, forecast.PARAMS.home, venue)
        return to_jsonable({"a": side_a, "b": side_b, "title_label": self._title_label(live_snapshot), **result})

    def matchups(self, snapshot, live_state) -> dict:
        """What any head-to-head needs, for every rated team-season at once: each side as head_to_head shows it plus
        its exact rating, the three goals models and the home advantage. The website plays head-to-heads in the
        browser from this (web/src/lib/h2h.ts)."""
        live_snapshot = live_state.snapshot if live_state is not None else None
        built = self._build(snapshot, live_snapshot)
        title = self._title(built)
        sides = {f"{team_id}:{season}": {**self._side(built, title, team_id, season), "exact": float(rating)}
                 for (season, team_id), rating in sorted(built.ratings.season_end.items())}
        models = {stage: {"base": goals.base, "slope": goals.slope} for stage, goals in
                  (("league", built.models.league), ("early", built.models.early), ("late", built.models.late))}
        return to_jsonable({"home": forecast.PARAMS.home, "max_goals": forecast.MAX_GOALS, "models": models,
                            "title_label": self._title_label(live_snapshot), "sides": sides})
