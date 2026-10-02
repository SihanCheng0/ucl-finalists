"""Title odds and head-to-head predictions.

Every Champions League match since 2011-12 moves an Elo rating per club, by its 90-minute result scaled by the goal
margin. At the start of each season a club's rating is pulled toward a prior set by its UEFA club coefficient, which
is also where a newcomer starts. A Poisson goals model turns a rating gap into expected goals, so the same ratings
give a match's win, draw and loss chances, a two-legged tie, and, by playing the rest of a season many times over,
each club's chance of winning the trophy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import config

BASE = 1500.0  # the rating of a club with an average coefficient for its season's field
MAX_GOALS = 10  # score matrices stop here; what lies beyond is spread back in by renormalising
STAGES = ["top8", "ko", "r16", "qf", "sf", "final", "win"]  # top 8 of a league phase, reach the knockouts, ..., win


@dataclass(frozen=True)
class Params:
    """Tuned on 2013-14 to 2018-19 by `scripts/tune_forecast.py` (the two seasons before warm the ratings up) and
    tested on 2019-20 onward."""
    k: float = 15.0  # rating points a narrow win moves, before scaling by surprise
    home: float = 50.0  # home advantage in rating points
    carry: float = 0.9  # share of a rating kept from one season to the next; the rest comes from the prior
    scale: float = 35.0  # rating points per standard deviation of log club coefficient


PARAMS = Params()
PEDIGREE = Params(k=0.0, home=100.0, carry=1.0, scale=100.0)  # coefficient only, tuned the same way: the baseline
TRAIN_SEASONS = range(2014, 2020)  # 2013-14 to 2018-19
TEST_SEASONS = range(2020, 2027)  # 2019-20 to 2025-26


# --- Fixtures -------------------------------------------------------------------------------------------------------

def _team(team: dict) -> str:
    return config.TEAM_ID_ALIASES.get(str(team.get("id")), str(team.get("id")))


def fixture_rows(raw: list[dict], season: int) -> pd.DataFrame:
    """Every match of a season, played or not, with its kick-off, leg and 90-minute and final scores."""
    rows = []
    for m in raw:
        round_name = m["round"]["metaData"]["name"]
        depth = config.ROUND_DEPTH[round_name]
        score = m.get("score") or {}
        regular, total, pens = (score.get(key) or {} for key in ("regular", "total", "penalty"))
        home, away = m["homeTeam"], m["awayTeam"]
        rows.append({
            "season": season,
            "match_id": str(m["id"]),
            "kickoff": (m.get("kickOffTime") or {}).get("dateTime") or "",
            "depth": depth,
            "leg": (m.get("leg") or {}).get("number") or 0,
            "home_id": _team(home),
            "home": home.get("internationalName"),
            "away_id": _team(away),
            "away": away.get("internationalName"),
            "known": not (home.get("isPlaceHolder") or away.get("isPlaceHolder")),
            "finished": m.get("status") == "FINISHED" and regular.get("home") is not None,
            "home_goals": regular.get("home"),
            "away_goals": regular.get("away"),
            "home_total": total.get("home"),
            "away_total": total.get("away"),
            "home_pens": pens.get("home"),
            "away_pens": pens.get("away"),
            # finals are at a neutral venue, and so were 2019-20's single-match rounds in Lisbon
            "neutral": depth == 5 or (season == 2020 and depth >= 3),
        })
    frame = pd.DataFrame(rows, columns=["season", "match_id", "kickoff", "depth", "leg", "home_id", "home", "away_id",
                                        "away", "known", "finished", "home_goals", "away_goals", "home_total",
                                        "away_total", "home_pens", "away_pens", "neutral"])
    return frame.sort_values(["kickoff", "match_id"], kind="stable").reset_index(drop=True)


def load_fixtures(client, seasons: list[int]) -> pd.DataFrame:
    """Every season's fixtures, from the client's cache where it has them."""
    return pd.concat([fixture_rows(client.matches(season), season) for season in seasons], ignore_index=True)


def field_priors(rows: pd.DataFrame, season: int) -> dict[tuple[int, str], float]:
    """A season's priors from its clubs' coefficients, z-scored across the whole field (the live season's rows only
    z-score the clubs that have played)."""
    log = pd.to_numeric(rows["coef_log"], errors="coerce")
    z = (log - log.mean()) / log.std(ddof=0)
    return {(season, str(team)): float(v) for team, v in zip(rows["team_id"], z) if pd.notna(v)}


def priors_from(team_seasons: pd.DataFrame) -> dict[tuple[int, str], float]:
    """(season, team id) -> the club's log coefficient as a z-score within that season's field."""
    return {(int(r.season), str(r.team_id)): float(r.z_coef_log) for r in team_seasons.itertuples()
            if pd.notna(r.z_coef_log)}


# --- Ratings --------------------------------------------------------------------------------------------------------

def expected_score(diff: float) -> float:
    """The home side's expected score (win 1, draw 0.5) for a rating gap that includes home advantage."""
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def margin_multiplier(goal_diff: int) -> float:
    """A bigger win moves ratings further, with diminishing returns (World Football Elo's scale)."""
    margin = abs(goal_diff)
    return 1.0 if margin <= 1 else 1.5 if margin == 2 else (11.0 + margin) / 8.0


@dataclass
class Ratings:
    current: dict[str, float]  # after the last match played
    season_end: dict[tuple[int, str], float]  # (season, team) -> after its last match that season
    history: pd.DataFrame  # one row per match played: the rating gap going in, and the 90-minute score


def rate(fixtures: pd.DataFrame, priors: dict[tuple[int, str], float], params: Params = PARAMS) -> Ratings:
    """Walk through every match in kick-off order. Each season starts by pulling its clubs toward their prior, once
    per season since the club last played, so a club returning after years away is mostly its prior again."""
    current: dict[str, float] = {}
    last_season: dict[str, int] = {}
    season_end: dict[tuple[int, str], float] = {}
    records = []
    for season in sorted(set(fixtures["season"])):
        games = fixtures[fixtures["season"] == season]
        field = {team: z for (s, team), z in priors.items() if s == season}
        floor = min(field.values()) if field else 0.0
        known = games[games["known"]]
        teams = set(field) | set(known["home_id"]) | set(known["away_id"])
        for team in teams:
            prior = BASE + params.scale * field.get(team, floor)
            if team in current:
                keep = params.carry ** (season - last_season[team])
                current[team] = keep * current[team] + (1 - keep) * prior
            else:
                current[team] = prior
            last_season[team] = season
        for g in games[games["finished"]].itertuples(index=False):
            diff = current[g.home_id] - current[g.away_id] + (0.0 if g.neutral else params.home)
            result = 1.0 if g.home_goals > g.away_goals else 0.5 if g.home_goals == g.away_goals else 0.0
            change = params.k * margin_multiplier(g.home_goals - g.away_goals) * (result - expected_score(diff))
            records.append((season, g.match_id, g.depth, diff, int(g.home_goals), int(g.away_goals)))
            current[g.home_id] += change
            current[g.away_id] -= change
        for team in teams:
            season_end[(season, team)] = current[team]
    history = pd.DataFrame(records, columns=["season", "match_id", "depth", "diff", "home_goals", "away_goals"])
    return Ratings(current, season_end, history)


# --- Goals ----------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Goals:
    """log(expected goals) = base ± slope × rating gap / 400, for the home and away side."""
    base: float
    slope: float

    def rates(self, diff):
        x = np.asarray(diff, dtype=float) / 400.0
        return np.exp(self.base + self.slope * x), np.exp(self.base - self.slope * x)


def fit_goals(history: pd.DataFrame) -> Goals:
    """Poisson maximum likelihood over 90-minute scores."""
    x = history["diff"].to_numpy(float) / 400.0
    home, away = history["home_goals"].to_numpy(float), history["away_goals"].to_numpy(float)

    def loss(p: np.ndarray) -> tuple[float, np.ndarray]:
        lh, la = p[0] + p[1] * x, p[0] - p[1] * x
        eh, ea = np.exp(lh), np.exp(la)
        value = np.sum(eh - home * lh) + np.sum(ea - away * la)
        grad = np.array([np.sum(eh - home) + np.sum(ea - away), np.sum((eh - home) * x) - np.sum((ea - away) * x)])
        return value, grad

    fit = minimize(loss, np.array([0.3, 0.5]), jac=True, method="BFGS")
    return Goals(float(fit.x[0]), float(fit.x[1]))


@dataclass(frozen=True)
class Models:
    """How a rating gap turns into goals, by stage. The deeper the round, the less a gap is worth: tested on seasons
    the models weren't fitted on, a separate model for each of these three stages forecasts better than one."""
    league: Goals  # group stage and league phase
    early: Goals  # knockout play-offs and round of 16
    late: Goals  # quarter-finals onward

    def at(self, depth: int) -> Goals:
        return self.league if depth == 0 else self.early if depth <= 2 else self.late


def fit_models(history: pd.DataFrame) -> Models:
    depth = history["depth"]
    return Models(fit_goals(history[depth == 0]), fit_goals(history[depth.isin([1, 2])]),
                  fit_goals(history[depth >= 3]))


def _poisson(rate: float, n: int = MAX_GOALS) -> np.ndarray:
    k = np.arange(n + 1)
    return np.exp(-rate + k * math.log(rate) - np.array([math.lgamma(i + 1) for i in k]))


def score_matrix(home_rate: float, away_rate: float) -> np.ndarray:
    """P(home scores i, away scores j), for i, j up to MAX_GOALS."""
    m = np.outer(_poisson(home_rate), _poisson(away_rate))
    return m / m.sum()


def outcome(m: np.ndarray) -> tuple[float, float, float]:
    """(home win, draw, away win) from a score matrix."""
    return float(np.tril(m, -1).sum()), float(np.trace(m)), float(np.triu(m, 1).sum())


def win_draw_loss(diff: float, goals: Goals) -> tuple[float, float, float]:
    return outcome(score_matrix(*map(float, goals.rates(diff))))


def outcome_probs(diff: np.ndarray, goals: Goals) -> np.ndarray:
    """(home win, draw, away win) per rating gap: an (n, 3) array."""
    rh, ra = goals.rates(np.asarray(diff, dtype=float))
    k = np.arange(MAX_GOALS + 1)
    log_fact = np.array([math.lgamma(i + 1) for i in k])
    ph = np.exp(-rh[:, None] + k * np.log(rh[:, None]) - log_fact)
    pa = np.exp(-ra[:, None] + k * np.log(ra[:, None]) - log_fact)
    m = ph[:, :, None] * pa[:, None, :]
    m /= m.sum(axis=(1, 2), keepdims=True)
    below = np.tril(np.ones((MAX_GOALS + 1, MAX_GOALS + 1)), -1).astype(bool)
    return np.stack([m[:, below].sum(1), np.trace(m, axis1=1, axis2=2), m[:, below.T].sum(1)], axis=1)


def match_metrics(history: pd.DataFrame, models: Models) -> dict[str, float]:
    """How good the 90-minute win/draw/loss forecasts were: log loss and ranked probability score (lower is better),
    and how often the likeliest outcome happened."""
    diff, depth = history["diff"].to_numpy(float), history["depth"].to_numpy()
    probs = np.zeros((len(history), 3))
    for stage, mask in ((models.league, depth == 0), (models.early, (depth == 1) | (depth == 2)),
                        (models.late, depth >= 3)):
        if mask.any():
            probs[mask] = outcome_probs(diff[mask], stage)
    home, away = history["home_goals"].to_numpy(), history["away_goals"].to_numpy()
    happened = np.where(home > away, 0, np.where(home == away, 1, 2))
    onehot = np.eye(3)[happened]
    rps = np.sum((np.cumsum(probs, axis=1)[:, :2] - np.cumsum(onehot, axis=1)[:, :2]) ** 2, axis=1) / 2
    return {
        "matches": int(len(history)),
        "log_loss": float(-np.mean(np.log(probs[np.arange(len(probs)), happened]))),
        "rps": float(np.mean(rps)),
        "accuracy": float(np.mean(probs.argmax(axis=1) == happened)),
        "draws_predicted": float(probs[:, 1].mean()),
        "draws_seen": float(np.mean(happened == 1)),
    }


# --- Head to head ---------------------------------------------------------------------------------------------------

def _margin(m: np.ndarray) -> np.ndarray:
    """Distribution of (first side's goals - second side's goals), indexed from -MAX_GOALS."""
    out = np.zeros(2 * MAX_GOALS + 1)
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            out[i - j + MAX_GOALS] += m[i, j]
    return out


def _extra_time_win(diff: float, goals: Goals) -> float:
    """Chance the first side wins after a level 90 (or 180) minutes: 30 more minutes, then a penalty coin flip."""
    rh, ra = goals.rates(diff)
    win, draw, _ = outcome(score_matrix(float(rh) / 3, float(ra) / 3))
    return win + draw / 2


def tie_probability(gap: float, goals: Goals, home: float = PARAMS.home, a_hosts_second: bool = True) -> float:
    """Chance A goes through a two-legged tie at today's rules (no away goals): 180 minutes, then extra time at the
    second leg's ground, then penalties. `gap` is A's rating minus B's."""
    first = score_matrix(*map(float, goals.rates(gap + (-home if a_hosts_second else home))))
    second = score_matrix(*map(float, goals.rates(gap + (home if a_hosts_second else -home))))
    total = np.convolve(_margin(first), _margin(second))  # A's aggregate margin, indexed from -2 * MAX_GOALS
    ahead, level = float(total[2 * MAX_GOALS + 1:].sum()), float(total[2 * MAX_GOALS])
    return ahead + level * _extra_time_win(gap + (home if a_hosts_second else -home), goals)


def head_to_head(rating_a: float, rating_b: float, models: Models, home: float = PARAMS.home,
                 venue: str = "neutral") -> dict:
    """A against B, three ways, each with its stage's goals model: one match like a league-phase game (at a neutral
    venue, or at A's or B's ground), a two-legged tie like the round of 16 (averaged over who hosts the second leg),
    and a final."""
    gap = rating_a - rating_b
    diff = gap + {"neutral": 0.0, "a": home, "b": -home}[venue]
    xg_a, xg_b = models.league.rates(diff)
    m = score_matrix(float(xg_a), float(xg_b))
    win, draw, loss = outcome(m)
    likeliest = np.argsort(m, axis=None)[::-1][:3]
    scores = [{"a": int(i), "b": int(j), "p": float(m[i, j])} for i, j in zip(*np.unravel_index(likeliest, m.shape))]
    final_win, final_draw, _ = outcome(score_matrix(*map(float, models.late.rates(gap))))
    return {
        "venue": venue,
        "win": win, "draw": draw, "loss": loss,
        "xg_a": float(xg_a), "xg_b": float(xg_b),
        "likely_scores": scores,
        "tie": (tie_probability(gap, models.early, home, True) + tie_probability(gap, models.early, home, False)) / 2,
        "final": final_win + final_draw * _extra_time_win(gap, models.late),
    }


# --- Simulation -----------------------------------------------------------------------------------------------------

class _Sim:
    """Plays matches for `n` simulated seasons at once. Team arguments are indices into `ratings`, either one per
    simulation (arrays of length n) or the same for all (ints)."""

    def __init__(self, ratings: np.ndarray, models: Models, home: float, n: int, rng: np.random.Generator):
        self.r, self.models, self.home, self.n, self.rng = ratings, models, home, n, rng

    def play(self, home_team, away_team, depth: int, neutral: bool = False, minutes: float = 90.0):
        diff = self.r[home_team] - self.r[away_team] + (0.0 if neutral else self.home)
        rh, ra = self.models.at(depth).rates(diff)
        size = None if np.ndim(diff) else self.n
        scale = minutes / 90.0
        return self.rng.poisson(rh * scale, size=size), self.rng.poisson(ra * scale, size=size)

    def _decide(self, a, b, a_goals, b_goals, depth: int, a_home: bool, neutral: bool, b_away_goals=None,
                a_away_goals=None):
        """Winner per simulation of a level tie: away goals (when they count), extra time, then penalties."""
        a, b = np.broadcast_to(a, (self.n,)), np.broadcast_to(b, (self.n,))
        winner = np.where(a_goals > b_goals, a, b)
        level = a_goals == b_goals
        if a_away_goals is not None:
            winner = np.where(level & (a_away_goals > b_away_goals), a, winner)
            winner = np.where(level & (a_away_goals < b_away_goals), b, winner)
            level = level & (a_away_goals == b_away_goals)
        if level.any():
            host, guest = (a, b) if a_home or neutral else (b, a)
            hg, gg = self.play(host, guest, depth, neutral=neutral, minutes=30)
            a_extra, b_extra = (hg, gg) if a_home or neutral else (gg, hg)
            winner = np.where(level & (a_extra > b_extra), a, winner)
            winner = np.where(level & (a_extra < b_extra), b, winner)
            if a_away_goals is not None:  # an away goal in extra time also counted double
                winner = np.where(level & (a_extra == b_extra) & (b_extra > 0), b, winner)
                level = level & (a_extra == b_extra) & (b_extra == 0)
            else:
                level = level & (a_extra == b_extra)
            winner = np.where(level & (self.rng.random(self.n) < 0.5), a, np.where(level, b, winner))
        return winner

    def tie(self, first_home, second_home, depth: int, away_goals: bool = False, legs: dict | None = None):
        """Winner of a two-legged tie; `second_home` hosts the second leg. `legs` fixes legs already played:
        {1: (first_home's goals, second_home's goals), 2: (second_home's goals, first_home's goals)}."""
        legs = legs or {}
        if 1 in legs:
            f1, s1 = (np.full(self.n, g) for g in legs[1])
        else:
            f1, s1 = self.play(first_home, second_home, depth)
        if 2 in legs:
            s2, f2 = (np.full(self.n, g) for g in legs[2])
        else:
            s2, f2 = self.play(second_home, first_home, depth)
        return self._decide(second_home, first_home, s1 + s2, f1 + f2, depth, a_home=True, neutral=False,
                            a_away_goals=s1 if away_goals else None, b_away_goals=f2 if away_goals else None)

    def single(self, a, b, depth: int, neutral: bool = True):
        ga, gb = self.play(a, b, depth, neutral=neutral)
        return self._decide(a, b, ga, gb, depth, a_home=not neutral, neutral=neutral)


def _known_ties(season_fx: pd.DataFrame, depth: int) -> list[dict] | None:
    """The round's ties when every one of them is drawn: teams, legs played so far, the winner if decided."""
    games = season_fx[(season_fx["depth"] == depth) & season_fx["known"]]
    if games.empty:
        return None
    ties: dict[frozenset, list] = {}
    for g in games.itertuples(index=False):
        ties.setdefault(frozenset((g.home_id, g.away_id)), []).append(g)
    expected = {1: 8, 2: 8, 3: 4, 4: 2, 5: 1}[depth]
    if len(ties) != expected:
        return None
    out = []
    for legs in ties.values():
        legs = sorted(legs, key=lambda g: (g.leg, g.kickoff))
        out.append({"legs": legs, "single": len(legs) == 1})
    return out


def _decided(tie: dict, away_goals: bool) -> str | None:
    """The winner of a tie whose last match has been played, from the final scores and penalties."""
    legs = tie["legs"]
    if not all(g.finished for g in legs):
        return None
    last = legs[-1]
    goals: dict[str, int] = {}
    away: dict[str, int] = {}
    for g in legs:
        goals[g.home_id] = goals.get(g.home_id, 0) + int(g.home_total)
        goals[g.away_id] = goals.get(g.away_id, 0) + int(g.away_total)
        away[g.away_id] = away.get(g.away_id, 0) + int(g.away_total)
    a, b = last.home_id, last.away_id
    if goals[a] != goals[b]:
        return a if goals[a] > goals[b] else b
    if away_goals and len(legs) == 2 and away.get(a, 0) != away.get(b, 0):
        return a if away.get(a, 0) > away.get(b, 0) else b
    if pd.notna(last.home_pens) and pd.notna(last.away_pens):
        return a if last.home_pens > last.away_pens else b
    return None


class Season:
    """The state of one season (its fixtures, played or not) and everything needed to play out the rest."""

    def __init__(self, fixtures: pd.DataFrame, ratings: dict[str, float], models: Models, params: Params = PARAMS):
        self.fx = fixtures
        self.season = int(fixtures["season"].iloc[0])
        self.away_goals = self.season <= 2021  # the away goals rule was dropped from 2021-22
        known = fixtures[fixtures["known"]]
        self.teams = sorted(set(known["home_id"]) | set(known["away_id"]))
        self.index = {team: i for i, team in enumerate(self.teams)}
        self.ratings = np.array([ratings[t] for t in self.teams])
        self.models, self.params = models, params

    def _ix(self, team: str) -> int:
        return self.index[team]

    def _play_tie(self, sim: _Sim, tie: dict) -> np.ndarray:
        depth = int(tie["legs"][0].depth)
        decided = _decided(tie, self.away_goals)
        if decided is not None:
            return np.full(sim.n, self._ix(decided))
        legs = tie["legs"]
        if tie["single"]:
            g = legs[0]
            return sim.single(self._ix(g.home_id), self._ix(g.away_id), depth, neutral=bool(g.neutral))
        first, second = legs[0], legs[-1]
        played = {}
        if first.finished:
            played[1] = (int(first.home_goals), int(first.away_goals))
        return sim.tie(self._ix(first.home_id), self._ix(first.away_id), depth, self.away_goals, played)

    def _league_table(self, sim: _Sim) -> np.ndarray:
        """League-phase ranks per simulation: (n, 36) team indices, best first."""
        league = self.fx[self.fx["depth"] == 0]
        teams = sorted(set(league["home_id"]) | set(league["away_id"]))
        col = {t: i for i, t in enumerate(teams)}
        tally = np.zeros((6, len(teams)))  # points, goal difference, goals, away goals, wins, away wins
        done = league[league["finished"]]
        for g in done.itertuples(index=False):
            for team, gf, ga, is_away in ((g.home_id, g.home_goals, g.away_goals, 0),
                                          (g.away_id, g.away_goals, g.home_goals, 1)):
                c = col[team]
                tally[:, c] += [3 * (gf > ga) + (gf == ga), gf - ga, gf, gf * is_away, gf > ga, (gf > ga) * is_away]
        todo = league[~league["finished"]]
        stats = np.broadcast_to(tally[:, None, :], (6, sim.n, len(teams))).copy()
        if len(todo):
            h = np.array([self._ix(t) for t in todo["home_id"]])
            a = np.array([self._ix(t) for t in todo["away_id"]])
            rh, ra = self.models.league.rates(self.ratings[h] - self.ratings[a] + self.params.home)
            gh = sim.rng.poisson(rh, size=(sim.n, len(todo)))
            ga = sim.rng.poisson(ra, size=(sim.n, len(todo)))
            home_onehot = np.zeros((len(todo), len(teams)))
            away_onehot = np.zeros((len(todo), len(teams)))
            home_onehot[np.arange(len(todo)), [col[t] for t in todo["home_id"]]] = 1
            away_onehot[np.arange(len(todo)), [col[t] for t in todo["away_id"]]] = 1
            hw, aw, dr = (gh > ga), (ga > gh), (gh == ga)
            stats[0] += (3 * hw + dr) @ home_onehot + (3 * aw + dr) @ away_onehot
            stats[1] += (gh - ga) @ home_onehot + (ga - gh) @ away_onehot
            stats[2] += gh @ home_onehot + ga @ away_onehot
            stats[3] += ga @ away_onehot
            stats[4] += hw @ home_onehot + aw @ away_onehot
            stats[5] += aw @ away_onehot
        key = np.zeros((sim.n, len(teams)))
        for row, width in zip(stats, (1e10, 1e7, 1e5, 1e3, 1e1, 1e0)):
            key += (row + (100 if width == 1e7 else 0)) * width
        key += sim.rng.random((sim.n, len(teams))) * 0.5  # whatever still ties is settled at random
        order = np.argsort(-key, axis=1)
        global_ix = np.array([self._ix(t) for t in teams])
        return global_ix[order]

    def simulate(self, n: int = 20000, seed: int = 0) -> pd.DataFrame:
        """Chance of reaching each stage, per club, from this season's state."""
        rng = np.random.default_rng(seed)
        sim = _Sim(self.ratings, self.models, self.params.home, n, rng)
        reached = {stage: np.zeros(len(self.teams)) for stage in STAGES}

        def mark(stage: str, teams: np.ndarray) -> None:
            np.add.at(reached[stage], teams.ravel(), 1)

        def mark_known(stage: str, ties: list[dict]) -> None:
            for team in {team for t in ties for g in t["legs"] for team in (g.home_id, g.away_id)}:
                reached[stage][self._ix(team)] += n

        if config.is_league_format(self.season):
            r16, by_seed = self._league_route(sim, mark, mark_known)
        else:
            r16, by_seed = self._group_route(sim, mark_known)
        qf_ties = _known_ties(self.fx, 3)
        if qf_ties is not None:
            mark_known("qf", qf_ties)
            qf_winners = [self._play_tie(sim, t) for t in qf_ties]
            pairs = self._random_pairs(sim, np.stack(qf_winners, axis=1))
        else:
            if by_seed is not None:  # league format: a fixed bracket from the round-of-16 seeds
                quarter = self._bracket_quarters(sim, by_seed)
            else:  # an open draw
                quarter = self._random_pairs(sim, r16)
            mark("qf", np.stack([np.stack(p, axis=1) for p in quarter], axis=1))
            qf_winners = [self._random_host_tie(sim, a, b, 3) for a, b in quarter]
            if by_seed is not None:  # winners of the 1/2-7/8 and 3/4-5/6 quarter-finals in each half meet
                pairs = [(qf_winners[0], qf_winners[1]), (qf_winners[2], qf_winners[3])]
            else:
                pairs = self._random_pairs(sim, np.stack(qf_winners, axis=1))
        sf_ties = _known_ties(self.fx, 4)
        if sf_ties is not None:
            mark_known("sf", sf_ties)
            finalists = [self._play_tie(sim, t) for t in sf_ties]
        else:
            mark("sf", np.stack([np.stack(p, axis=1) for p in pairs], axis=1))
            finalists = [self._random_host_tie(sim, a, b, 4) for a, b in pairs]
        final = _known_ties(self.fx, 5)
        if final is not None:
            mark_known("final", final)
            champion = self._play_tie(sim, final[0])
        else:
            mark("final", np.stack(finalists, axis=1))
            champion = sim.single(finalists[0], finalists[1], 5, neutral=True)
        mark("win", champion)
        out = pd.DataFrame({"team_id": self.teams, "rating": self.ratings})
        for stage in STAGES:
            out[f"p_{stage}"] = reached[stage] / n
        return out.sort_values(["p_win", "p_final", "rating"], ascending=False).reset_index(drop=True)

    def _random_host_tie(self, sim: _Sim, a: np.ndarray, b: np.ndarray, depth: int) -> np.ndarray:
        """A tie whose second-leg host comes from the draw, not the table: either side, at random."""
        flip = sim.rng.random(sim.n) < 0.5
        first, second = np.where(flip, a, b), np.where(flip, b, a)
        return sim.tie(first, second, depth, self.away_goals)

    def _random_pairs(self, sim: _Sim, teams: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """An open draw: shuffle each simulation's teams and pair them off in order."""
        shuffled = np.take_along_axis(teams, np.argsort(sim.rng.random(teams.shape), axis=1), axis=1)
        return [(shuffled[:, i], shuffled[:, i + 1]) for i in range(0, teams.shape[1], 2)]

    def _group_route(self, sim: _Sim, mark_known) -> tuple[np.ndarray, None]:
        """Group-stage seasons start at a drawn round of 16 (the group stage itself isn't simulated)."""
        ties = _known_ties(self.fx, 2)
        if ties is None:
            raise ValueError(f"season {self.season}: the round of 16 isn't drawn yet")
        mark_known("ko", ties)
        mark_known("r16", ties)
        return np.stack([self._play_tie(sim, t) for t in ties], axis=1), None

    def _league_route(self, sim: _Sim, mark, mark_known) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """League phase, knockout play-offs and round of 16, by the seeding rules of 2024-25 onward. Returns the
        round-of-16 winners and, keyed by league position 1-8, the winner of the tie that position seeded."""
        order = self._league_table(sim)  # (n, 36)
        mark("top8", order[:, :8])
        mark("ko", order[:, :24])
        playoff = _known_ties(self.fx, 1)
        position = self._final_positions()
        # play-off winners, grouped by seeding pair: (9/10 v 23/24), (11/12 v 21/22), (13/14 v 19/20), (15/16 v 17/18)
        groups: dict[int, list[np.ndarray]] = {g: [] for g in range(4)}
        if playoff is not None and position is not None:
            for t in playoff:
                seeded = min((t["legs"][0].home_id, t["legs"][0].away_id), key=lambda team: position[team])
                groups[(position[seeded] - 9) // 2].append(self._play_tie(sim, t))
        else:
            for g in range(4):
                seeds = order[:, [8 + 2 * g, 9 + 2 * g]]
                unseeded = order[:, [23 - 2 * g - 1, 23 - 2 * g]]
                flip = sim.rng.random(sim.n) < 0.5
                for s in range(2):
                    rival = np.where(flip, unseeded[:, s], unseeded[:, 1 - s])
                    groups[g].append(sim.tie(rival, seeds[:, s], 1))  # the better-placed team hosts the second leg
        r16_ties = _known_ties(self.fx, 2)
        by_seed: dict[int, np.ndarray] = {}
        if r16_ties is not None and position is not None:
            mark_known("r16", r16_ties)
            for t in r16_ties:
                seeded = min((t["legs"][0].home_id, t["legs"][0].away_id), key=lambda team: position[team])
                by_seed[position[seeded]] = self._play_tie(sim, t)
        else:
            top = order[:, :8]
            # positions 1/2 meet the 15/16-17/18 winners, 3/4 the 13/14-19/20, 5/6 the 11/12-21/22, 7/8 the 9/10-23/24
            entrants = []
            for pair, g in ((0, 3), (1, 2), (2, 1), (3, 0)):
                flip = sim.rng.random(sim.n) < 0.5
                for s in range(2):
                    rival = np.where(flip, groups[g][s], groups[g][1 - s])
                    seed = top[:, 2 * pair + s]
                    entrants.append(np.stack([seed, rival], axis=1))
                    by_seed[2 * pair + s + 1] = sim.tie(rival, seed, 2)
            mark("r16", np.concatenate(entrants, axis=1))
        r16 = np.stack([by_seed[p] for p in range(1, 9)], axis=1)
        return r16, by_seed

    def _bracket_quarters(self, sim: _Sim, by_seed: dict[int, np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        """Each half gets one of each seeding pair (1/2, 3/4, 5/6, 7/8): 1/2 meets 7/8 and 3/4 meets 5/6."""
        halves = []
        flips = {pair: sim.rng.random(sim.n) < 0.5 for pair in (1, 3, 5, 7)}
        for half in (0, 1):
            pick = {pair: np.where(flips[pair] ^ bool(half), by_seed[pair], by_seed[pair + 1]) for pair in (1, 3, 5, 7)}
            halves += [(pick[1], pick[7]), (pick[3], pick[5])]
        return halves

    def _final_positions(self) -> dict[str, int] | None:
        """Final league positions, once every league-phase match is played."""
        league = self.fx[self.fx["depth"] == 0]
        if league.empty or not league["finished"].all():
            return None
        sim = _Sim(self.ratings, self.models, self.params.home, 1, np.random.default_rng(0))
        order = self._league_table(sim)[0]
        return {self.teams[i]: rank + 1 for rank, i in enumerate(order)}


# --- Live season ----------------------------------------------------------------------------------------------------

def league_table(season_fx: pd.DataFrame) -> pd.DataFrame:
    """The league phase so far: played, points and goal difference per club."""
    league = season_fx[(season_fx["depth"] == 0) & season_fx["known"]]
    teams = sorted(set(league["home_id"]) | set(league["away_id"]))
    table = {t: {"played": 0, "points": 0, "goal_diff": 0, "goals": 0} for t in teams}
    for g in league[league["finished"]].itertuples(index=False):
        for team, gf, ga in ((g.home_id, g.home_goals, g.away_goals), (g.away_id, g.away_goals, g.home_goals)):
            row = table[team]
            row["played"] += 1
            row["points"] += 3 if gf > ga else 1 if gf == ga else 0
            row["goal_diff"] += int(gf - ga)
            row["goals"] += int(gf)
    return pd.DataFrame([{"team_id": t, **v} for t, v in table.items()])


def as_of(fixtures: pd.DataFrame, season: int, depth: int, drawn: bool = True, before: str | None = None) -> pd.DataFrame:
    """A past season as it stood before round `depth` was played: earlier rounds as played, round `depth` drawn but
    unplayed (or, with drawn=False, not drawn yet) and later rounds not drawn. With `before` (an ISO time), league-phase
    matches from then on count as unplayed too. Earlier seasons are kept whole, for the ratings."""
    past = fixtures[fixtures["season"] < season]
    keep = fixtures["depth"] <= depth if drawn else fixtures["depth"] < depth
    now = fixtures[(fixtures["season"] == season) & keep].copy()
    unplayed = now["depth"] == depth
    if before is not None:
        unplayed |= (now["depth"] == 0) & (now["kickoff"] >= before)
    now.loc[unplayed, "finished"] = False
    for column in ("home_goals", "away_goals", "home_total", "away_total", "home_pens", "away_pens"):
        now.loc[unplayed, column] = None
    return pd.concat([past, now], ignore_index=True)


# --- Backtests ------------------------------------------------------------------------------------------------------

def champion(fixtures: pd.DataFrame, season: int) -> str | None:
    final = _known_ties(fixtures[fixtures["season"] == season], 5)
    return _decided(final[0], season <= 2021) if final else None


def title_backtest(fixtures: pd.DataFrame, priors: dict[tuple[int, str], float], models: Models, seasons,
                   params: Params = PARAMS, n: int = 10000, before_matchday: int | None = None) -> pd.DataFrame:
    """Each season's title odds as they stood when the knockouts began: group-stage seasons once the round of 16 was
    drawn, league-phase seasons once the league phase ended. With `before_matchday`, league-phase seasons are taken
    from earlier still: before that matchday was played."""
    rows = []
    for season in seasons:
        if config.is_league_format(season):
            before = None
            if before_matchday is not None:
                league = fixtures[(fixtures["season"] == season) & (fixtures["depth"] == 0)].sort_values("kickoff")
                before = league["kickoff"].iloc[(before_matchday - 1) * 18]
            cut = as_of(fixtures, season, 1, drawn=False, before=before)
        elif before_matchday is not None:
            continue
        else:
            cut = as_of(fixtures, season, 2)
        state = Season(cut[cut["season"] == season], rate(cut, priors, params).current, models, params)
        odds = state.simulate(n, seed=season)
        winner = champion(fixtures, season)
        hit = odds.index[odds["team_id"] == winner]
        rows.append({
            "season": season,
            "winner_id": winner,
            "p_winner": float(odds.loc[hit, "p_win"].iloc[0]) if len(hit) else 0.0,
            "winner_rank": int(hit[0]) + 1 if len(hit) else None,
            "favourite_id": odds["team_id"].iloc[0],
            "p_favourite": float(odds["p_win"].iloc[0]),
            "teams_left": int((odds["p_ko"] > 0).sum()),
        })
    return pd.DataFrame(rows)


def tie_calibration(fixtures: pd.DataFrame, history: pd.DataFrame, models: Models, seasons) -> dict:
    """Two-legged ties: how often the favourite went through, against how often the model said it would."""
    gap_of = history.set_index("match_id")["diff"]
    predicted, happened = [], []
    for season in seasons:
        season_fx = fixtures[fixtures["season"] == season]
        for depth in (1, 2, 3, 4):
            for tie in _known_ties(season_fx, depth) or []:
                if len(tie["legs"]) != 2 or tie["legs"][0].match_id not in gap_of.index:
                    continue
                first, second = tie["legs"]
                gap = -(gap_of[first.match_id] - PARAMS.home)  # the second leg's host minus its guest, going in
                p_host = tie_probability(gap, models.at(depth), PARAMS.home, a_hosts_second=True)
                winner = _decided(tie, season <= 2021)
                predicted.append(max(p_host, 1 - p_host))
                happened.append(winner == (second.home_id if p_host >= 0.5 else first.home_id))
    return {"ties": len(predicted), "predicted": float(np.mean(predicted)), "happened": float(np.mean(happened))}


def track_record(fixtures: pd.DataFrame, priors: dict[tuple[int, str], float], params: Params = PARAMS,
                 n: int = 10000) -> dict:
    """How the forecasts did on the seasons the settings weren't tuned on, with goals models fitted before them."""
    history = rate(fixtures, priors, params).history
    models = fit_models(history[history["season"].isin(TRAIN_SEASONS)])
    test = history[history["season"].isin(TEST_SEASONS)]
    pedigree = rate(fixtures, priors, PEDIGREE).history
    pedigree_models = fit_models(pedigree[pedigree["season"].isin(TRAIN_SEASONS)])
    train = history[history["season"].isin(TRAIN_SEASONS)]
    outcome_of = lambda h: np.where(h["home_goals"] > h["away_goals"], 0,  # noqa: E731
                                    np.where(h["home_goals"] == h["away_goals"], 1, 2))
    base_rates = np.bincount(outcome_of(train), minlength=3) / len(train)
    titles = title_backtest(fixtures, priors, models, TEST_SEASONS, params, n)
    return {
        "seasons": [config.season_label(s) for s in (TEST_SEASONS.start, TEST_SEASONS.stop - 1)],
        "matches": {**match_metrics(test, models),
                    "pedigree_log_loss": match_metrics(pedigree[pedigree["season"].isin(TEST_SEASONS)],
                                                       pedigree_models)["log_loss"],
                    "base_rate_log_loss": float(-np.mean(np.log(base_rates[outcome_of(test)])))},
        "ties": tie_calibration(fixtures, history, models, TEST_SEASONS),
        "titles": titles,
    }
