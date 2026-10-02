import numpy as np
import pandas as pd
import pytest

from ucl import config, forecast
from ucl.forecast import Goals, Models, Params

COLUMNS = ["season", "match_id", "kickoff", "depth", "leg", "home_id", "home", "away_id", "away", "known", "finished",
           "home_goals", "away_goals", "home_total", "away_total", "home_pens", "away_pens", "neutral"]
MODELS = Models(Goals(0.3, 1.4), Goals(0.35, 1.1), Goals(0.35, 0.75))


def match(season, depth, home, away, score=None, *, leg=0, day=1, pens=None, total=None, neutral=None, n=[0]):
    n[0] += 1
    done = score is not None
    total = total or score
    return {"season": season, "match_id": str(n[0]), "kickoff": f"{season - 1}-09-{day:02d}T20:00:00Z",
            "depth": depth, "leg": leg, "home_id": home, "home": home, "away_id": away, "away": away, "known": True,
            "finished": done, "home_goals": score[0] if done else None, "away_goals": score[1] if done else None,
            "home_total": total[0] if done else None, "away_total": total[1] if done else None,
            "home_pens": pens[0] if pens else None, "away_pens": pens[1] if pens else None,
            "neutral": depth == 5 if neutral is None else neutral}


def frame(rows):
    return pd.DataFrame(rows, columns=COLUMNS)


def league_fixtures(season=2027, teams=36, played=None):
    """A 36-team league phase: everyone plays 4 at home and 4 away. `played` maps (home, away) to a score."""
    ids = [f"t{i:02d}" for i in range(teams)]
    rows = []
    for offset in range(1, 5):
        for i, home in enumerate(ids):
            away = ids[(i + offset) % teams]
            rows.append(match(season, 0, home, away, (played or {}).get((home, away)), day=offset))
    return frame(rows), ids


# --- fixtures ---

def raw_match(status="FINISHED", round_name="Round of 16", leg=1, placeholder=False, home_id=50051):
    return {"id": 9, "status": status, "round": {"metaData": {"name": round_name}},
            "kickOffTime": {"dateTime": "2026-02-17T20:00:00Z"}, "leg": {"number": leg} if leg else None,
            "homeTeam": {"id": home_id, "internationalName": "Real Madrid", "isPlaceHolder": placeholder},
            "awayTeam": {"id": 52747, "internationalName": "Paris"},
            "score": {"regular": {"home": 1, "away": 1}, "total": {"home": 2, "away": 1}} if status == "FINISHED" else None}


def test_fixture_rows_keep_played_and_unplayed_matches_with_their_leg_and_venue():
    rows = forecast.fixture_rows([raw_match(), raw_match(status="UPCOMING", round_name="Final", leg=None)], 2026)
    played, final = rows.iloc[0], rows.iloc[1]
    assert (played["leg"], played["home_goals"], played["home_total"], played["finished"]) == (1, 1, 2, True)
    assert not played["neutral"] and final["neutral"] and not final["finished"] and pd.isna(final["home_goals"])
    placeholder = forecast.fixture_rows([raw_match(status="UPCOMING", placeholder=True)], 2026).iloc[0]
    assert not placeholder["known"]


def test_the_match_feeds_aliases_apply(monkeypatch):
    monkeypatch.setitem(config.TEAM_ID_ALIASES, "999", "50051")
    assert forecast.fixture_rows([raw_match(home_id=999)], 2026).iloc[0]["home_id"] == "50051"


# --- ratings ---

def test_expected_scores_are_symmetric_and_bigger_wins_count_for_more():
    assert forecast.expected_score(120) + forecast.expected_score(-120) == pytest.approx(1)
    assert [forecast.margin_multiplier(gd) for gd in (0, 1, -1, 2, 3, -5)] == [1, 1, 1, 1.5, 14 / 8, 16 / 8]


def test_ratings_move_by_the_surprise_and_sum_to_zero_within_a_match():
    fx = frame([match(2020, 0, "a", "b", (2, 0))])
    priors = {(2020, "a"): 0.0, (2020, "b"): 0.0}
    params = Params(k=20, home=0, carry=1.0, scale=100)
    r = forecast.rate(fx, priors, params)
    assert r.current["a"] == pytest.approx(1500 + 20 * 1.5 * 0.5)  # an even game, won by two
    assert r.current["a"] + r.current["b"] == pytest.approx(3000)
    assert r.history.iloc[0]["diff"] == 0


def test_each_season_pulls_ratings_toward_the_coefficient_prior_once_per_season_away():
    fx = frame([match(2020, 0, "a", "b", (3, 0)), match(2021, 0, "b", "c", None), match(2023, 0, "a", "c", None)])
    priors = {(2020, "a"): 0.0, (2020, "b"): 0.0, (2021, "b"): 1.0, (2021, "c"): -1.0, (2023, "a"): 2.0,
              (2023, "c"): -1.0}
    params = Params(k=40, home=0, carry=0.5, scale=100)
    r = forecast.rate(fx, priors, params)
    after_2020 = r.season_end[(2020, "a")]
    assert r.season_end[(2021, "c")] == 1400  # a newcomer starts at its prior
    assert r.season_end[(2021, "b")] == pytest.approx(0.5 * r.season_end[(2020, "b")] + 0.5 * 1600)
    # back in 2022-23 after missing two seasons: three season starts since it last played, an eighth is left
    assert r.season_end[(2023, "a")] == pytest.approx(0.125 * after_2020 + 0.875 * 1700)


def test_a_cut_off_never_sees_a_later_seasons_prior():
    fx = frame([match(2020, 0, "a", "b", (1, 0))])
    priors = {(2020, "a"): 0.0, (2020, "b"): 0.0, (2021, "a"): 3.0}
    r = forecast.rate(fx, priors)
    assert (2021, "a") not in r.season_end and r.current["a"] < 1600


def test_field_priors_z_score_the_whole_field():
    rows = pd.DataFrame({"team_id": ["a", "b", "c"], "coef_log": [1.0, 2.0, 3.0]})
    z = forecast.field_priors(rows, 2027)
    assert z[(2027, "b")] == 0 and z[(2027, "c")] == pytest.approx(-z[(2027, "a")]) == pytest.approx(1.2247, abs=1e-4)


# --- goals ---

def test_the_goals_model_recovers_its_parameters_from_simulated_scores():
    rng = np.random.default_rng(3)
    diff = rng.normal(0, 200, 20000)
    x = diff / 400
    history = pd.DataFrame({"diff": diff, "home_goals": rng.poisson(np.exp(0.3 + 1.2 * x)),
                            "away_goals": rng.poisson(np.exp(0.3 - 1.2 * x)), "depth": 0})
    fit = forecast.fit_goals(history)
    assert fit.base == pytest.approx(0.3, abs=0.02) and fit.slope == pytest.approx(1.2, abs=0.05)


def test_outcome_probabilities_add_up_and_favour_the_stronger_side():
    m = forecast.score_matrix(1.6, 1.1)
    assert m.sum() == pytest.approx(1)
    win, draw, loss = forecast.outcome(m)
    assert win + draw + loss == pytest.approx(1) and win > loss
    even = forecast.win_draw_loss(0, Goals(0.3, 1.0))
    assert even[0] == pytest.approx(even[2])
    assert forecast.outcome_probs(np.array([0.0, 200.0]), Goals(0.3, 1.0))[0] == pytest.approx(even)


def test_models_are_chosen_by_stage():
    assert MODELS.at(0) is MODELS.league and MODELS.at(1) is MODELS.at(2) is MODELS.early
    assert MODELS.at(3) is MODELS.at(5) is MODELS.late


def test_match_metrics_score_each_stage_with_its_own_model():
    history = pd.DataFrame({"diff": [0.0, 0.0], "home_goals": [1, 0], "away_goals": [0, 0], "depth": [0, 4]})
    metrics = forecast.match_metrics(history, MODELS)
    p_win = forecast.win_draw_loss(0, MODELS.league)[0]
    p_draw = forecast.win_draw_loss(0, MODELS.late)[1]
    assert metrics["log_loss"] == pytest.approx(-(np.log(p_win) + np.log(p_draw)) / 2)
    assert (metrics["matches"], metrics["draws_seen"]) == (2, 0.5)


# --- head to head ---

def test_head_to_head_is_symmetric_and_the_venue_matters():
    ab = forecast.head_to_head(1650, 1580, MODELS)
    ba = forecast.head_to_head(1580, 1650, MODELS)
    assert ab["win"] == pytest.approx(ba["loss"]) and ab["draw"] == pytest.approx(ba["draw"])
    assert ab["tie"] + ba["tie"] == pytest.approx(1) and ab["final"] + ba["final"] == pytest.approx(1)
    assert ab["win"] > ab["loss"] and ab["tie"] > 0.5 and ab["final"] > 0.5
    assert forecast.head_to_head(1650, 1580, MODELS, venue="a")["win"] > ab["win"] > \
        forecast.head_to_head(1650, 1580, MODELS, venue="b")["win"]
    assert len(ab["likely_scores"]) == 3 and ab["likely_scores"][0]["p"] >= ab["likely_scores"][1]["p"]


def test_hosting_the_second_leg_helps_only_through_extra_time():
    goals = MODELS.early
    host, guest = forecast.tie_probability(0, goals, a_hosts_second=True), forecast.tie_probability(0, goals, a_hosts_second=False)
    assert host > 0.5 and host + guest == pytest.approx(1)


# --- ties ---

def tie(*legs):
    rows = frame(list(legs))
    return {"legs": list(rows.itertuples(index=False)), "single": len(legs) == 1}


def test_a_played_tie_is_decided_by_aggregate_then_away_goals_then_penalties():
    assert forecast._decided(tie(match(2025, 2, "a", "b", (2, 0), leg=1), match(2025, 2, "b", "a", (1, 0), leg=2)),
                             False) == "a"
    level = (match(2020, 2, "a", "b", (2, 1), leg=1), match(2020, 2, "b", "a", (1, 0), leg=2))
    assert forecast._decided(tie(*level), away_goals=True) == "b"  # 2-2, but b scored away
    shootout = (match(2025, 2, "a", "b", (1, 0), leg=1), match(2025, 2, "b", "a", (1, 0), leg=2, pens=(5, 4)))
    assert forecast._decided(tie(*shootout), away_goals=False) == "b"
    assert forecast._decided(tie(match(2025, 2, "a", "b", (1, 0), leg=1), match(2025, 2, "b", "a", leg=2)), False) is None


def test_a_simulated_tie_respects_a_first_leg_already_played():
    sim = forecast._Sim(np.array([1500.0, 1500.0]), MODELS, 50, 4000, np.random.default_rng(1))
    winners = sim.tie(0, 1, 2, legs={1: (4, 0)})  # team 0 won the first leg 4-0 at home
    assert (winners == 0).mean() > 0.97
    even = sim.tie(0, 1, 2)
    assert 0.44 < (even == 1).mean() < 0.60  # equal teams: a coin flip, with a small edge to the second-leg host


# --- seasons ---

def ratings_for(ids, top=1700.0, step=8.0):
    return {team: top - step * i for i, team in enumerate(ids)}


def test_a_league_phase_season_hands_out_exactly_one_title_and_the_right_number_of_places():
    fx, ids = league_fixtures()
    odds = forecast.Season(fx, ratings_for(ids), MODELS).simulate(n=3000, seed=4)
    totals = {stage: odds[f"p_{stage}"].sum() for stage in forecast.STAGES}
    assert totals == pytest.approx({"top8": 8, "ko": 24, "r16": 16, "qf": 8, "sf": 4, "final": 2, "win": 1})
    columns = [f"p_{stage}" for stage in ["ko", "r16", "qf", "sf", "final", "win"]]
    assert (np.diff(odds[columns].to_numpy(), axis=1) <= 1e-12).all()  # each stage is harder to reach than the last
    assert odds.iloc[0]["team_id"] == "t00"  # the strongest club is the favourite


def test_once_the_league_phase_is_played_the_table_decides_who_goes_through():
    fx, ids = league_fixtures()
    goals = {team: 35 - i for i, team in enumerate(ids)}  # every club scores its own number, so no two finish level
    fx["finished"] = True
    fx["home_goals"] = fx["home_total"] = fx["home_id"].map(goals)
    fx["away_goals"] = fx["away_total"] = fx["away_id"].map(goals)
    odds = forecast.Season(fx, ratings_for(ids), MODELS).simulate(n=500, seed=1).set_index("team_id")
    assert set(odds["p_ko"].round(9)) <= {0.0, 1.0} and odds["p_ko"].sum() == 24
    assert set(odds["p_top8"].round(9)) <= {0.0, 1.0}


def test_the_league_format_bracket_keeps_each_seeding_pair_apart_until_the_final():
    fx, ids = league_fixtures()
    season = forecast.Season(fx, ratings_for(ids), MODELS)
    sim = forecast._Sim(season.ratings, MODELS, 50, 200, np.random.default_rng(0))
    by_seed = {p: np.full(200, p) for p in range(1, 9)}  # the "winner" of each tie is its seed, to read the bracket
    quarters = season._bracket_quarters(sim, by_seed)
    for half in (quarters[:2], quarters[2:]):
        pairs = [tuple(sorted((int(a[0]), int(b[0])))) for a, b in half]
        assert {pairs[0][0], pairs[0][1]} & {1, 2} and {pairs[0][0], pairs[0][1]} & {7, 8}
        assert {pairs[1][0], pairs[1][1]} & {3, 4} and {pairs[1][0], pairs[1][1]} & {5, 6}
    first_half = {int(x[0]) for a, b in quarters[:2] for x in (a, b)}
    assert len(first_half) == 4 and all(len(first_half & {p, p + 1}) == 1 for p in (1, 3, 5, 7))


def test_a_finished_season_is_certain():
    rows = []
    teams = [f"k{i}" for i in range(16)]
    for i in range(0, 16, 2):
        rows += [match(2019, 2, teams[i + 1], teams[i], (0, 1), leg=1), match(2019, 2, teams[i], teams[i + 1], (1, 0), leg=2)]
    for i in range(0, 16, 4):
        rows += [match(2019, 3, teams[i + 2], teams[i], (0, 1), leg=1), match(2019, 3, teams[i], teams[i + 2], (1, 0), leg=2)]
    for i in range(0, 16, 8):
        rows += [match(2019, 4, teams[i + 4], teams[i], (0, 1), leg=1), match(2019, 4, teams[i], teams[i + 4], (1, 0), leg=2)]
    rows.append(match(2019, 5, "k8", "k0", (1, 1), pens=(3, 4)))
    fx = frame(rows)
    odds = forecast.Season(fx, {t: 1500.0 for t in teams}, MODELS).simulate(n=200).set_index("team_id")
    assert odds.loc["k0", "p_win"] == 1 and odds.loc["k8", "p_final"] == 1 and odds["p_win"].sum() == 1
    assert forecast.champion(fx, 2019) == "k0"


def test_a_group_stage_season_starts_from_its_drawn_round_of_16():
    teams = [f"g{i}" for i in range(16)]
    rows = [match(2019, 2, teams[i + 1], teams[i], leg=1) for i in range(0, 16, 2)]
    rows += [match(2019, 2, teams[i], teams[i + 1], leg=2) for i in range(0, 16, 2)]
    odds = forecast.Season(frame(rows), ratings_for(teams), MODELS).simulate(n=2000, seed=2)
    assert odds["p_r16"].sum() == pytest.approx(16) and odds["p_qf"].sum() == pytest.approx(8)
    assert odds["p_win"].sum() == pytest.approx(1) and odds["p_top8"].sum() == 0
    with pytest.raises(ValueError, match="isn't drawn yet"):
        forecast.Season(frame(rows[:3]), ratings_for(teams), MODELS).simulate(n=10)


def test_as_of_hides_what_was_not_known_yet():
    rows = [match(2018, 0, "a", "b", (1, 0)), match(2019, 0, "a", "b", (2, 2), day=1),
            match(2019, 0, "b", "a", (0, 1), day=9), match(2019, 2, "a", "b", (1, 0), leg=1, day=20),
            match(2019, 3, "a", "c", (1, 0), leg=1, day=28)]
    fx = frame(rows)
    cut = forecast.as_of(fx, 2019, 2)
    assert set(cut["season"]) == {2018, 2019} and (cut["depth"] <= 2).all()
    r16 = cut[cut["depth"] == 2].iloc[0]
    assert not r16["finished"] and pd.isna(r16["home_goals"])
    early = forecast.as_of(fx, 2019, 1, drawn=False, before="2018-09-05")
    assert list(early.loc[early["season"] == 2019, "finished"]) == [True, False]
