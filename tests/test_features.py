import numpy as np
import pandas as pd
import pytest

from ucl import features


def stats(team_id, **values):
    return {"teamId": team_id, "statistics": [{"name": k, "value": str(v)} for k, v in values.items()]}


BASE = dict(goals=0, attempts_on_target=0, attempts_off_target=0, ball_possession=50, passes_attempted=400,
            passes_completed=320, passes_long_attempted=40, attacks=40, distance_covered=110,
            fouls_committed=10, saves=0)


def entry(*raw_stats):
    return {"teamId": "1", "statistics": list(raw_stats)}


def cleaned(name, value, attributes=None):
    """What _stat_values keeps for one raw stat: its per-match value, or None if it is dropped."""
    stat = {"name": name, "value": value}
    if attributes:
        stat["attributes"] = attributes
    return features._stat_values(entry(stat)).get(name)


def one_match(*entries):
    matches = pd.DataFrame([{"season": 2013, "match_id": "a1", "round": "Group stage", "depth": 0,
                             "home_id": "1", "home": "One", "away_id": "2", "away": "Two",
                             "home_goals": 1, "away_goals": 0}])
    return features.team_match_rows(matches, {"a1": list(entries)}).set_index("team_id")


@pytest.fixture
def two_matches():
    matches = pd.DataFrame([
        {"season": 2026, "match_id": "m1", "round": "League Phase", "depth": 0, "home_id": "1", "home": "One",
         "away_id": "2", "away": "Two", "home_goals": 2, "away_goals": 1},
        {"season": 2026, "match_id": "m2", "round": "League Phase", "depth": 0, "home_id": "3", "home": "Three",
         "away_id": "1", "away": "One", "home_goals": 0, "away_goals": 0},
        {"season": 2026, "match_id": "k1", "round": "Round of 16", "depth": 2, "home_id": "1", "home": "One",
         "away_id": "3", "away": "Three", "home_goals": 5, "away_goals": 0},
    ])
    stat_map = {
        "m1": [
            stats("1", **{**BASE, "goals": 2, "attempts_on_target": 5, "attempts_off_target": 5,
                          "ball_possession": 60, "passes_attempted": 500, "passes_completed": 450,
                          "passes_long_attempted": 50, "saves": 2}),
            stats("2", **{**BASE, "goals": 1, "attempts_on_target": 3, "attempts_off_target": 4}),
        ],
        # team 1's own stats are missing for m2; the opponent's are present
        "m2": [stats("3", **{**BASE, "attempts_on_target": 2, "attempts_off_target": 2})],
    }
    return matches, stat_map


def test_team_match_rows_use_only_phase_matches(two_matches):
    tm = features.team_match_rows(*two_matches)
    assert set(tm["match_id"]) == {"m1", "m2"}
    assert len(tm) == 4


def test_aliased_stats_team_id_is_matched_to_the_unmatched_side():
    # real case: the match feed has Steaua as 2614166, the stats feed as 50065 (FCSB)
    matches = pd.DataFrame([{"season": 2014, "match_id": "a1", "round": "Group stage", "depth": 0,
                             "home_id": "1", "home": "One", "away_id": "2614166", "away": "Steaua",
                             "home_goals": 1, "away_goals": 1}])
    stat_map = {"a1": [stats("1", **{**BASE, "attempts_on_target": 4}),
                       stats("50065", **{**BASE, "attempts_on_target": 6})]}
    tm = features.team_match_rows(matches, stat_map).set_index("team_id")
    assert tm.loc["2614166", "has_stats"]
    assert tm.loc["2614166", "attempts_on_target"] == 6
    assert tm.loc["1", "opp_attempts_on_target"] == 6


@pytest.mark.parametrize("value, attributes, expected", [
    ("1442", {"Fractional": "0.414"}, 41.4),   # value is seconds
    ("64", {"Fractional": "0.636"}, 63.6),     # the fraction is finer than the rounded percentage
    ("45", {"totalSeconds": "1488"}, 45),      # no fraction: value already is the percentage
    ("1442", {"Fractional": "0.414", "Percentage": "41", "TotalMinutesAndSeconds": "24'02\""}, 41.4),
    ("1442", {"Percentage": "41"}, 41),
    ("55", None, 55),
])
def test_possession_is_a_percentage(value, attributes, expected):
    assert cleaned("ball_possession", value, attributes) == pytest.approx(expected)


@pytest.mark.parametrize("value, attributes", [
    ("1442", None),                       # seconds, and nothing to convert them with
    ("1442", {"Fractional": "8.788"}),    # a fraction above 1
])
def test_possession_outside_0_to_100_is_dropped(value, attributes):
    assert cleaned("ball_possession", value, attributes) is None


@pytest.mark.parametrize("value, attributes, expected", [
    ("106837", {"DistanceKilometers": "106.83"}, 106.83),   # value is metres
    ("129.114", {"DistanceMeters": "129114"}, 129.114),
    ("121.39", {"meter": "121388"}, 121.388),
    ("106837", None, 106.837),                              # no attributes: metres, told apart by size
    ("112.5", None, 112.5),
    ("80", None, 80.0),                                     # the floor itself is kept
])
def test_distance_is_in_km(value, attributes, expected):
    assert cleaned("distance_covered", value, attributes) == pytest.approx(expected)


@pytest.mark.parametrize("value, attributes", [
    ("1.59", None),
    ("79.99", None),
    ("42.1", {"DistanceMeters": "42100"}),
])
def test_partial_tracking_distance_is_dropped(value, attributes):
    assert cleaned("distance_covered", value, attributes) is None


def test_zero_attacks_is_a_placeholder():
    assert cleaned("attacks", "0") is None
    assert cleaned("attacks", "23") == 23


def test_other_stats_are_taken_as_given():
    values = features._stat_values(entry(
        {"name": "fouls_committed", "value": "0"},   # zero is a real count for everything but attacks
        {"name": "goals", "value": "3"},
        {"name": "passing_distribution_by_delivery_zone", "value": "4,36,48"},   # unparsable: skipped
    ))
    assert values == {"fouls_committed": 0.0, "goals": 3.0}


def test_old_feeds_omit_zero_shot_counts():
    # the 2012-14 feeds leave out a shot count the team didn't have; nothing else is defaulted
    tm = one_match(stats("1", goals=1, saves=3),
                   stats("2", goals=0, attempts_on_target=4, attempts_off_target=2))
    assert tm.loc["1", ["attempts_on_target", "attempts_off_target"]].tolist() == [0.0, 0.0]
    assert tm.loc["1", "saves"] == 3 and np.isnan(tm.loc["1", "attacks"])
    assert np.isnan(tm.loc["2", "saves"])


def test_omitted_zero_shot_counts_also_reach_the_opponent_columns():
    # shots_against_pg must not drop the games where the other side had no shots
    tm = one_match(stats("1", goals=1), stats("2", goals=0, attempts_on_target=4, attempts_off_target=2))
    assert tm.loc["2", ["opp_attempts_on_target", "opp_attempts_off_target"]].tolist() == [0.0, 0.0]
    assert tm.loc["1", ["opp_attempts_on_target", "opp_attempts_off_target"]].tolist() == [4.0, 2.0]


def test_distance_only_payload_is_not_with_stats_but_keeps_its_distance():
    # real case: Sevilla v Mönchengladbach 2015-16 has only distance_covered and top_speed
    tm = one_match(stats("1", distance_covered=112.5, top_speed=33.1), stats("2", **BASE))
    assert not tm.loc["1", "has_stats"] and tm.loc["2", "has_stats"]
    assert tm.loc["1", "distance_covered"] == 112.5
    assert tm.loc["1", ["attempts_on_target", "attempts_off_target", "saves"]].isna().all()


def test_season_features_follow_missing_stat_rules(two_matches):
    tm = features.team_match_rows(*two_matches)
    row = features.season_features(tm).set_index("team_id").loc["1"]
    assert (row.n_matches, row.n_with_stats) == (2, 1)
    assert row.points_pg == pytest.approx(2.0)          # win + draw, from the score
    assert row.goal_diff_pg == pytest.approx(0.5)
    assert row.shots_pg == pytest.approx(10.0)          # only m1 has own stats
    assert row.shot_accuracy == pytest.approx(0.5)
    assert row.conversion == pytest.approx(0.2)
    assert row.shots_against_pg == pytest.approx(5.5)   # opponents: 7 in m1, 4 in m2
    assert row.on_target_against_pg == pytest.approx(2.5)
    assert row.save_pct == pytest.approx(2 / 3)         # m2 excluded: own saves missing
    assert row.pass_accuracy == pytest.approx(0.9)
    assert row.long_pass_share == pytest.approx(0.1)


def test_covered_features_drop_sparse_stats():
    tm = pd.DataFrame({
        "season": [2025] * 4 + [2026] * 4,
        "attacks": [1, 2, 3, 4, 1, np.nan, np.nan, 4],
        "fouls_committed": [1] * 8,
    })
    kept, worst = features.covered_features(tm, ["attacks_pg", "fouls_pg", "points_pg"])
    assert kept == ["fouls_pg", "points_pg"]
    assert worst["attacks_pg"] == pytest.approx(0.5)


def test_coefficients_shift_year_and_fill_debutants():
    coefs = features.coefficient_table({2025: [
        {"member": {"id": "1"}, "overallRanking": {"totalValue": 100.0}},
        {"member": {"id": "2"}, "overallRanking": {"totalValue": 50.0}},
    ]})
    ts = pd.DataFrame({"season": [2026, 2026, 2026], "team_id": ["1", "2", "3"]})
    out, rate = features.attach_coefficients(ts, coefs)
    out = out.set_index("team_id")
    assert out.loc["3", "coef"] == 50.0 and out.loc["3", "coef_imputed"]
    assert not out.loc["1", "coef_imputed"]
    assert out.loc["1", "coef_log"] == pytest.approx(np.log1p(100.0))
    assert rate == {2026: pytest.approx(2 / 3)}


def test_impute_uses_the_season_median():
    df = pd.DataFrame({"season": [1, 1, 1, 2], "save_pct": [0.5, np.nan, 0.7, 0.9]})
    out, n = features.impute_season_median(df, ["save_pct"])
    assert n == 1
    assert out.loc[1, "save_pct"] == pytest.approx(0.6)


def test_zscores_and_percentiles_are_within_season():
    df = pd.DataFrame({"season": [1, 1, 1, 2, 2, 2], "shots_pg": [10.0, 12.0, 14.0, 1.0, 2.0, 9.0]})
    z = features.zscore_within_season(df, ["shots_pg"])
    assert z.groupby(df["season"])["z_shots_pg"].mean().abs().max() < 1e-12
    pct = features.add_percentiles(df, ["shots_pg"])
    assert pct.loc[2, "pct_season_shots_pg"] == 100 and pct.loc[5, "pct_season_shots_pg"] == 100
    assert pct.loc[5, "pct_all_shots_pg"] == 50  # 9.0 is the 3rd smallest of 6


def test_display_value_formats():
    row = pd.Series({"coef": 148.0, "coef_log": 5.0, "save_pct": 0.7123, "passes_pg": 523.4})
    assert features.display_value(row, "coef_log") == 148.0
    assert features.display_value(row, "save_pct") == 71.2
    assert features.display_value(row, "passes_pg") == 523
