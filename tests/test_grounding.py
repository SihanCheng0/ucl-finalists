import numpy as np
import pytest

from ucl.grounding import check_grounding, fact_numbers


def test_percent_matches_a_fraction_or_a_percent():
    assert check_grounding("They had a 31% chance.", {"p": 0.31}) == []
    assert check_grounding("They had a 31% chance.", {"p": 31}) == []
    # only a number written as a percentage is scaled; a bare one must match as it is
    assert check_grounding("A rate of 0.31 per game.", {"p": 31}) == ["0.31"]
    assert check_grounding("31 per game", {"p": 0.31}) == ["31"]


def test_the_word_percent_marks_a_percentage_too():
    assert check_grounding("a 31 percent chance", {"p": 0.31}) == []
    assert check_grounding("a 31 percentile rank", {"p": 0.31}) == ["31"]


def test_a_scaled_percentage_keeps_the_text_precision():
    assert check_grounding("a 31.4% chance", {"p": 0.314}) == []
    assert check_grounding("a 31% chance", {"p": 0.314}) == []
    assert check_grounding("a 31.5% chance", {"p": 0.314}) == ["31.5"]


def test_rounding_tolerance_follows_the_text_precision():
    assert check_grounding("17.4 shots a game", {"shots": 18.43, "other": 17.38}) == []
    assert check_grounding("18.4 shots", {"shots": 18.43}) == []
    assert check_grounding("18.5 shots", {"shots": 18.45}) == []


def test_invented_numbers_are_flagged():
    assert check_grounding("AUC of 0.88", {"auc": 0.78}) == ["0.88"]
    assert check_grounding("a 47% chance", {"p": 0.31}) == ["47"]


def test_years_seasons_and_small_integers_are_ignored():
    text = "In 2025-26 and 2025–26 (and 2024) they made the top 3 after 2025/26, as in 2011 and 2027."
    assert check_grounding(text, {}) == []


def test_numbers_just_outside_the_year_range_are_checked():
    assert check_grounding("2010 and 2028", {}) == ["2010", "2028"]


def test_numbers_inside_fact_strings_and_keys_count():
    facts = {"Actual result": "Winner: beat Inter 5-0 in the final", "Teams since 2011-12 (488)": 1}
    assert check_grounding("one of 488 teams", facts) == []
    assert 488.0 in fact_numbers(facts)


def test_season_labels_in_fact_strings_and_keys_donate_no_numbers():
    assert check_grounding("12 clean sheets", {"label": "since 2011-12"}) == ["12"]
    assert fact_numbers({"Teams since 2011-12": "from 2012/13"}) == []


@pytest.mark.parametrize("dash", ["\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2015", "\u2212"])
def test_season_labels_with_any_dash_are_ignored(dash):
    assert check_grounding(f"in 2023{dash}24 they won", {}) == []


def test_numpy_numbers_count_as_facts():
    assert check_grounding("ranked 24th", {"n": np.int64(24)}) == []


def test_signs_are_ignored_when_matching():
    assert check_grounding("down by -0.42 and up +1.83", {"a": 0.42, "b": -1.83}) == []


def test_each_unsupported_number_is_reported_once():
    assert check_grounding("47 and 47 again", {}) == ["47"]
