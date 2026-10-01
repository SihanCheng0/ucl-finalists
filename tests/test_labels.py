import math

import pandas as pd
import pytest

from ucl import labels


def raw_match(mid, round_name, home, away, hg, ag, winner=None, pens=None, reason="WIN_REGULAR", city="Budapest"):
    match = {
        "id": mid,
        "round": {"metaData": {"name": round_name}},
        "homeTeam": {"id": home, "internationalName": f"Team {home}"},
        "awayTeam": {"id": away, "internationalName": f"Team {away}"},
        "score": {"total": {"home": hg, "away": ag}},
        "stadium": {"city": {"translations": {"name": {"EN": city}}}},
    }
    if pens:
        match["score"]["penalty"] = {"home": pens[0], "away": pens[1]}
    if winner:
        match["winner"] = {"match": {"reason": reason, "team": {"id": winner}}}
    return match


def test_unknown_round_name_raises():
    with pytest.raises(ValueError, match="Play-in"):
        labels.match_rows([raw_match("1", "Play-in", "A", "B", 1, 0)], 2026)


def test_aliased_club_ids_are_canonical():
    # the match feed calls Steaua 2614166; the stats feed and the coefficient ranking call it 50065
    raw = [raw_match("1", "Group stage", "2614166", "7889", 1, 0, winner="2614166"),
           raw_match("2", "Group stage", "50051", "2614166", 0, 0)]
    rows = labels.match_rows(raw, 2015)
    assert rows[["home_id", "away_id"]].values.tolist() == [["50065", "7889"], ["50051", "50065"]]
    assert rows.loc[0, "winner_id"] == "50065"
    assert pd.isna(rows.loc[1, "winner_id"])
    assert labels.stages(rows).set_index("team_id").loc["50065", "team"] == "Team 2614166"


def test_penalty_final_2026_style():
    raw = raw_match("9", "Final", "52747", "52280", 1, 1, winner="52747", pens=(4, 3), reason="WIN_ON_PENALTIES")
    final = labels.finals_table(labels.match_rows([raw], 2026)).iloc[0]
    assert (final.winner_id, final.runner_up_id) == ("52747", "52280")
    assert (final.winner_goals, final.runner_up_goals, final.winner_pens, final.runner_up_pens) == (1, 1, 4, 3)
    assert final.city == "Budapest"


def test_draw_reason_final_with_away_winner_2012_style():
    raw = raw_match("8", "Final", "50037", "52914", 1, 1, winner="52914", pens=(3, 4), reason="DRAW")
    final = labels.finals_table(labels.match_rows([raw], 2012)).iloc[0]
    assert (final.winner_id, final.runner_up_id) == ("52914", "50037")
    assert (final.winner_goals, final.runner_up_goals, final.winner_pens, final.runner_up_pens) == (1, 1, 4, 3)


def test_regular_final_is_oriented_to_the_winner():
    # 2022 is stored as Liverpool 0-1 Real Madrid
    raw = raw_match("7", "Final", "7889", "50051", 0, 1, winner="50051")
    final = labels.finals_table(labels.match_rows([raw], 2022)).iloc[0]
    assert (final.winner_id, final.winner_goals, final.runner_up_goals) == ("50051", 1, 0)
    assert pd.isna(final.winner_pens)


def test_stages_league_format():
    raw = [
        raw_match("1", "League Phase", "A", "G", 1, 0),
        raw_match("2", "League Phase", "B", "C", 1, 0),
        raw_match("3", "League Phase", "D", "E", 1, 0),
        raw_match("4", "League Phase", "F", "A", 1, 0),
        raw_match("5", "Knock-out Play-off", "A", "B", 0, 1, winner="B"),
        raw_match("6", "Round of 16", "B", "C", 0, 1, winner="C"),
        raw_match("7", "Quarter-finals", "C", "D", 0, 1, winner="D"),
        raw_match("8", "Semi-finals", "D", "E", 0, 1, winner="E"),
        raw_match("9", "Final", "E", "F", 0, 1, winner="F"),
    ]
    s = labels.stages(labels.match_rows(raw, 2026)).set_index("team_id")
    assert not s.loc["G", "in_ko"]
    assert math.isnan(s.loc["G", "ko_stage"])
    assert s.loc["G", "stage_label"] == "League phase"
    assert (s.loc["A", "ko_stage"], s.loc["A", "stage_label"]) == (0, "Knockout play-off")
    assert (s.loc["B", "ko_stage"], s.loc["B", "stage_label"]) == (0, "Round of 16")
    assert (s.loc["C", "ko_stage"], s.loc["D", "ko_stage"]) == (1, 2)
    assert (s.loc["E", "ko_stage"], s.loc["E", "stage_label"]) == (3, "Runner-up")
    assert (s.loc["F", "ko_stage"], s.loc["F", "stage_label"]) == (4, "Winner")


def test_group_stage_label_before_2025():
    s = labels.stages(labels.match_rows([raw_match("1", "Group stage", "A", "B", 0, 0)], 2024))
    assert set(s["stage_label"]) == {"Group stage"}


def test_stages_group_format_ko_levels():
    raw = [
        raw_match("1", "Group stage", "A", "B", 1, 0),
        raw_match("2", "Group stage", "C", "D", 1, 0),
        raw_match("3", "Group stage", "E", "F", 1, 0),
        raw_match("4", "Group stage", "G", "A", 1, 0),
        raw_match("5", "Round of 16", "A", "B", 0, 1, winner="B"),
        raw_match("6", "Quarter-finals", "B", "C", 0, 1, winner="C"),
        raw_match("7", "Semi-finals", "C", "D", 0, 1, winner="D"),
        raw_match("8", "Final", "D", "E", 2, 1, winner="D"),
    ]
    s = labels.stages(labels.match_rows(raw, 2024)).set_index("team_id")["ko_stage"]
    assert s[["A", "B", "C", "D", "E"]].tolist() == [0, 1, 2, 4, 3]
    assert math.isnan(s["F"]) and math.isnan(s["G"])
