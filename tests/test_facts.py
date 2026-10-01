import json

from ucl import facts


def test_result_text_handles_penalties_and_orientation():
    final = {"winner_id": "52747", "winner": "Paris", "runner_up_id": "52280", "runner_up": "Arsenal",
             "winner_goals": 1, "runner_up_goals": 1, "winner_pens": 4.0, "runner_up_pens": 3.0}
    assert facts.result_text("52747", final) == (
        "Winner: beat Arsenal on penalties in the final (1-1, 4-3 on penalties)")
    assert facts.result_text("52280", final) == (
        "Runner-up: lost the final to Paris Saint-Germain on penalties (1-1, 3-4 on penalties)")


def test_team_sheets_use_plain_labels_and_mark_lower_is_better(built):
    ds, res = built
    sheets = facts.build_facts(ds.team_seasons, ds.finals, res)
    team_keys = [k for k in sheets if k != "synthesis"]
    assert len(team_keys) == 10
    sheet = sheets[team_keys[0]]
    assert "Model probability of reaching the final (%)" in sheet
    stats = sheet["All league/group-phase stats"]
    assert "Opponent shots per game (lower is better)" in stats
    assert "Possession (%)" in stats
    json.dumps(sheets, allow_nan=False)  # raises if any NaN slipped into the facts


def test_synthesis_facts_include_context_counts_and_thresholds(built):
    ds, res = built
    synth = facts.build_facts(ds.team_seasons, ds.finals, res)["synthesis"]
    assert synth["Context"]["Finals"] == 6 and synth["Context"]["Finalists"] == 12
    assert len(synth["The 10 finalists"]) == 10
    assert synth["Significance threshold for Holm-adjusted p"] == 0.05
    assert all("number of stats" in row for row in synth["Feature-set comparison (Spearman / AUC)"])
