import json

from ucl import config, facts


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


def test_better_than_shares_are_flipped_for_lower_is_better_stats(built):
    ds, res = built
    sheets = facts.build_facts(ds.team_seasons, ds.finals, res)
    key = next(k for k in sheets if k != "synthesis")
    season, team_id = key.split("-", 1)
    row = ds.team_seasons.set_index(["season", "team_id"]).loc[(int(season), team_id)]
    stats = sheets[key]["All league/group-phase stats"]
    for feature in ds.features:
        raw = int(row[f"pct_season_{feature}"])
        expected = 100 - raw if feature in config.LOWER_IS_BETTER else raw
        assert stats[facts.stat_label(feature)][facts.BEATS_SEASON] == expected
    assert any(f in config.LOWER_IS_BETTER for f in ds.features)


def test_strongest_and_weakest_stats_are_picked_by_teams_beaten(built):
    ds, res = built
    sheets = facts.build_facts(ds.team_seasons, ds.finals, res)
    for key in (k for k in sheets if k != "synthesis"):
        sheet = sheets[key]
        shares = {label: s[facts.BEATS_SEASON] for label, s in sheet["All league/group-phase stats"].items()}
        strongest = [shares[d["stat"]] for d in sheet[facts.STRONGEST]]
        assert strongest == sorted(shares.values(), reverse=True)[:3]
        weakest = sheet[facts.WEAKEST]
        below_half = sorted(v for v in shares.values() if v < 50)
        if below_half:
            assert [d[facts.BEATS_SEASON] for d in weakest] == below_half[:3]
        else:
            assert weakest == facts.NO_WEAK_STAT


def test_nearest_stage_rounds_the_expected_stage_to_the_scale():
    assert facts.nearest_stage(0.38) == "out before quarter-finals"
    assert facts.nearest_stage(0.87) == "quarter-finals"
    assert facts.nearest_stage(1.52) == "semi-finals"
    assert facts.nearest_stage(1.82) == "semi-finals"
    assert facts.nearest_stage(2.06) == "semi-finals"
    assert facts.nearest_stage(3.4) == "lost final"
    assert facts.nearest_stage(4.2) == "won final"


def test_pick_verdict_counts_only_the_top_two(built):
    assert facts.pick_verdict(1).startswith("Yes") and facts.pick_verdict(2).startswith("Yes")
    assert "leading contender" in facts.pick_verdict(3) and facts.pick_verdict(4).startswith("No")
    assert "leading contender" not in facts.pick_verdict(5) and facts.pick_verdict(8).startswith("No: ranked 8")
    ds, res = built
    sheet = next(v for k, v in facts.build_facts(ds.team_seasons, ds.finals, res).items() if k != "synthesis")
    assert sheet["Would the model have picked them?"] == facts.pick_verdict(
        sheet["Rank by that probability among the season's knockout teams"])


def test_synthesis_facts_include_context_counts_and_thresholds(built):
    ds, res = built
    synth = facts.build_facts(ds.team_seasons, ds.finals, res)["synthesis"]
    assert synth["Context"]["Finals"] == 6 and synth["Context"]["Finalists"] == 12
    assert len(synth["The 10 finalists"]) == 10
    assert synth["Significance threshold for Holm-adjusted p"] == 0.05
    assert all("number of stats" in row for row in synth["Feature-set comparison (Spearman / AUC)"])
