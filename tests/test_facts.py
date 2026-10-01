import json
import math

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


QUALITY = "Model quality (each season scored by a model trained on the other seasons)"


def test_synthesis_quality_gives_intervals_the_skill_and_the_chance_share(built):
    ds, res = built
    m = res.metrics
    quality = facts.build_facts(ds.team_seasons, ds.finals, res)["synthesis"][QUALITY]
    assert quality["Intervals"] == "95% intervals from resampling seasons"
    # each interval is rounded like its point value; the top-4 share is in percent
    ci = m["ci"]
    assert quality["95% interval for the mean Spearman correlation"] == [round(x, 2) for x in ci["spearman_mean"]]
    assert quality["95% interval for the share of finalists in the model's top 4 (%)"] == [
        round(100 * x, 1) for x in ci["finalists_in_top4"]]
    assert quality["95% interval for the AUC"] == [round(x, 2) for x in ci["auc"]]
    assert quality["95% interval for the Brier skill"] == [round(x, 2) for x in ci["brier_skill"]]
    assert quality["Brier skill vs the base-rate guess (0 = no better, 1 = perfect)"] == round(m["brier_skill"], 2)
    assert quality["Share of finalists a random ranking would put in the top 4 (%)"] == round(
        100 * m["finalists_in_top4_chance"], 1)
    # the point values are still there
    assert quality["AUC for reaching the final"] == round(m["auc"], 2)
    assert quality["Brier score of the base-rate guess"] == round(m["brier_base_rate"], 3)


def test_robustness_rule_defines_all_three_labels_and_which_stats_are_labelled(built):
    ds, res = built
    synth = facts.build_facts(ds.team_seasons, ds.finals, res)["synthesis"]
    rule = synth["Robustness rule"]
    needed = math.ceil(config.ROBUST_SHARE * res.metrics["n_seasons"])
    for label in ("robust", "conditional", "model-dependent"):
        assert f"{label} = " in rule
    assert f"at least {needed} of {res.metrics['n_seasons']} seasons" in rule
    assert f"Only the top {config.TOP_DRIVERS} stats are labelled" in rule
    assert "on its own" in rule and "other stats held fixed" in rule  # what conditional means


def test_driver_entries_carry_the_direction_each_stat_has_on_its_own(built):
    ds, res = built
    synth = facts.build_facts(ds.team_seasons, ds.finals, res)["synthesis"]
    drivers = synth["What drives deep runs, most important first"]
    assert len(drivers) == 8
    for entry, row in zip(drivers, res.drivers.head(8).itertuples()):
        assert entry["direction on its own (Spearman with knockout stage)"] == round(row.marginal_rho, 2)
        assert {"stat", "group", "direction", "robustness label"} <= set(entry)  # the existing keys stay


def test_feature_set_rows_carry_their_intervals(built):
    ds, res = built
    rows = facts.build_facts(ds.team_seasons, ds.finals, res)["synthesis"]["Feature-set comparison (Spearman / AUC)"]
    assert len(rows) == len(res.ablation)
    for entry, row in zip(rows, res.ablation.itertuples()):
        assert entry["Spearman 95% interval"] == [round(row.spearman_lo, 2), round(row.spearman_hi, 2)]
        assert entry["AUC 95% interval"] == [round(row.auc_lo, 2), round(row.auc_hi, 2)]
        assert entry["Spearman"] == round(row.spearman, 2)


def test_vs_no_skill_says_whether_an_interval_reaches_the_no_skill_value():
    # the local model judged "30.0 to 60.0 includes 23.9" in nearly every sample, so the facts say it outright
    edge = "edge over a random ranking"
    assert facts.vs_no_skill([-0.04, 0.13], 0, "edge over the base-rate guess") == (
        "Yes: the edge over the base-rate guess is inconclusive")
    assert facts.vs_no_skill([30.0, 60.0], 23.9, edge) == "No: the whole interval is above it"
    assert facts.vs_no_skill([-0.2, -0.05], 0, edge) == "No: the whole interval is below it"
    assert facts.vs_no_skill([0.0, 0.1], 0, edge).startswith("Yes")  # touching the value counts as including it
    assert facts.vs_no_skill([0.1, 0.3], 0.3, edge).startswith("Yes")


def test_synthesis_quality_says_whether_each_edge_interval_includes_its_no_skill_value(built):
    ds, res = built
    quality = facts.build_facts(ds.team_seasons, ds.finals, res)["synthesis"][QUALITY]
    skill_lo, skill_hi = quality["95% interval for the Brier skill"]
    top4 = quality["95% interval for the share of finalists in the model's top 4 (%)"]
    chance = quality["Share of finalists a random ranking would put in the top 4 (%)"]
    assert quality["Does the Brier skill interval include 0 (no skill)?"] == facts.vs_no_skill(
        [skill_lo, skill_hi], 0, "edge over the base-rate guess")
    assert quality["Does the top-4 share interval include the random-ranking share?"] == facts.vs_no_skill(
        top4, chance, "edge over a random ranking")
