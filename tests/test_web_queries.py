import json

import pandas as pd
import pytest

from ucl import analyst, config, dataset, facts, model
from ucl.analyst import Analysis, Narrative
from ucl.facts import build_facts
from ucl.web import queries
from ucl.web.store import DataStore


def test_every_feature_has_exactly_one_section():
    placed = [f for features in config.STAT_SECTIONS.values() for f in features]
    assert sorted(placed) == sorted(config.FEATURES) and len(placed) == len(set(placed))


def test_feature_meta_says_how_to_show_a_stat():
    assert queries.feature_meta("possession_pct") == {
        "feature": "possession_pct", "label": "Possession (%)", "section": "Control", "group": "style",
        "decimals": 1, "percent": True, "lower_is_better": False}
    assert queries.feature_meta("fouls_pg")["lower_is_better"] is True
    assert queries.feature_meta("conversion")["percent"] is True and queries.feature_meta("coef_log")["percent"] is False


def teams(*rows):
    return pd.DataFrame([{"season": s, "team_id": t, "team": n, "team_display": d, "stage_label": stage}
                         for s, t, n, d, stage in rows])


TS = teams(
    (2026, "52747", "Paris", "Paris Saint-Germain", "Winner"),
    (2025, "52747", "Paris", "Paris Saint-Germain", "Winner"),
    (2026, "50124", "Atleti", "Atleti", "Round of 16"),
    (2024, "50051", "Real Madrid", "Real Madrid", "Winner"),
    (2026, "999", "Surreal FC", "Surreal FC", "League phase"),
    (2024, "52758", "B. Dortmund", "Borussia Dortmund", "Runner-up"),
    (2026, "59324", "Bodø/Glimt", "Bodø/Glimt", "Knockout play-off"),
)


def ids(results):
    return [r["team_id"] for r in results]


def test_short_queries_find_nothing():
    assert queries.search(TS, "") == [] and queries.search(TS, " p ") == []


def test_an_alias_finds_a_team_with_its_seasons_newest_first():
    assert queries.search(TS, "PSG") == [{"team_id": "52747", "name": "Paris Saint-Germain", "seasons": [
        {"season": 2026, "label": "2025-26", "stage_label": "Winner"},
        {"season": 2025, "label": "2024-25", "stage_label": "Winner"}]}]


def test_matching_ignores_case_accents_and_punctuation():
    assert ids(queries.search(TS, "atlético")) == ["50124"]  # through the alias "atletico madrid"
    assert ids(queries.search(TS, "b dortmund")) == ["52758"]  # UEFA's "B. Dortmund"
    assert ids(queries.search(TS, "DORTMUND")) == ["52758"]  # inside "Borussia Dortmund"
    assert ids(queries.search(TS, "bodo")) == ["59324"]  # ø doesn't decompose, so it is mapped by hand


def test_prefix_matches_rank_before_matches_inside_a_name():
    assert ids(queries.search(TS, "real")) == ["50051", "999"]  # Real Madrid first, though Surreal FC is newer
    assert ids(queries.search(TS, "real", limit=1)) == ["50051"]


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory, built):
    """The synthetic outputs on disk, with narratives that exercise every badge."""
    ds, results = built
    root = tmp_path_factory.mktemp("outputs")
    processed, out = root / "processed", root / "out"
    dataset.save(ds, processed)
    model.save(results, out)
    loaded = dataset.load(processed)
    sheets = build_facts(loaded.team_seasons, loaded.finals, model.load(out))
    narratives = {k: Narrative(k, "ok", text="## How they got there\nThey beat **everyone**.") for k in sheets}
    narratives["2026-52747"] = Narrative("2026-52747", "ok", text="## How they got there\nA 99.9 figure.",
                                         unsupported=["99.9"])
    narratives["2025-52747"] = Narrative("2025-52747", "unavailable", reason="empty answer")
    sheets["2024-50051"] = {"Club": "numbers from an earlier run"}
    analyst.save(Analysis("ok", "m", narratives, sheets), out / "analysis.json")
    return DataStore(processed, out).snapshot


def test_meta_lists_the_datasets_seasons_features_and_stages(snapshot):
    meta = queries.meta(snapshot)
    assert meta["ready"] is True and meta["problems"] == []
    assert [s["season"] for s in meta["seasons"]] == [2021, 2022, 2023, 2024, 2025, 2026]
    assert meta["seasons"][0] == {"season": 2021, "label": "2020-21", "live": False}
    assert [f["feature"] for f in meta["features"]] == snapshot.dataset.features
    assert [s["name"] for s in meta["stages"]] == ["fetch", "build", "model", "analyze", "report"]
    assert meta["stages"][0] == {"name": "fetch", "label": "Fetch", "optional": False}
    assert (meta["live_season"], meta["live_status"], meta["player_stats"]) == (None, "unavailable", [])
    assert meta["data"]["built_at"].startswith("20") and meta["data"]["modelled_at"].startswith("20")


def test_meta_before_anything_is_built(tmp_path):
    meta = queries.meta(DataStore(tmp_path / "processed", tmp_path / "out").snapshot)
    assert meta["ready"] is False and meta["features"] == []
    assert meta["problems"] == ["processed data missing: run build", "model outputs missing: run model"]
    assert [s["season"] for s in meta["seasons"]] == config.SEASONS
    json.dumps(meta, allow_nan=False)


def row_of(snapshot, team_id, season):
    ts = snapshot.dataset.team_seasons
    return ts[(ts["team_id"] == team_id) & (ts["season"] == season)].iloc[0]


def test_profile_header_result_and_seasons_played(snapshot):
    p = queries.profile(snapshot, "52280", 2026)
    assert (p["team_id"], p["name"], p["season"], p["label"]) == ("52280", "Team 52280", 2026, "2025-26")
    assert (p["ko_stage"], p["matches_played"], p["live"], p["live_available"]) == (3, 8, False, False)
    assert (p["stale"], p["fetched_at"]) == (False, None)
    assert p["result"] == "Runner-up: lost the final to Paris Saint-Germain 1-2"
    assert p["seasons"] == [{"season": 2026, "label": "2025-26"}]
    json.dumps(p, allow_nan=False)


def test_profile_stats_are_display_values_with_teams_beaten(snapshot):
    p = queries.profile(snapshot, "52280", 2026)
    row = row_of(snapshot, "52280", 2026)
    stats = {s["feature"]: s for s in p["features"]}
    assert list(stats) == snapshot.dataset.features
    assert stats["fouls_pg"]["value"] == round(row["fouls_pg"], 1) and stats["fouls_pg"]["section"] == "Intensity"
    assert stats["fouls_pg"]["beats_season"] == 100 - int(row["pct_season_fouls_pg"])  # lower is better: flipped
    assert stats["points_pg"]["beats_all"] == int(row["pct_all_points_pg"])
    assert stats["shot_accuracy"]["value"] == round(row["shot_accuracy"] * 100, 1)  # stored as a fraction
    assert stats["coef_log"]["value"] == round(row["coef"], 1)  # shown as coefficient points
    assert stats["points_pg"]["z"] == pytest.approx(row["z_points_pg"]) and stats["points_pg"]["provisional"] is False


def test_profile_trend_covers_every_season_the_team_played(snapshot):
    p = queries.profile(snapshot, "52747", 2026)
    assert [point["season"] for point in p["trend"]["points_pg"]] == [2025, 2026]
    assert p["trend"]["points_pg"][0]["live"] is False
    assert p["seasons"] == [{"season": 2025, "label": "2024-25"}, {"season": 2026, "label": "2025-26"}]


def test_profile_model_card_comes_from_the_predictions_and_shap(snapshot):
    card = queries.profile(snapshot, "52280", 2026)["model"]
    preds = snapshot.results.predictions
    pred = preds[(preds["season"] == 2026) & (preds["team_id"] == "52280")].iloc[0]
    assert (card["rank"], card["ko_size"]) == (int(pred["rank_in_season"]), 24)
    assert card["p_final"] == pytest.approx(pred["p_final"]) and card["base_rate"] == pytest.approx(2 / 24)
    assert card["nearest_stage"] in facts.STAGES
    assert 0 < len(card["top_up"]) <= 4 and len(card["top_down"]) <= 3
    assert all(c["contribution"] > 0 for c in card["top_up"]) and all(c["contribution"] < 0 for c in card["top_down"])


def test_a_team_outside_the_knockouts_has_no_model_card_or_narrative(snapshot):
    ts = snapshot.dataset.team_seasons
    out = ts[(ts["season"] == 2026) & ~ts["in_ko"].astype(bool)].iloc[0]
    p = queries.profile(snapshot, out["team_id"], 2026)
    assert p["model"] is None and p["narrative"] is None and p["ko_stage"] is None and p["result"] is None


def test_narrative_badges_cover_good_warn_unavailable_and_stale(snapshot):
    def badge(team_id, season):
        return queries.profile(snapshot, team_id, season)["narrative"]["badge"]

    assert badge("52280", 2026) == {"kind": "good", "label": "All figures found in the data"}
    assert badge("52747", 2026) == {"kind": "warn", "label": "1 figure not found in the data"}
    assert badge("52747", 2025) == {"kind": "unavailable", "label": "AI write-up unavailable"}
    assert badge("50051", 2024) == {"kind": "stale", "label": "Written for earlier numbers"}
    narrative = queries.profile(snapshot, "52280", 2026)["narrative"]
    assert "<strong>everyone</strong>" in narrative["html"] and narrative["text"].startswith("## How")
    assert queries.profile(snapshot, "52747", 2025)["narrative"]["html"] is None


def test_an_unknown_team_season_is_not_found(snapshot):
    with pytest.raises(queries.NotFound):
        queries.profile(snapshot, "52280", 2021)
    with pytest.raises(queries.NotFound):
        queries.profile(snapshot, "nope", 2026)


def test_parse_pick():
    assert queries.parse_pick("52280:2026") == ("52280", 2026)
    for bad in ("", "52280", "52280:", ":2026", "52280:20x6", "52280:²"):
        with pytest.raises(ValueError):
            queries.parse_pick(bad)


def test_compare_orients_z_so_that_positive_is_always_better(snapshot):
    out = queries.compare(snapshot, ("52280", 2026), ("52747", 2026))
    rows = {r["feature"]: r for r in out["rows"]}
    arsenal = row_of(snapshot, "52280", 2026)
    assert rows["fouls_pg"]["a_z"] == pytest.approx(-arsenal["z_fouls_pg"])  # fewer fouls is better
    assert rows["points_pg"]["a_z"] == pytest.approx(arsenal["z_points_pg"])
    assert rows["points_pg"]["a_value"] == round(arsenal["points_pg"], 2)
    assert out["a"] == {"team_id": "52280", "name": "Team 52280", "season": 2026, "label": "2025-26",
                        "stage_label": "synthetic", "live": False}
    for r in out["rows"]:
        assert r["ahead"] == queries.ahead(r["a_z"], r["b_z"])
    json.dumps(out, allow_nan=False)


def test_ahead_calls_a_small_gap_or_a_missing_value_a_tie():
    assert queries.ahead(0.30, 0.27) == "tie" and queries.ahead(None, 1.0) == "tie"
    assert queries.ahead(0.4, 0.2) == "a" and queries.ahead(-1.0, 0.5) == "b"


def test_compare_with_an_unknown_pick_is_not_found(snapshot):
    with pytest.raises(queries.NotFound):
        queries.compare(snapshot, ("52280", 2026), ("nope", 2026))


def test_summary_has_metrics_with_intervals_the_top_drivers_and_the_synthesis(snapshot):
    s = queries.summary(snapshot)
    m = snapshot.results.metrics
    assert s["metrics"]["spearman"]["value"] == pytest.approx(m["spearman_mean"])
    assert s["metrics"]["auc"]["ci"] == pytest.approx(m["ci"]["auc"])
    assert s["metrics"]["brier_skill"]["value"] == pytest.approx(m["brier_skill"])
    assert s["metrics"]["top4_share"]["chance"] == pytest.approx(m["finalists_in_top4_chance"])
    assert len(s["drivers"]) == 6
    assert {d["kind"] for d in s["drivers"]} <= {"robust", "conditional", "model-dependent"}
    assert all(d["helps"] in ("higher", "lower", None) for d in s["drivers"])
    assert s["drivers"][0]["importance"] >= s["drivers"][-1]["importance"]
    assert s["synthesis"]["badge"]["kind"] == "good"
    json.dumps(s, allow_nan=False)


def test_profiles_survive_model_outputs_older_than_the_data(snapshot):
    import dataclasses

    results = snapshot.results
    older = dataclasses.replace(results, predictions=results.predictions[results.predictions["season"] != 2026],
                                shap=results.shap[results.shap["season"] != 2026])
    p = queries.profile(dataclasses.replace(snapshot, results=older), "52280", 2026)
    assert p["model"] is None and p["features"] and p["narrative"] is not None


def test_a_missing_stat_is_null_everywhere(snapshot):
    import dataclasses

    ts = snapshot.dataset.team_seasons.copy()
    hit = (ts["team_id"] == "52280") & (ts["season"] == 2026)
    for column in ("passes_pg", "pct_season_passes_pg", "pct_all_passes_pg", "z_passes_pg"):
        ts.loc[hit, column] = float("nan")
    holes = dataclasses.replace(snapshot, dataset=dataclasses.replace(snapshot.dataset, team_seasons=ts))
    passes = next(s for s in queries.profile(holes, "52280", 2026)["features"] if s["feature"] == "passes_pg")
    assert (passes["value"], passes["z"], passes["beats_season"], passes["beats_all"]) == (None, None, None, None)
    row = next(r for r in queries.compare(holes, ("52280", 2026), ("52747", 2026))["rows"]
               if r["feature"] == "passes_pg")
    assert (row["a_z"], row["ahead"]) == (None, "tie")
