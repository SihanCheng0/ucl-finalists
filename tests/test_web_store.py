import pytest
from synthetic import make_synthetic_dataset

from ucl import analyst, dataset, model
from ucl.analyst import Analysis, Narrative
from ucl.facts import build_facts
from ucl.web.store import DataStore


@pytest.fixture
def outputs(tmp_path, built):
    ds, results = built
    processed, out = tmp_path / "processed", tmp_path / "out"
    dataset.save(ds, processed)
    model.save(results, out)
    return processed, out


def save_analysis(processed, out, tamper=False):
    """An analysis written from the saved outputs, as `ucl analyze` writes it; returns one team key."""
    ds, results = dataset.load(processed), model.load(out)
    facts = build_facts(ds.team_seasons, ds.finals, results)
    key = next(k for k in facts if k != "synthesis")
    if tamper:
        facts[key] = {"Club": "numbers from an earlier run"}
    narratives = {k: Narrative(k, "ok", text="## Heading\nText.") for k in facts}
    analyst.save(Analysis("ok", "m", narratives, facts), out / "analysis.json")
    return key


def test_a_complete_set_of_outputs_is_ready_with_timestamps(outputs):
    processed, out = outputs
    save_analysis(processed, out)
    snap = DataStore(processed, out).snapshot
    assert snap.ready and snap.errors == {}
    assert snap.analysis.status == "ok"
    assert snap.built_at and snap.modelled_at and snap.analysed_at
    assert snap.stale_keys == frozenset()


def test_missing_outputs_are_not_ready_and_say_which_stage_to_run(tmp_path):
    snap = DataStore(tmp_path / "processed", tmp_path / "out").snapshot
    assert not snap.ready
    assert snap.errors == {"dataset": "processed data missing: run build", "results": "model outputs missing: run model"}
    assert snap.analysis.status == "skipped"  # no analysis.json is normal, not an error
    assert snap.built_at is None and snap.stale_keys == frozenset()


def test_a_corrupt_file_means_not_ready_rather_than_a_crash(outputs):
    processed, out = outputs
    (out / "metrics.json").write_text("{not json")
    snap = DataStore(processed, out).snapshot
    assert not snap.ready and snap.results is None and snap.modelled_at is None
    assert snap.errors["results"].startswith("model outputs could not be read: JSONDecodeError")


def test_reload_swaps_in_a_new_snapshot_and_leaves_the_old_one_alone(outputs):
    processed, out = outputs
    store = DataStore(processed, out)
    before = store.snapshot
    save_analysis(processed, out)
    after = store.reload("analysis")
    assert store.snapshot is after and after is not before
    assert before.analysis.status == "skipped" and after.analysis.status == "ok"
    assert after.dataset is before.dataset  # parts not reloaded are carried over, not re-read


def test_a_fixed_file_clears_its_error_on_reload(outputs):
    processed, out = outputs
    good = (out / "metrics.json").read_text()
    (out / "metrics.json").write_text("{not json")
    store = DataStore(processed, out)
    (out / "metrics.json").write_text(good)
    assert store.reload("results").errors == {} and store.snapshot.ready


def test_narratives_written_for_other_numbers_are_flagged_stale(outputs):
    processed, out = outputs
    key = save_analysis(processed, out, tamper=True)
    assert DataStore(processed, out).snapshot.stale_keys == frozenset({key})


def test_every_narrative_is_stale_when_data_and_model_no_longer_line_up(outputs):
    processed, out = outputs
    save_analysis(processed, out)
    dataset.save(make_synthetic_dataset(seasons=(2021, 2022)), processed)  # a rebuild the model hasn't seen
    snap = DataStore(processed, out).snapshot
    assert snap.stale_keys == frozenset(snap.analysis.narratives)
