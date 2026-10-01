import json

from fakes import FakeLLM

from ucl import analyst
from ucl.llm import ChatResult


def team_reply():
    # bold headings instead of '##': accepted, then normalised
    return ChatResult("**How they got there**\nx\n**Would the model have picked them?**\ny\n**Weak spots**\nz", "stop")


def synth_reply():
    return ChatResult("## Why the best teams win\nx\n## Winners vs runners-up\ny", "stop")


def test_run_skipped_and_unavailable_paths(built):
    ds, res = built
    assert analyst.run(ds.team_seasons, ds.finals, res, enabled=False).status == "skipped"

    class Down(FakeLLM):
        def ensure_ready(self):
            return False

    down = analyst.run(ds.team_seasons, ds.finals, res, llm=Down([]))
    assert down.status == "unavailable" and down.reason is None  # this fake has no reason to give


def test_run_writes_eleven_normalised_narratives(built, tmp_path):
    ds, res = built
    llm = FakeLLM([team_reply()] * 10 + [synth_reply()])
    analysis = analyst.run(ds.team_seasons, ds.finals, res, llm=llm, log=lambda _: None)
    assert analysis.status == "ok"
    assert len(analysis.narratives) == 11 and analysis.narratives["synthesis"].status == "ok"
    team = next(n for k, n in analysis.narratives.items() if k != "synthesis")
    assert team.text.startswith("## How they got there\n")
    analyst.save(analysis, tmp_path / "analysis.json")
    loaded = analyst.load(tmp_path / "analysis.json")
    assert loaded.narratives["synthesis"].text == analysis.narratives["synthesis"].text


def test_load_without_a_file_is_skipped(tmp_path):
    assert analyst.load(tmp_path / "missing.json").status == "skipped"


def test_unavailable_run_carries_the_clients_reason_through_save_and_load(built, tmp_path):
    ds, res = built

    class Down(FakeLLM):
        reason = "load failed: not enough memory"

        def ensure_ready(self):
            return False

    analysis = analyst.run(ds.team_seasons, ds.finals, res, llm=Down([]))
    assert analysis.status == "unavailable" and analysis.reason == "load failed: not enough memory"
    analyst.save(analysis, tmp_path / "analysis.json")
    assert analyst.load(tmp_path / "analysis.json").reason == "load failed: not enough memory"


def test_load_accepts_a_file_saved_before_reasons_existed(tmp_path):
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps({"status": "unavailable", "model": "m/key", "narratives": {}, "facts": {}}))
    loaded = analyst.load(path)
    assert loaded.status == "unavailable" and loaded.reason is None


def test_a_narratives_style_hits_survive_save_and_load(tmp_path):
    narrative = analyst.Narrative("2024-50051", "ok", text="t", calls=2, style=["significantly"])
    analyst.save(analyst.Analysis("ok", "m/key", {narrative.key: narrative}), tmp_path / "analysis.json")
    loaded = analyst.load(tmp_path / "analysis.json").narratives[narrative.key]
    assert loaded.style == ["significantly"] and loaded == narrative
    assert analyst.Narrative("k", "ok").style == []


def test_load_accepts_narratives_saved_before_style_existed(tmp_path):
    path = tmp_path / "analysis.json"
    old = {"key": "k", "status": "ok", "text": "t", "unsupported": ["47"], "calls": 2, "reason": None}
    path.write_text(json.dumps({"status": "ok", "model": "m/key", "narratives": {"k": old}, "facts": {}}))
    loaded = analyst.load(path).narratives["k"]
    assert loaded.style == [] and loaded.unsupported == ["47"]


def test_run_bans_loose_significance_wording_in_team_narratives_only(built):
    ds, res = built
    loose_team = ChatResult(team_reply().content.replace("\nx\n", "\nThey did significantly well.\n"), "stop")
    loose_synth = ChatResult(synth_reply().content.replace("\ny", "\nNo gap was significant."), "stop")
    llm = FakeLLM([loose_team, team_reply()] * 10 + [loose_synth])
    analysis = analyst.run(ds.team_seasons, ds.finals, res, llm=llm, log=lambda _: None)
    teams = [n for k, n in analysis.narratives.items() if k != "synthesis"]
    assert len(teams) == 10 and all((n.calls, n.style) == (2, []) for n in teams)  # retried once, then clean
    assert "significantly" not in teams[0].text
    synth = analysis.narratives["synthesis"]
    assert (synth.calls, synth.style) == (1, []) and "significant" in synth.text  # a p-value comparison may use it
    assert analyst.BANNED_TEAM_WORDS == ("significant", "significantly", "significance")


def test_synthesis_instructions_cover_intervals_labels_and_the_training_data():
    text = analyst.SYNTH_INSTRUCTIONS
    for needed in ("exactly two paragraphs", "Spearman, AUC, Brier skill and top-4 share each with its 95% interval",
                   "no-skill value", "inconclusive", "tentative", "nothing about the feature sets was tested",
                   "'conditional'", "'model-dependent'", "knockout count", "first four stats"):
        assert needed in text, needed
    assert "Facts (JSON):" in text and text.endswith("\n")  # the facts follow straight after
