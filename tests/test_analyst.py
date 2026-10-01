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
