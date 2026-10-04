import hashlib
import json

from fakes import FakeLLM

from ucl import analyst, config, facts
from ucl.llm import ChatResult


def team_reply():
    # bold headings instead of '##': accepted, then normalised
    return ChatResult("**How they got there**\nx\n**Would the model have picked them?**\ny\n**Weak spots**\nz", "stop")


def synth_reply():
    return ChatResult("## Why the best teams win\nx\n## Winners vs runners-up\ny", "stop")


def padded(reply, words):
    """`reply` with `words` filler words in its first paragraph."""
    return ChatResult(reply.content.replace("\nx\n", "\n" + " ".join(["pad"] * words) + "\n"), "stop")


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


class Offline(FakeLLM):
    """LM Studio is down: only saved answers can be given."""
    name = "LM Studio"

    def ensure_ready(self):
        self.asked_ready = True
        return False


def test_a_rerun_with_every_answer_saved_replays_without_lm_studio(built):
    ds, res = built
    first = FakeLLM([team_reply()] * 10 + [synth_reply()])
    before = analyst.run(ds.team_seasons, ds.finals, res, llm=first, log=lambda _: None)
    offline = Offline([])
    offline.cache = first.cache
    lines = []
    again = analyst.run(ds.team_seasons, ds.finals, res, llm=offline, log=lines.append)
    assert again.status == "ok" and not hasattr(offline, "asked_ready") and offline.requests == []
    assert again.narratives == before.narratives
    assert lines[0].startswith("Every answer was saved") and len(lines) == 12


def test_one_new_question_needs_lm_studio_and_nothing_is_asked_twice(built):
    ds, res = built
    first = FakeLLM([team_reply()] * 10 + [synth_reply()])
    analyst.run(ds.team_seasons, ds.finals, res, llm=first, log=lambda _: None)
    synthesis = next(m for m in first.requests if m[1]["content"].startswith(analyst.SYNTH_INSTRUCTIONS))
    partial = dict(first.cache)
    del partial[json.dumps(synthesis, sort_keys=True)]

    offline = Offline([])
    offline.cache = dict(partial)
    lines = []
    down = analyst.run(ds.team_seasons, ds.finals, res, llm=offline, log=lines.append)
    assert down.status == "unavailable" and offline.asked_ready and offline.requests == []
    assert lines == ["Some answers aren't saved yet, so LM Studio is needed (loading the model can take a few minutes)"]

    online = FakeLLM([synth_reply()])  # only the missing answer is asked for
    online.cache = dict(partial)
    analysis = analyst.run(ds.team_seasons, ds.finals, res, llm=online, log=lambda _: None)
    assert analysis.status == "ok" and len(online.requests) == 1 and len(analysis.narratives) == 11


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


def test_run_holds_team_narratives_to_250_words_and_the_synthesis_to_450(built):
    ds, res = built
    # 300 words: over the team limit, inside the synthesis limit
    llm = FakeLLM([padded(team_reply(), 300), team_reply()] * 10 + [padded(synth_reply(), 300)])
    analysis = analyst.run(ds.team_seasons, ds.finals, res, llm=llm, log=lambda _: None)
    teams = [n for k, n in analysis.narratives.items() if k != "synthesis"]
    assert all((n.calls, n.style) == (2, []) for n in teams)  # retried once, then short enough
    assert "250" in llm.requests[1][-1]["content"]
    synth = analysis.narratives["synthesis"]
    assert (synth.calls, synth.style) == (1, [])
    # 500 words: over the synthesis limit too
    llm = FakeLLM([team_reply()] * 10 + [padded(synth_reply(), 500), synth_reply()])
    synth = analyst.run(ds.team_seasons, ds.finals, res, llm=llm, log=lambda _: None).narratives["synthesis"]
    assert (synth.calls, synth.style) == (2, []) and "450" in llm.requests[-1][-1]["content"]


class SaysGroupStage(FakeLLM):
    """Answers by request: team narratives say 'group stage' until told off, the synthesis says 'group/league-phase'."""

    def __init__(self):
        super().__init__([])

    def chat(self, messages, offline=False):
        if offline:  # nothing is saved, so the replay stops at once
            return super().chat(messages, offline=True)
        self.requests.append(messages)
        if "## Why the best teams win" in messages[1]["content"]:
            return ChatResult(synth_reply().content.replace("\nx\n", "\nThe group/league-phase stats.\n"), "stop")
        if len(messages) > 2:  # the feedback has been added to the conversation
            return team_reply()
        return ChatResult(team_reply().content.replace("\nx\n", "\nTheir group stage.\n"), "stop")


def test_run_checks_the_first_phase_name_of_team_narratives_by_their_season_and_not_the_synthesis(built):
    ds, res = built
    llm = SaysGroupStage()
    analysis = analyst.run(ds.team_seasons, ds.finals, res, llm=llm, log=lambda _: None)
    teams = {k: n for k, n in analysis.narratives.items() if k != "synthesis"}
    seasons = {key: int(key.split("-", 1)[0]) for key in teams}
    assert {config.is_league_format(s) for s in seasons.values()} == {True, False}  # both formats are covered
    for key, narrative in teams.items():
        if config.is_league_format(seasons[key]):  # 'group stage' is wrong here: retried once, then clean
            assert (narrative.calls, narrative.style) == (2, []), key
        else:  # and right here
            assert (narrative.calls, narrative.style) == (1, []), key
            assert "group stage" in narrative.text
    retried = [r for r in llm.requests if len(r) > 2]
    assert len(retried) == sum(config.is_league_format(s) for s in seasons.values())
    assert all("this season had a league phase" in r[-1]["content"] for r in retried)
    synth = analysis.narratives["synthesis"]  # spans both formats, so the hedge is right there
    assert (synth.calls, synth.style) == (1, []) and "group/league-phase" in synth.text


def test_the_word_limits_match_the_ones_the_instructions_ask_for():
    assert f"-{analyst.TEAM_MAX_WORDS} words" in analyst.TEAM_INSTRUCTIONS
    assert f"-{analyst.SYNTH_MAX_WORDS} words" in analyst.SYNTH_INSTRUCTIONS


def test_team_instructions_tell_the_model_to_name_the_first_phase_as_the_facts_do(built):
    text = analyst.TEAM_INSTRUCTIONS
    assert ("Call the first phase exactly as 'First-phase format that season' names it "
            "(league phase or group stage).") in text
    assert "Facts (JSON):" in text and text.endswith("\n")  # the facts follow straight after
    ds, res = built
    sheet = next(v for k, v in facts.build_facts(ds.team_seasons, ds.finals, res).items() if k != "synthesis")
    assert "First-phase format that season" in sheet  # the key the instruction points at


def test_team_instructions_make_weak_spots_say_how_many_of_the_seasons_teams_the_stat_beat():
    # the local model wrote 'placed them below only 11% of teams' where the facts say the team beat 11%
    text = analyst.TEAM_INSTRUCTIONS
    sentence = ("For each weak spot give the stat's value and say it beat N% of that season's teams, "
                "using exactly that phrase.")
    assert text.count(sentence) == 1
    assert text.index("Under 'Weak spots'") < text.index(sentence) < text.index("Call the first phase")


def test_synthesis_instructions_cover_intervals_labels_and_the_training_data():
    text = analyst.SYNTH_INSTRUCTIONS
    for needed in ("exactly two paragraphs", "Spearman, AUC, Brier skill and top-4 share each with its 95% interval",
                   "no-skill value", "inconclusive", "tentative", "nothing about the feature sets was tested",
                   "'conditional'", "'model-dependent'", "knockout count", "first four stats"):
        assert needed in text, needed
    assert "Facts (JSON):" in text and text.endswith("\n")  # the facts follow straight after


def test_saved_answers_replay_without_any_provider(built, tmp_path, monkeypatch):
    ds, res = built
    first = FakeLLM([team_reply()] * 10 + [synth_reply()])
    analyst.run(ds.team_seasons, ds.finals, res, llm=first, log=lambda _: None)
    monkeypatch.setattr(config, "LLM_CACHE_DIR", tmp_path)
    for request, content in first.cache.items():  # the fake's answers, saved where the real clients look
        body = {"model": config.LLM_MODEL, "messages": json.loads(request), **config.LLM_PARAMS}
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        (tmp_path / f"{key}.json").write_text(json.dumps({"content": content.content, "finish_reason": "stop"}))
    monkeypatch.setenv(config.LLM_PROVIDER_ENV, "elsewhere")  # no provider at all: replaying needs none
    lines = []
    analysis = analyst.run(ds.team_seasons, ds.finals, res, log=lines.append)
    assert analysis.status == "ok" and len(analysis.narratives) == 11
    assert lines[0] == "Every answer was saved, so the write-ups replay without asking the model"


def test_a_new_question_with_no_provider_says_why(built, monkeypatch, tmp_path):
    ds, res = built
    monkeypatch.setattr(config, "LLM_CACHE_DIR", tmp_path)
    monkeypatch.setenv(config.LLM_PROVIDER_ENV, "elsewhere")
    analysis = analyst.run(ds.team_seasons, ds.finals, res, log=lambda _: None)
    assert analysis.status == "unavailable" and "UCL_LLM_PROVIDER" in analysis.reason
