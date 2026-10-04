import dataclasses
import json
import os
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest

from ucl import analyst, config, dataset, model
from ucl.analyst import Analysis, Narrative
from ucl.facts import build_facts
from ucl.web import checks
from ucl.web.checks import FAIL, INFO, OK, WARN
from ucl.web.store import DataStore

MODEL = config.LLM_MODEL


def lms(responses):
    """A fake `lms`: {first argument: (exit code, output)}."""
    def run(*args, timeout):
        return responses[args[0]]
    return run


def ps(context):
    return json.dumps([{"identifier": MODEL, "contextLength": context}])


def down(url, timeout):
    raise OSError("Connection refused")


# --- AI model: LM Studio ---

def test_the_lms_command_is_found_or_its_install_explained(monkeypatch, tmp_path):
    exe = tmp_path / "lms"
    exe.write_text("")
    monkeypatch.setattr(config, "LMS_BIN", exe)
    assert checks.lms_cli().status == OK
    monkeypatch.setattr(config, "LMS_BIN", tmp_path / "missing")
    monkeypatch.setattr(checks.shutil, "which", lambda name: None)
    missing = checks.lms_cli()
    assert missing.status == FAIL and "lms bootstrap" in missing.fix


def test_the_server_check_knows_whether_the_analyze_stage_can_start_it():
    up = checks.lm_server(lambda url, timeout: json.dumps({"data": [{"id": "a"}, {"id": "b"}]}).encode(), True)
    assert (up.status, up.detail) == (OK, f"Answering at {config.LLM_BASE_URL} (2 models listed)")
    assert checks.lm_server(down, cli_found=True).status == WARN  # `lms server start` will run
    assert checks.lm_server(down, cli_found=False).status == FAIL


def test_the_model_must_be_downloaded():
    listed = json.dumps([{"modelKey": MODEL, "sizeBytes": 22_072_117_341}])
    assert checks.lm_model_downloaded(lms({"ls": (0, listed)}), MODEL).detail == f"{MODEL} is downloaded (22.1 GB)"
    missing = checks.lm_model_downloaded(lms({"ls": (0, "[]")}), MODEL)
    assert missing.status == FAIL and f"lms get {MODEL}" in missing.fix
    assert checks.lm_model_downloaded(lms({"ls": (None, "lms not found")}), MODEL).status == WARN


def test_the_model_should_be_loaded_with_the_context_the_write_ups_need():
    assert checks.lm_model_loaded(lms({"ps": (0, ps(16384))}), MODEL).status == OK
    small = checks.lm_model_loaded(lms({"ps": (0, ps(4096))}), MODEL)
    assert small.status == WARN and "4,096" in small.detail and "16,384" in small.fix
    unloaded = checks.lm_model_loaded(lms({"ps": (0, "[]")}), MODEL)
    assert (unloaded.status, unloaded.detail) == (WARN, "Not loaded yet")


class FakeLLM:
    name = "LM Studio"

    def __init__(self, ready=True, response=None, error=None):
        self.ready, self.reason = ready, None if ready else "server did not start: lms not found"
        self.response = response or {"choices": [{"message": {"content": " ready \n"}}]}
        self.error, self.sent = error, []

    def ensure_ready(self):
        return self.ready

    def send(self, body, timeout=None):
        self.sent.append((body, timeout))
        if self.error:
            raise self.error
        return self.response

    def scrub(self, text):
        return text


def test_trying_the_model_loads_it_then_asks_one_uncached_question():
    llm = FakeLLM()
    ticks = iter([0.0, 2.0, 2.5])
    result = checks.try_model("m", llm_factory=lambda name: llm, clock=lambda: next(ticks))
    assert result == {"ok": True, "detail": "m answered through LM Studio in 0.5 s", "reply": "ready",
                      "load_seconds": 2.0, "answer_seconds": 0.5}
    ((body, timeout),) = llm.sent
    assert (body["model"], body["max_tokens"], body["reasoning_effort"]) == ("m", 16, "none")
    assert timeout == checks.MODEL_TEST_TIMEOUT_S  # one tiny question, not a write-up's five minutes


def test_trying_the_model_reports_what_openrouter_charged():
    llm = FakeLLM(response={"choices": [{"message": {"content": "ready"}}], "usage": {"cost": 0.0000042}})
    llm.name = "OpenRouter"
    result = checks.try_model("m", llm_factory=lambda name: llm, clock=lambda: 0.0)
    assert result["detail"] == "m answered through OpenRouter in 0.0 s for $0.000004"


def test_trying_the_model_says_why_it_failed():
    not_ready = checks.try_model("m", llm_factory=lambda name: FakeLLM(ready=False), clock=lambda: 0.0)
    assert not not_ready["ok"] and not_ready["detail"] == "LM Studio isn't ready: server did not start: lms not found"
    silent = checks.try_model("m", llm_factory=lambda name: FakeLLM(error=OSError("Connection refused")),
                              clock=lambda: 0.0)
    assert not silent["ok"] and silent["detail"].startswith("The model didn't answer: OSError")


def test_trying_the_model_with_an_unknown_provider_says_so(monkeypatch):
    monkeypatch.setenv(config.LLM_PROVIDER_ENV, "elsewhere")
    result = checks.try_model("m")
    assert not result["ok"] and "UCL_LLM_PROVIDER" in result["detail"]


# --- UEFA feeds ---

def test_uefa_probes_ask_for_as_little_as_possible(tmp_path):
    (tmp_path / "stats").mkdir()
    (tmp_path / "stats" / "2015.json").write_text("[]")
    urls = {probe_id: url for probe_id, _, url in checks.uefa_probes(tmp_path)}
    assert "limit=1&" in urls["uefa_matches"] and "pagesize=1&" in urls["uefa_coefficients"]
    assert urls["uefa_stats"].endswith("/2015") and "limit=1&" in urls["uefa_players"]
    assert dict((p[0], p[2]) for p in checks.uefa_probes(tmp_path / "empty"))["uefa_stats"] is None


def test_a_feed_that_answers_is_ok_and_one_that_doesnt_is_a_warning():
    assert checks.uefa_feed("uefa_matches", "Matches", "https://x", lambda url, timeout: b"[]").status == OK
    unreachable = checks.uefa_feed("uefa_matches", "Matches", "https://x", down)
    assert unreachable.status == WARN and "offline" in unreachable.fix
    assert checks.uefa_feed("uefa_stats", "Match stats", None, down).status == INFO


# --- Pipeline outputs ---

@pytest.fixture
def outputs(tmp_path, built):
    ds, results = built
    processed, out = tmp_path / "processed", tmp_path / "out"
    dataset.save(ds, processed)
    model.save(results, out)
    loaded = dataset.load(processed)
    sheets = build_facts(loaded.team_seasons, loaded.finals, model.load(out))
    narratives = {k: Narrative(k, "ok", text="## Heading\nText.") for k in sheets}
    analyst.save(Analysis("ok", "m", narratives, sheets), out / "analysis.json")
    return processed, out


def test_the_raw_cache_check_counts_seasons(tmp_path):
    (tmp_path / "matches").mkdir()
    for season in config.SEASONS[:-1]:
        (tmp_path / "matches" / f"{season}.json").write_text("[]")
    partial = checks.raw_cache(tmp_path)
    assert partial.status == WARN and partial.detail == f"{len(config.SEASONS) - 1} of {len(config.SEASONS)} seasons cached"
    (tmp_path / "matches" / f"{config.SEASONS[-1]}.json").write_text("[]")
    assert checks.raw_cache(tmp_path).status == OK


def test_dataset_model_write_ups_and_report_when_everything_is_in_place(outputs):
    processed, out = outputs
    snap = DataStore(processed, out).snapshot
    (out / "report.html").write_text("<html></html>")
    assert [check.status for check in (checks.dataset_check(snap), checks.model_check(snap),
                                       checks.write_ups_check(snap), checks.report_check(snap, out))] == [OK] * 4
    assert checks.write_ups_check(snap).detail.endswith("every figure found in the data")


def test_missing_outputs_say_which_stage_to_run(tmp_path):
    snap = DataStore(tmp_path / "processed", tmp_path / "out").snapshot
    assert (checks.dataset_check(snap).status, checks.dataset_check(snap).fix) == (FAIL, "Run build")
    assert (checks.model_check(snap).status, checks.model_check(snap).fix) == (FAIL, "Run model")
    assert checks.write_ups_check(snap).status == WARN and checks.report_check(snap, tmp_path).status == WARN


def test_a_model_older_than_the_dataset_is_flagged(outputs):
    snap = DataStore(*outputs).snapshot
    rebuilt = dataclasses.replace(snap, built_at=snap.modelled_at + timedelta(hours=1))
    assert checks.model_check(rebuilt).status == WARN


def test_write_ups_that_are_missing_unavailable_or_stale_are_flagged(outputs):
    snap = DataStore(*outputs).snapshot
    failed = dataclasses.replace(snap, analysis=Analysis("unavailable", "m", snap.analysis.narratives,
                                                         reason="server did not start"))
    assert checks.write_ups_check(failed).status == WARN and "server did not start" in checks.write_ups_check(failed).detail
    stale = dataclasses.replace(snap, stale_keys=frozenset(list(snap.analysis.narratives)[:2]))
    assert checks.write_ups_check(stale).detail.startswith("2 of")
    holes = dict(snap.analysis.narratives)
    key = next(iter(holes))
    holes[key] = Narrative(key, "unavailable")
    gappy = dataclasses.replace(snap, analysis=Analysis("ok", "m", holes))
    assert checks.write_ups_check(gappy).detail.startswith("1 of")


def test_a_report_older_than_the_write_ups_is_flagged(outputs):
    processed, out = outputs
    (out / "report.html").write_text("<html></html>")
    old = time.time() - 7200
    os.utime(out / "report.html", (old, old))
    assert checks.report_check(DataStore(processed, out).snapshot, out).status == WARN


def test_saved_ai_answers_are_counted(tmp_path):
    for i in range(3):
        (tmp_path / f"{i}.json").write_text("{}")
    assert (checks.llm_cache_check(tmp_path).status, checks.llm_cache_check(tmp_path).detail[:7]) == (INFO, "3 saved")


# --- Live season and players ---

def live(status, snapshot=None, message=""):
    return SimpleNamespace(status=status, snapshot=snapshot, message=message)


def test_the_live_season_check_follows_the_live_service():
    snapshot = SimpleNamespace(rows=[1] * 36, finished_matches=18, fetched_at=None)
    assert checks.live_check(None).status == INFO
    assert checks.live_check(live("loading")).status == INFO
    assert checks.live_check(live("ready", snapshot)).detail.startswith("36 teams, 18 finished matches")
    assert checks.live_check(live("stale", snapshot)).status == WARN
    assert checks.live_check(live("unavailable", message="UEFA down")).detail == "Unavailable (UEFA down)"


def test_the_player_index_check_says_how_much_is_cached():
    def players(done, of, running=False):
        return SimpleNamespace(index_status=lambda: {"complete": done == of, "squads": {"done": done, "of": of},
                                                     "running": running})

    assert checks.player_index_check(players(524, 524)).status == OK
    assert checks.player_index_check(players(2, 524)).status == WARN
    assert checks.player_index_check(players(200, 524, running=True)).detail == "Building: 200 of 524 squads"


# --- Dashboard ---

def test_the_dashboard_build_must_exist_and_be_newer_than_its_sources(tmp_path):
    assert checks.frontend_check(tmp_path).status == FAIL
    (tmp_path / "src").mkdir()
    (tmp_path / "dist").mkdir()
    (tmp_path / "src" / "App.tsx").write_text("x")
    (tmp_path / "dist" / "index.html").write_text("x")
    old = time.time() - 600
    os.utime(tmp_path / "dist" / "index.html", (old, old))
    assert checks.frontend_check(tmp_path).status == WARN
    os.utime(tmp_path / "src" / "App.tsx", (old - 600, old - 600))
    assert checks.frontend_check(tmp_path).status == OK


def test_run_checks_groups_everything_in_order_with_a_summary(outputs, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LMS_BIN", tmp_path / "lms")
    (tmp_path / "lms").write_text("")
    processed, out = outputs
    services = SimpleNamespace(store=DataStore(processed, out), live=None, players=None)
    fake_lms = lms({"ls": (0, json.dumps([{"modelKey": MODEL}])), "ps": (0, ps(16384))})
    report = checks.run_checks(services, MODEL, get=lambda url, timeout: b"{}", run_lms=fake_lms,
                               raw_dir=tmp_path / "raw", out_dir=out, cache_dir=tmp_path / "cache",
                               web_dir=tmp_path / "web")
    groups = [check["group"] for check in report["checks"]]
    assert groups == sorted(groups, key=checks.GROUPS.index) and groups[0] == "AI model"
    assert (report["model"], report["provider"]) == (MODEL, "lmstudio")
    assert sum(report["summary"].values()) == len(report["checks"]) == 17
    by_id = {check["id"]: check for check in report["checks"]}
    assert by_id["lm_model_loaded"]["status"] == OK and by_id["frontend"]["status"] == FAIL
    json.dumps(report, allow_nan=False)


# --- AI model: OpenRouter ---

def answers(table):
    """A fake GET for JSON: {url suffix: a dict to return or an exception to raise}."""
    def get(url, timeout, headers):
        for suffix, answer in table.items():
            if url.endswith(suffix):
                if isinstance(answer, Exception):
                    raise answer
                return answer
        raise AssertionError(url)
    return get


def http_error(code):
    import urllib.error
    return urllib.error.HTTPError("https://openrouter.ai/api/v1/x", code, "no", {}, None)


def endpoint(quantization, prompt, completion):
    return {"quantization": quantization, "pricing": {"prompt": str(prompt / 1e6), "completion": str(completion / 1e6)}}


def test_the_openrouter_key_must_be_set(monkeypatch):
    assert checks.openrouter_key().status == FAIL
    monkeypatch.setenv(config.OPENROUTER_KEY_ENV, "sk-or-x")
    check = checks.openrouter_key()
    assert check.status == OK and "sk-or-x" not in check.detail


def test_the_openrouter_account_reports_spending_and_what_is_left(monkeypatch):
    monkeypatch.setenv(config.OPENROUTER_KEY_ENV, "sk-or-x")
    unlimited = checks.openrouter_account(answers({"/key": {"data": {"usage": 0.04, "limit_remaining": None}}}), True)
    assert (unlimited.status, unlimited.detail) == (OK, "The key works: $0.04 spent with this key")
    capped = checks.openrouter_account(answers({"/key": {"data": {"usage": 1.5, "limit_remaining": 8.5}}}), True)
    assert capped.status == OK and capped.detail.endswith("$8.50 left")
    low = checks.openrouter_account(answers({"/key": {"data": {"usage": 9.7, "limit_remaining": 0.3}}}), True)
    assert low.status == WARN and "$0.30" in low.detail


def test_the_openrouter_account_check_says_what_went_wrong(monkeypatch):
    monkeypatch.setenv(config.OPENROUTER_KEY_ENV, "sk-or-x")
    assert checks.openrouter_account(answers({"/key": {}}), False).status == INFO
    assert checks.openrouter_account(answers({"/key": http_error(401)}), True).status == FAIL
    assert checks.openrouter_account(answers({"/key": http_error(502)}), True).status == WARN
    assert checks.openrouter_account(answers({"/key": OSError("down")}), True).status == WARN


def test_the_model_check_counts_providers_with_allowed_weights_and_their_prices():
    listing = {"data": {"endpoints": [endpoint("bf16", 0.15, 1.875), endpoint("fp8", 0.40, 3.0),
                                      endpoint("fp4", 0.05, 2.2), endpoint("unknown", 0.02, 4.35)]}}
    check = checks.openrouter_model(answers({"/endpoints": listing}), MODEL)
    assert check.status == OK
    assert check.detail == (f"2 of 4 providers serve {MODEL} at bf16, fp16, fp8; per million tokens in and out, "
                            "from $0.15 and $1.88 to $0.40 and $3.00")


def test_the_model_check_fails_when_no_allowed_provider_or_no_model():
    only_fp4 = {"data": {"endpoints": [endpoint("fp4", 0.05, 2.2)]}}
    assert checks.openrouter_model(answers({"/endpoints": only_fp4}), MODEL).status == FAIL
    assert checks.openrouter_model(answers({"/endpoints": http_error(404)}), MODEL).status == FAIL
    assert checks.openrouter_model(answers({"/endpoints": {"data": {}}}), MODEL).status == WARN
    assert checks.openrouter_model(answers({"/endpoints": OSError("down")}), MODEL).status == WARN


def test_run_checks_uses_the_openrouter_checks_when_it_answers(outputs, tmp_path, monkeypatch):
    monkeypatch.setenv(config.OPENROUTER_KEY_ENV, "sk-or-x")
    processed, out = outputs
    services = SimpleNamespace(store=DataStore(processed, out), live=None, players=None)
    listing = {"data": {"endpoints": [endpoint("bf16", 0.15, 1.875)]}}
    report = checks.run_checks(services, MODEL, get=lambda url, timeout: b"{}", raw_dir=tmp_path / "raw",
                               out_dir=out, cache_dir=tmp_path / "cache", web_dir=tmp_path / "web",
                               get_json=answers({"/key": {"data": {"usage": 0.0}}, "/endpoints": listing}))
    ai = [check for check in report["checks"] if check["group"] == "AI model"]
    assert report["provider"] == "openrouter"
    assert [(c["id"], c["status"]) for c in ai] == [("openrouter_key", OK), ("openrouter_account", OK),
                                                    ("openrouter_model", OK)]


def test_an_unknown_provider_is_a_failed_check(outputs, tmp_path, monkeypatch):
    monkeypatch.setenv(config.LLM_PROVIDER_ENV, "elsewhere")
    processed, out = outputs
    services = SimpleNamespace(store=DataStore(processed, out), live=None, players=None)
    report = checks.run_checks(services, MODEL, get=lambda url, timeout: b"{}", raw_dir=tmp_path / "raw",
                               out_dir=out, cache_dir=tmp_path / "cache", web_dir=tmp_path / "web")
    first = report["checks"][0]
    assert (report["provider"], first["id"], first["status"]) == ("unknown", "llm_provider", FAIL)


def test_the_website_build_is_checked_with_its_own_command(tmp_path):
    check = checks.frontend_check(tmp_path, "dist-site")
    assert check.status == FAIL and check.detail == "web/dist-site is missing" and "npm run build:site" in check.fix


def test_a_key_pasted_with_a_line_break_fails_without_being_repeated(monkeypatch):
    monkeypatch.setenv(config.OPENROUTER_KEY_ENV, "sk-or-v1-secret\r\nmore")
    check = checks.openrouter_key()
    assert check.status == FAIL and "secret" not in check.detail and "line breaks" in check.detail


def test_the_website_leaves_out_what_the_key_has_spent(monkeypatch):
    monkeypatch.setenv(config.OPENROUTER_KEY_ENV, "sk-or-x")
    get = answers({"/key": {"data": {"usage": 1.5, "limit_remaining": 8.5}}})
    assert checks.openrouter_account(get, True, show_spending=False).detail == "The key works"
    low = checks.openrouter_account(answers({"/key": {"data": {"usage": 9.7, "limit_remaining": 0.3}}}), True,
                                    show_spending=False)
    assert low.status == WARN and "$" not in low.detail


def stages(*pairs, run_id="b-r1", running=False):
    return {"running": running, "run_id": run_id,
            "stages": [{"name": name, "status": status, "message": message} for name, status, message in pairs]}


def test_the_last_run_check_follows_the_runner():
    assert checks.last_run_check(stages(("live", "idle", ""), run_id=None)).status == INFO
    assert checks.last_run_check(stages(("live", "running", ""), running=True)).detail == "Running now"
    done = checks.last_run_check(stages(("fetch", "idle", ""), ("live", "done", "36 teams"), ("players", "done", "")))
    assert (done.status, done.detail) == (OK, "Done: live, players")
    warned = checks.last_run_check(stages(("live", "warning", "UEFA couldn't be reached"), ("players", "done", "")))
    assert warned.status == WARN and "UEFA couldn't be reached" in warned.detail
    failed = checks.last_run_check(stages(("live", "done", ""), ("players", "failed", "timed out")))
    assert (failed.status, failed.detail) == (FAIL, "players failed: timed out")
