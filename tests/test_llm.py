import http.client
import json
import os
import subprocess
import sys
import threading
import urllib.error

import pytest

from ucl import config
from ucl import llm as llm_module
from ucl.llm import ChatResult, LLMError, LMStudio, parse_loaded


class FakePost:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.bodies = response, error, []

    def __call__(self, url, body, timeout):
        self.bodies.append(body)
        if self.error:
            raise self.error
        return self.response


def reply(content, finish="stop"):
    return {"choices": [{"message": {"content": content, "reasoning_content": "hidden"}, "finish_reason": finish}]}


MSGS = [{"role": "user", "content": "x"}]
MODEL = "qwen/qwen3.5-35b-a3b"


def ps(*loaded):
    """What `lms ps --json` prints: a list of objects with an identifier and a context length."""
    return json.dumps([{"identifier": name, "contextLength": context, "type": "llm"} for name, context in loaded])


def cache_file(directory):
    (path,) = directory.glob("*.json")
    return path


def test_chat_returns_content_and_finish_reason(tmp_path):
    llm = LMStudio(cache_dir=tmp_path, post=FakePost(reply("## Hi\nThere")))
    assert llm.chat([{"role": "user", "content": "x"}]) == ChatResult("## Hi\nThere", "stop")


def test_request_disables_reasoning_and_uses_the_model(tmp_path):
    post = FakePost(reply("ok"))
    LMStudio(model="m/key", cache_dir=tmp_path, post=post).chat([{"role": "user", "content": "x"}])
    body = post.bodies[0]
    assert body["model"] == "m/key" and body["reasoning_effort"] == "none" and body["max_tokens"] == 8000


def test_identical_requests_are_served_from_cache(tmp_path):
    post = FakePost(reply("ok"))
    llm = LMStudio(cache_dir=tmp_path, post=post)
    llm.chat([{"role": "user", "content": "x"}])
    llm.chat([{"role": "user", "content": "x"}])
    llm.chat([{"role": "user", "content": "y"}])
    assert len(post.bodies) == 2


def test_cache_key_includes_the_model(tmp_path):
    post = FakePost(reply("ok"))
    LMStudio(model="a/one", cache_dir=tmp_path, post=post).chat([{"role": "user", "content": "x"}])
    LMStudio(model="b/two", cache_dir=tmp_path, post=post).chat([{"role": "user", "content": "x"}])
    assert len(post.bodies) == 2


def test_invalid_responses_are_cached_too(tmp_path):
    post = FakePost(reply("", finish="length"))
    llm = LMStudio(cache_dir=tmp_path, post=post)
    assert llm.chat([{"role": "user", "content": "x"}]).finish_reason == "length"
    assert llm.chat([{"role": "user", "content": "x"}]).finish_reason == "length"
    assert len(post.bodies) == 1


@pytest.mark.parametrize("error", [urllib.error.URLError("refused"), http.client.IncompleteRead(b""), TimeoutError()])
def test_failures_raise_llm_error_and_are_not_cached(tmp_path, error):
    llm = LMStudio(cache_dir=tmp_path, post=FakePost(error=error))
    with pytest.raises(LLMError):
        llm.chat([{"role": "user", "content": "x"}])
    assert not list(tmp_path.glob("*.json"))


@pytest.mark.parametrize("junk", [
    b"{trunc", b"", b"{}", b"[]", b"null", b'"text"', b"\xff\xfe\x00\x80",
    b'{"content": ["x"], "finish_reason": "stop"}',
])
def test_an_unusable_cache_file_is_a_miss(tmp_path, junk):
    post = FakePost(reply("fresh"))
    llm = LMStudio(cache_dir=tmp_path, post=post)
    llm.chat(MSGS)
    path = cache_file(tmp_path)
    path.write_bytes(junk)
    assert llm.chat(MSGS) == ChatResult("fresh", "stop")
    assert len(post.bodies) == 2
    assert json.loads(path.read_text())["content"] == "fresh"  # the bad entry was replaced


def test_a_cache_dir_that_cannot_be_created_does_not_lose_the_answer(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    llm = LMStudio(cache_dir=blocker / "cache", post=FakePost(reply("ok")))
    assert llm.chat(MSGS) == ChatResult("ok", "stop")


def test_a_directory_in_the_cache_slot_is_a_miss_and_leaves_no_temp_file(tmp_path):
    post = FakePost(reply("ok"))
    llm = LMStudio(cache_dir=tmp_path, post=post)
    llm.chat(MSGS)
    path = cache_file(tmp_path)
    path.unlink()
    path.mkdir()  # reading it raises OSError, and so does renaming a file over it
    assert llm.chat(MSGS) == ChatResult("ok", "stop")
    assert len(post.bodies) == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_cache_writes_use_a_temp_name_unique_to_the_process_and_thread(tmp_path, monkeypatch):
    sources, real_replace = [], os.replace

    def spy(src, dst):
        sources.append(os.path.basename(src))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    LMStudio(cache_dir=tmp_path, post=FakePost(reply("ok"))).chat(MSGS)
    assert sources == [f"{cache_file(tmp_path).name}.{os.getpid()}.{threading.get_ident()}.tmp"]


@pytest.mark.parametrize("content", [["x"], {"a": 1}, 5, 0, False])
def test_non_text_content_is_an_error_and_is_not_cached(tmp_path, content):
    llm = LMStudio(cache_dir=tmp_path, post=FakePost(reply(content)))
    with pytest.raises(LLMError, match="content"):
        llm.chat(MSGS)
    assert not list(tmp_path.iterdir())


def test_null_content_is_an_empty_answer(tmp_path):
    llm = LMStudio(cache_dir=tmp_path, post=FakePost(reply(None, finish="length")))
    assert llm.chat(MSGS) == ChatResult("", "length")


def test_parse_loaded_reads_the_context_of_the_exact_identifier():
    assert parse_loaded(ps(("other/model", 4096), (MODEL, 16384)), MODEL) == 16384


@pytest.mark.parametrize("identifier", [MODEL + "-mlx", "other/" + MODEL, MODEL + ":2"])
def test_parse_loaded_ignores_near_misses(identifier):
    assert parse_loaded(ps((identifier, 16384)), MODEL) is None


@pytest.mark.parametrize("listing", [
    "", "not json", "{}", "null", "[]", "[1, 2]",
    json.dumps({"identifier": MODEL, "contextLength": 5}),  # an object, not a list of them
    json.dumps([{"identifier": MODEL}]),
    json.dumps([{"identifier": MODEL, "contextLength": "16384"}]),
    json.dumps([{"identifier": MODEL, "contextLength": True}]),
])
def test_parse_loaded_is_none_for_bad_json_or_a_missing_context(listing):
    assert parse_loaded(listing, MODEL) is None


def stand_in_lms(llm, monkeypatch):
    """Point the client at the Python interpreter, so `_run_lms` runs a harmless child instead of lms."""
    monkeypatch.setattr(llm, "_lms_path", lambda: sys.executable)


def test_run_lms_returns_the_exit_code_and_both_output_streams(tmp_path, monkeypatch):
    llm = LMStudio(cache_dir=tmp_path)
    stand_in_lms(llm, monkeypatch)
    child = "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"
    code, output = llm._run_lms("-c", child, timeout=10)
    assert code == 3 and "out" in output and "err" in output


def test_run_lms_gives_the_child_no_stdin_and_no_pipes(tmp_path, monkeypatch):
    llm = LMStudio(cache_dir=tmp_path)
    stand_in_lms(llm, monkeypatch)
    seen, real_run = {}, subprocess.run
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: seen.update(kwargs) or real_run(*args, **kwargs))
    llm._run_lms("-c", "pass", timeout=10)
    assert seen["stdin"] is subprocess.DEVNULL and seen["stderr"] is subprocess.STDOUT
    assert seen["stdout"] is not subprocess.PIPE and not seen.get("capture_output")


def test_run_lms_does_not_wait_for_a_child_that_outlives_lms(tmp_path, monkeypatch):
    # `lms server start` leaves the server running with lms's output still open; a pipe would make run() wait for it
    llm = LMStudio(cache_dir=tmp_path)
    stand_in_lms(llm, monkeypatch)
    code = ("import subprocess, sys; print('started', flush=True); "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(4)'])")
    assert llm._run_lms("-c", code, timeout=2) == (0, "started\n")


def test_run_lms_survives_output_that_is_not_utf8(tmp_path, monkeypatch):
    llm = LMStudio(cache_dir=tmp_path)
    stand_in_lms(llm, monkeypatch)
    code = "import sys; sys.stdout.buffer.write(b'bad \\xff byte\\n')"
    assert llm._run_lms("-c", code, timeout=10) == (0, "bad \ufffd byte\n")


def test_run_lms_reports_a_timeout(tmp_path, monkeypatch):
    llm = LMStudio(cache_dir=tmp_path)
    stand_in_lms(llm, monkeypatch)
    code, why = llm._run_lms("-c", "import time; time.sleep(5)", timeout=0.3)
    assert code is None and "timed out" in why


def test_run_lms_reports_a_missing_or_unrunnable_lms(tmp_path, monkeypatch):
    llm = LMStudio(cache_dir=tmp_path)
    monkeypatch.setattr(llm, "_lms_path", lambda: None)
    assert llm._run_lms("ps", timeout=1) == (None, "lms not found")
    monkeypatch.setattr(llm, "_lms_path", lambda: str(tmp_path / "no-such-lms"))
    code, why = llm._run_lms("ps", timeout=1)
    assert code is None and "lms" in why


@pytest.mark.parametrize("result, ok, output", [
    ((0, "listing"), True, "listing"), ((1, "boom"), False, None), ((None, "lms not found"), False, None),
])
def test_lms_and_lms_output_follow_the_exit_code(tmp_path, monkeypatch, result, ok, output):
    llm = LMStudio(cache_dir=tmp_path)
    monkeypatch.setattr(llm, "_run_lms", lambda *args, timeout: result)
    assert llm._lms("ps", timeout=1) is ok
    assert llm._lms_output("ps", timeout=1) == output


def instant_waits(monkeypatch):
    """Make `_wait` check once instead of polling for 30 s; returns the list of durations it was asked for."""
    waits = []
    monkeypatch.setattr(llm_module, "_wait", lambda check, seconds, interval=1.0: waits.append(seconds) or check())
    return waits


def test_loaded_context_asks_lms_ps_with_a_short_timeout(tmp_path, monkeypatch):
    llm = LMStudio(model="m/key", cache_dir=tmp_path)
    seen = []

    def fake_run(*args, timeout):
        seen.append((args, timeout))
        return 0, ps(("m/key", 4096))

    monkeypatch.setattr(llm, "_run_lms", fake_run)
    assert llm._loaded_context() == 4096
    assert seen == [(("ps", "--json"), 10)]


def test_loaded_context_is_none_when_ps_fails_or_the_model_is_not_listed(tmp_path, monkeypatch):
    llm = LMStudio(model="m/key", cache_dir=tmp_path)
    monkeypatch.setattr(llm, "_run_lms", lambda *args, timeout: (1, "boom"))
    assert llm._loaded_context() is None
    monkeypatch.setattr(llm, "_run_lms", lambda *args, timeout: (0, ps(("other/model", 4096))))
    assert llm._loaded_context() is None


def test_ensure_ready_never_raises(tmp_path, monkeypatch):
    llm = LMStudio(cache_dir=tmp_path)
    monkeypatch.setattr(llm, "_server_up", lambda: False)
    monkeypatch.setattr(llm, "_lms", lambda *a, **k: False)
    assert llm.ensure_ready() is False
    assert llm.reason == "server did not start"


def test_ensure_ready_turns_an_unexpected_error_into_a_reason(tmp_path, monkeypatch):
    llm = LMStudio(cache_dir=tmp_path)

    def boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(llm, "_server_up", boom)
    assert llm.ensure_ready() is False
    assert llm.reason == "RuntimeError: boom"


@pytest.mark.parametrize("context", [config.LLM_CONTEXT_LENGTH, 2 * config.LLM_CONTEXT_LENGTH])
def test_ensure_ready_when_already_up_does_not_call_lms(tmp_path, monkeypatch, context):
    llm = LMStudio(cache_dir=tmp_path)
    llm.reason = "stale"
    monkeypatch.setattr(llm, "_server_up", lambda: True)
    monkeypatch.setattr(llm, "_loaded_context", lambda: context)

    def no_lms(*args, **kwargs):
        raise AssertionError("lms must not run")

    monkeypatch.setattr(llm, "_lms", no_lms)
    assert llm.ensure_ready() is True
    assert llm.reason is None


def test_ensure_ready_starts_a_server_that_is_down(tmp_path, monkeypatch):
    llm = LMStudio(cache_dir=tmp_path)
    up, calls = iter([False, True]), []  # down at first, up once started
    monkeypatch.setattr(llm, "_server_up", lambda: next(up))
    monkeypatch.setattr(llm, "_loaded_context", lambda: 16384)
    monkeypatch.setattr(llm, "_lms", lambda *args, timeout: calls.append(args) or True)
    assert llm.ensure_ready() is True
    assert calls == [("server", "start")]


def test_ensure_ready_reports_a_server_that_never_comes_up(tmp_path, monkeypatch):
    llm = LMStudio(cache_dir=tmp_path)
    waits = instant_waits(monkeypatch)
    monkeypatch.setattr(llm, "_server_up", lambda: False)
    monkeypatch.setattr(llm, "_lms", lambda *args, timeout: True)  # the start command itself succeeded
    assert llm.ensure_ready() is False
    assert llm.reason == "server did not start" and waits == [30]


def test_ensure_ready_loads_the_model_with_its_context_length(tmp_path, monkeypatch):
    llm = LMStudio(model="m/key", cache_dir=tmp_path)
    state, calls = {"loaded": False}, []

    def fake_lms(*args, timeout):
        calls.append(args)
        state["loaded"] = True
        return True

    monkeypatch.setattr(llm, "_server_up", lambda: True)
    monkeypatch.setattr(llm, "_loaded_context", lambda: 16384 if state["loaded"] else None)
    monkeypatch.setattr(llm, "_lms", fake_lms)
    assert llm.ensure_ready() is True
    assert calls == [("load", "m/key", "--context-length", "16384", "-y")]
    assert llm.reason is None


def test_ensure_ready_reloads_a_model_loaded_with_too_small_a_context(tmp_path, monkeypatch):
    llm = LMStudio(model="m/key", cache_dir=tmp_path)
    contexts, calls = iter([4096, 16384]), []  # LM Studio's default, then what the reload gives
    monkeypatch.setattr(llm, "_server_up", lambda: True)
    monkeypatch.setattr(llm, "_loaded_context", lambda: next(contexts))
    monkeypatch.setattr(llm, "_lms", lambda *args, timeout: calls.append(args) or True)
    assert llm.ensure_ready() is True
    assert calls == [("unload", "m/key"), ("load", "m/key", "--context-length", "16384", "-y")]
    assert llm.reason is None


def test_ensure_ready_does_not_load_a_second_copy_when_the_unload_fails(tmp_path, monkeypatch):
    llm = LMStudio(model="m/key", cache_dir=tmp_path)
    calls = []
    monkeypatch.setattr(llm, "_server_up", lambda: True)
    monkeypatch.setattr(llm, "_loaded_context", lambda: 4096)
    monkeypatch.setattr(llm, "_lms", lambda *args, timeout: calls.append(args) or False)
    assert llm.ensure_ready() is False
    assert calls == [("unload", "m/key")] and llm.reason == "unload failed"


def test_ensure_ready_reports_a_load_that_never_reaches_the_context(tmp_path, monkeypatch):
    llm = LMStudio(cache_dir=tmp_path)
    waits = instant_waits(monkeypatch)
    contexts = iter([None, 4096])  # not loaded, then still small after the load
    monkeypatch.setattr(llm, "_server_up", lambda: True)
    monkeypatch.setattr(llm, "_loaded_context", lambda: next(contexts))
    monkeypatch.setattr(llm, "_lms", lambda *args, timeout: True)
    assert llm.ensure_ready() is False
    assert "16384" in llm.reason and waits == [30]


def test_ensure_ready_gives_up_when_the_load_command_fails(tmp_path, monkeypatch):
    llm = LMStudio(cache_dir=tmp_path)
    checks = []
    monkeypatch.setattr(llm, "_server_up", lambda: True)
    monkeypatch.setattr(llm, "_loaded_context", lambda: checks.append(1) or None)
    monkeypatch.setattr(llm, "_lms", lambda *args, timeout: False)
    assert llm.ensure_ready() is False
    assert len(checks) == 1  # gave up at once instead of polling
    assert llm.reason == "load failed"


@pytest.mark.parametrize("printed, reason", [
    ("\n  Error: not enough memory  \nsecond line\n", "load failed: Error: not enough memory"),
    ("", "load failed: exit status 1"),
    ("x" * 500, "load failed: " + "x" * 200),
], ids=["first non-blank line", "silent failure", "long line capped"])
def test_ensure_ready_says_why_the_load_failed_from_what_lms_printed(tmp_path, monkeypatch, printed, reason):
    llm = LMStudio(cache_dir=tmp_path)
    monkeypatch.setattr(llm, "_server_up", lambda: True)
    monkeypatch.setattr(llm, "_loaded_context", lambda: None)
    monkeypatch.setattr(llm, "_run_lms", lambda *args, timeout: (1, printed))
    assert llm.ensure_ready() is False
    assert llm.reason == reason


@pytest.mark.parametrize("server_up", [False, True])
def test_ensure_ready_says_when_lms_is_missing(tmp_path, monkeypatch, server_up):
    llm = LMStudio(cache_dir=tmp_path)
    monkeypatch.setattr(llm, "_server_up", lambda: server_up)
    monkeypatch.setattr(llm, "_lms_path", lambda: None)
    assert llm.ensure_ready() is False
    assert "lms" in llm.reason
