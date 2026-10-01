import http.client
import urllib.error

import pytest

from ucl.llm import ChatResult, LLMError, LMStudio


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


def test_ensure_ready_never_raises(tmp_path, monkeypatch):
    llm = LMStudio(base_url="http://127.0.0.1:9/v1", cache_dir=tmp_path)
    monkeypatch.setattr(llm, "_lms", lambda *a, **k: False)
    assert llm.ensure_ready() is False


def test_ensure_ready_when_already_up_does_not_call_lms(tmp_path, monkeypatch):
    llm = LMStudio(cache_dir=tmp_path)
    monkeypatch.setattr(llm, "_server_up", lambda: True)
    monkeypatch.setattr(llm, "_model_loaded", lambda: True)

    def no_lms(*args, **kwargs):
        raise AssertionError("lms must not run")

    monkeypatch.setattr(llm, "_lms", no_lms)
    assert llm.ensure_ready() is True


def test_ensure_ready_loads_the_model_with_its_context_length(tmp_path, monkeypatch):
    llm = LMStudio(model="m/key", cache_dir=tmp_path)
    state, calls = {"loaded": False}, []

    def fake_lms(*args, timeout):
        calls.append(args)
        state["loaded"] = True
        return True

    monkeypatch.setattr(llm, "_server_up", lambda: True)
    monkeypatch.setattr(llm, "_model_loaded", lambda: state["loaded"])
    monkeypatch.setattr(llm, "_lms", fake_lms)
    assert llm.ensure_ready() is True
    assert calls == [("load", "m/key", "--context-length", "16384", "-y")]


def test_ensure_ready_gives_up_when_the_load_command_fails(tmp_path, monkeypatch):
    llm = LMStudio(cache_dir=tmp_path)
    checks = []
    monkeypatch.setattr(llm, "_server_up", lambda: True)
    monkeypatch.setattr(llm, "_model_loaded", lambda: checks.append(1) or False)
    monkeypatch.setattr(llm, "_lms", lambda *args, timeout: False)
    assert llm.ensure_ready() is False
    assert len(checks) == 1  # gave up at once instead of polling
