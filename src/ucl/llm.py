"""The model clients, LM Studio on this machine or OpenRouter's hosted copy of the same model: readiness checks and
cached chat completions (spec §7). Both ask the same question (model, messages, LLM_PARAMS) under the same cache
key, so an answer saved from one replays for the other."""
from __future__ import annotations

import contextlib
import hashlib
import http.client
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import config

Post = Callable[..., dict]  # (url, body, timeout[, headers]) -> the JSON response
Get = Callable[[str, float, dict], dict]  # (url, timeout, headers) -> the JSON response
# OSError covers URLError, TimeoutError and ConnectionError; the rest cover malformed responses.
CALL_ERRORS = (OSError, http.client.HTTPException, KeyError, IndexError, TypeError, ValueError, AttributeError)
PROVIDERS = ("lmstudio", "openrouter")


class LLMError(Exception):
    """A chat call failed: network, timeout, HTTP error or malformed response."""


class CacheMiss(Exception):
    """An offline chat found no saved answer for its request. Not an LLMError: nothing failed, nothing was asked."""


@dataclass(frozen=True)
class ChatResult:
    content: str
    finish_reason: str | None


@dataclass
class Usage:
    """What the questions this client actually sent used up. Only OpenRouter reports a cost (in US dollars)."""
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float | None = None

    def add(self, usage: dict | None) -> None:
        """Count one answered call. Never raises: an answer that was paid for must still be saved."""
        usage = usage if isinstance(usage, dict) else {}
        self.calls += 1
        self.input_tokens += _count(usage.get("prompt_tokens"))
        self.output_tokens += _count(usage.get("completion_tokens"))
        if _number(usage.get("cost")):
            self.cost = (self.cost or 0.0) + float(usage["cost"])

    def describe(self) -> str:
        cost = "" if self.cost is None else f", ${self.cost:.4f}"
        return (f"{self.calls} call{'s' if self.calls != 1 else ''}, {self.input_tokens:,} input and "
                f"{self.output_tokens:,} output tokens{cost}")


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _count(value) -> int:
    return int(value) if _number(value) and value >= 0 else 0


def key_problem(key: str) -> str | None:
    """Why an API key can't be sent, without repeating any of it: none set, or spaces or line breaks pasted in with
    it (a header carrying those fails with an error that would quote the key)."""
    if not key:
        return f"{config.OPENROUTER_KEY_ENV} isn't set"
    if any(c.isspace() or not c.isprintable() for c in key):
        return f"{config.OPENROUTER_KEY_ENV} has spaces or line breaks in it: set it again without them"
    return None


def http_post_json(url: str, body: dict, timeout: float, headers: dict | None = None) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **(headers or {})}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def http_get_json(url: str, timeout: float, headers: dict | None = None) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/json", **(headers or {})})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _http_ok(url: str, timeout: float) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except (OSError, http.client.HTTPException):
        return False


def _wait(check: Callable[[], bool], seconds: float, interval: float = 1.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(interval)
    return check()


def _read_cache(path: Path) -> ChatResult | None:
    """A cached answer, or None when the entry is missing, unreadable or not a valid answer."""
    try:
        entry = json.loads(path.read_text())
        content, finish_reason = entry["content"], entry["finish_reason"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return ChatResult(content, finish_reason) if isinstance(content, str) else None


def _write_cache(path: Path, entry: dict) -> None:
    """Best effort: a cache that cannot be written must not lose the answer it was given."""
    # a name of its own, so that overlapping runs cannot clobber each other's half-written file
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(entry))
        os.replace(tmp, path)
    except OSError:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


TERMINAL_CODES = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _failure_reason(text: str, limit: int = 200) -> str:
    """Why an lms call failed, from what it printed: without terminal codes or progress spinners, the last line that
    mentions an error (lms prints its summary first and the cause after), else the first line; trimmed and capped."""
    lines = [line.strip() for line in TERMINAL_CODES.sub("", text).replace("\r", "\n").splitlines() if line.strip()]
    errors = [line for line in lines if "error" in line.lower()]
    reason = errors[-1] if errors else (lines[0] if lines else "")
    return reason.replace(str(Path.home()), "~")[:limit]


def parse_loaded(ps_json: str, model: str) -> int | None:
    """Context length of the loaded model whose identifier is exactly `model`, from `lms ps --json`."""
    try:
        entries = json.loads(ps_json)
    except (TypeError, ValueError):
        return None
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if isinstance(entry, dict) and entry.get("identifier") == model:
            length = entry.get("contextLength")
            return length if type(length) is int else None  # exactly int: bool is an int subclass
    return None


class CachedChat:
    """Chat completions that answer from the cache first. A subclass says how to send one request (`send`, which
    skips the cache) and how to get ready for one (`ensure_ready`, which never raises and leaves its reason in
    `reason`)."""
    name = "the model"

    def __init__(self, model: str, cache_dir: Path, post: Post):
        self.model = model
        self.cache_dir = Path(cache_dir)
        self._post = post
        self.reason: str | None = None  # why ensure_ready last said no
        self.spent = Usage()

    def chat(self, messages: list[dict], offline: bool = False) -> ChatResult:
        """Every response, valid or not, is cached under its full request; failures are not. `offline` answers from
        the cache only and raises CacheMiss instead of asking the model."""
        body = {"model": self.model, "messages": messages, **config.LLM_PARAMS}
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        cached = _read_cache(path)
        if cached is not None:
            return cached
        if offline:
            raise CacheMiss(key)
        try:
            response = self.send(body)
            choice = response["choices"][0]
            if choice.get("error") or choice.get("finish_reason") == "error":  # OpenRouter: the provider failed
                error = choice.get("error")
                detail = error.get("message") if isinstance(error, dict) else error
                raise LLMError(f"the provider failed mid-answer: {self.scrub(str(detail or 'no detail'))}")
            content = choice["message"].get("content")
            if content is not None and not isinstance(content, str):
                raise LLMError(f"message content is {type(content).__name__}, not text")
            result = ChatResult(content or "", choice.get("finish_reason"))
        except CALL_ERRORS as exc:
            raise LLMError(self.scrub(f"{type(exc).__name__}: {exc}")) from None
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else None
        self.spent.add(usage)
        entry = {"content": result.content, "finish_reason": result.finish_reason, "request": body}
        _write_cache(path, {**entry, "usage": usage} if usage else entry)
        return result

    def send(self, body: dict, timeout: float = config.LLM_TIMEOUT_S) -> dict:
        raise NotImplementedError

    def ensure_ready(self) -> bool:
        raise NotImplementedError

    def scrub(self, text: str) -> str:
        """Text about a failure, safe to log and to publish (OpenRouter takes out its key)."""
        return text


class OpenRouter(CachedChat):
    """The same model hosted on OpenRouter, which passes each request to one of the providers serving it. Reasoning
    stays off, as in LM Studio, and each response reports its tokens and cost."""
    name = "OpenRouter"

    def __init__(self, model: str = config.LLM_MODEL, base_url: str = config.OPENROUTER_BASE_URL,
                 cache_dir: Path = config.LLM_CACHE_DIR, post: Post = http_post_json, get: Get = http_get_json,
                 api_key: str | None = None):
        super().__init__(model, cache_dir, post)
        self.base_url = base_url.rstrip("/")
        self._get = get
        self._key = (os.environ.get(config.OPENROUTER_KEY_ENV, "") if api_key is None else api_key).strip()

    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self._key}", "X-Title": "UCL Lab", "User-Agent": "ucl-lab"}

    def scrub(self, text: str) -> str:
        return text.replace(self._key, "<key>") if self._key else text

    def send(self, body: dict, timeout: float = config.LLM_TIMEOUT_S) -> dict:
        problem = key_problem(self._key)
        if problem:  # never put a key that can't be sent into a header: the error would quote it
            raise LLMError(problem)
        return self._post(f"{self.base_url}/chat/completions", openrouter_body(body), timeout, self.headers())

    def ensure_ready(self) -> bool:
        """The key is set and OpenRouter accepts it. Never raises: see `reason`."""
        self.reason = self._not_ready_reason()
        return self.reason is None

    def _not_ready_reason(self) -> str | None:
        problem = key_problem(self._key)
        if problem:
            return problem
        try:
            self._get(f"{self.base_url}/key", config.OPENROUTER_CHECK_TIMEOUT_S, self.headers())
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                return f"OpenRouter turned the key down (HTTP {exc.code})"
            return f"OpenRouter answered HTTP {exc.code}"
        except Exception as exc:  # noqa: BLE001 - readiness must never crash the pipeline
            return self.scrub(f"OpenRouter isn't reachable ({type(exc).__name__}: {exc})")
        return None


def openrouter_body(body: dict) -> dict:
    """An LM Studio request in OpenRouter's terms: `reasoning_effort` becomes its `reasoning` object, only providers
    with full or 8-bit weights may answer, and usage accounting is on so that each response carries its cost."""
    sent = {key: value for key, value in body.items() if key != "reasoning_effort"}
    effort = body.get("reasoning_effort")
    if effort is not None:
        sent["reasoning"] = {"enabled": False} if effort == "none" else {"effort": effort}
    sent["provider"] = {"quantizations": list(config.OPENROUTER_QUANTIZATIONS)}
    sent["usage"] = {"include": True}
    return sent


def provider() -> str:
    """Who answers new questions: UCL_LLM_PROVIDER when it is set, else OpenRouter when its key is, else LM Studio."""
    chosen = os.environ.get(config.LLM_PROVIDER_ENV, "").strip().lower()
    if chosen:
        if chosen not in PROVIDERS:
            raise ValueError(f"{config.LLM_PROVIDER_ENV} must be one of {', '.join(PROVIDERS)}, not {chosen!r}")
        return chosen
    return "openrouter" if os.environ.get(config.OPENROUTER_KEY_ENV) else "lmstudio"


def make_llm(model: str = config.LLM_MODEL) -> CachedChat:
    return OpenRouter(model=model) if provider() == "openrouter" else LMStudio(model=model)


class LMStudio(CachedChat):
    name = "LM Studio"

    def __init__(
        self,
        model: str = config.LLM_MODEL,
        base_url: str = config.LLM_BASE_URL,
        cache_dir: Path = config.LLM_CACHE_DIR,
        post: Post = http_post_json,
    ):
        super().__init__(model, cache_dir, post)
        self.base_url = base_url.rstrip("/")
        self._lms_error = ""  # what lms said went wrong (or why it could not run) on its last failed call

    def send(self, body: dict, timeout: float = config.LLM_TIMEOUT_S) -> dict:
        return self._post(f"{self.base_url}/chat/completions", body, timeout)

    def ensure_ready(self) -> bool:
        """Start the server and load the model with the configured context if needed. Never raises: see `reason`."""
        try:
            self.reason = self._not_ready_reason()
        except Exception as exc:  # noqa: BLE001 - readiness must never crash the pipeline
            self.reason = f"{type(exc).__name__}: {exc}"
        return self.reason is None

    def _not_ready_reason(self) -> str | None:
        """Why the server and model are still unusable after trying to fix that, or None when they are usable."""
        want = config.LLM_CONTEXT_LENGTH
        if not self._server_up():
            if not self._lms("server", "start", timeout=60):
                return self._lms_failure("server did not start")
            if not _wait(self._server_up, 30):
                return "server did not start"
        loaded = self._loaded_context()
        if loaded is not None and loaded >= want:
            return None
        # loaded with too small a context (LM Studio defaults to 4096): reload it rather than load a second copy
        if loaded is not None and not self._lms("unload", self.model, timeout=60):
            return self._lms_failure("unload failed")
        if not self._lms("load", self.model, "--context-length", str(want), "-y", timeout=200):
            return self._lms_failure("load failed")
        if not _wait(lambda: (self._loaded_context() or 0) >= want, 30):
            return f"model not ready with a context of {want} after loading"
        return None

    def _lms_failure(self, what: str) -> str:
        """`what` plus the detail of the `_lms` call that just failed."""
        return f"{what}: {self._lms_error}" if self._lms_error else what

    def _server_up(self) -> bool:
        return _http_ok(f"{self.base_url}/models", timeout=3)

    def _loaded_context(self) -> int | None:
        """The context length the model is loaded with; None when it is not loaded (or `lms ps` failed)."""
        output = self._lms_output("ps", "--json", timeout=10)
        return None if output is None else parse_loaded(output, self.model)

    def _lms_path(self) -> str | None:
        return str(config.LMS_BIN) if config.LMS_BIN.exists() else shutil.which("lms")

    def _run_lms(self, *args: str, timeout: float) -> tuple[int | None, str]:
        """(exit code, stdout+stderr) of an lms call; (None, why) if lms is missing, cannot run or times out."""
        exe = self._lms_path()
        if not exe:
            return None, "lms not found"
        try:
            # a file, not a pipe: a daemonising `lms server start` would hold a pipe open and make run() wait it out
            with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as out:
                code = subprocess.run([exe, *args], stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                      timeout=timeout).returncode
                out.seek(0)
                return code, out.read()
        except subprocess.TimeoutExpired:
            return None, f"timed out after {timeout:g}s"
        except OSError as exc:
            return None, f"could not run lms: {exc}"

    def _lms(self, *args: str, timeout: float) -> bool:
        code, output = self._run_lms(*args, timeout=timeout)
        self._lms_error = "" if code == 0 else (_failure_reason(output) or f"exit status {code}")
        return code == 0

    def _lms_output(self, *args: str, timeout: float) -> str | None:
        code, output = self._run_lms(*args, timeout=timeout)
        return output if code == 0 else None
