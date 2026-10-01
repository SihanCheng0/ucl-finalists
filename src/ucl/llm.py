"""LM Studio client: readiness checks and cached chat completions (spec §7)."""
from __future__ import annotations

import contextlib
import hashlib
import http.client
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import config

Post = Callable[[str, dict, float], dict]
# OSError covers URLError, TimeoutError and ConnectionError; the rest cover malformed responses.
CALL_ERRORS = (OSError, http.client.HTTPException, KeyError, IndexError, TypeError, ValueError, AttributeError)


class LLMError(Exception):
    """A chat call failed: network, timeout, HTTP error or malformed response."""


@dataclass(frozen=True)
class ChatResult:
    content: str
    finish_reason: str | None


def http_post_json(url: str, body: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
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


def _first_line(text: str, limit: int = 200) -> str:
    """The first non-blank line of `text`, trimmed and capped, so that a failure reason stays short."""
    return next((line.strip() for line in text.splitlines() if line.strip()), "")[:limit]


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


class LMStudio:
    def __init__(
        self,
        model: str = config.LLM_MODEL,
        base_url: str = config.LLM_BASE_URL,
        cache_dir: Path = config.LLM_CACHE_DIR,
        post: Post = http_post_json,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self._post = post
        self.reason: str | None = None  # why ensure_ready last said no
        self._lms_error = ""  # first line lms printed (or why it could not run) on its last failed call

    def chat(self, messages: list[dict]) -> ChatResult:
        """Every response, valid or not, is cached under its full request; failures are not."""
        body = {"model": self.model, "messages": messages, **config.LLM_PARAMS}
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        cached = _read_cache(path)
        if cached is not None:
            return cached
        try:
            choice = self._post(f"{self.base_url}/chat/completions", body, config.LLM_TIMEOUT_S)["choices"][0]
            content = choice["message"].get("content")
            if content is not None and not isinstance(content, str):
                raise LLMError(f"message content is {type(content).__name__}, not text")
            result = ChatResult(content or "", choice.get("finish_reason"))
        except CALL_ERRORS as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        _write_cache(path, {"content": result.content, "finish_reason": result.finish_reason, "request": body})
        return result

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
        self._lms_error = "" if code == 0 else (_first_line(output) or f"exit status {code}")
        return code == 0

    def _lms_output(self, *args: str, timeout: float) -> str | None:
        code, output = self._run_lms(*args, timeout=timeout)
        return output if code == 0 else None
