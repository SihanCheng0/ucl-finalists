"""LM Studio client: readiness checks and cached chat completions (spec §7)."""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import shutil
import subprocess
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

    def chat(self, messages: list[dict]) -> ChatResult:
        """Every response, valid or not, is cached under its full request; failures are not."""
        body = {"model": self.model, "messages": messages, **config.LLM_PARAMS}
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            cached = json.loads(path.read_text())
            return ChatResult(cached["content"], cached["finish_reason"])
        try:
            choice = self._post(f"{self.base_url}/chat/completions", body, config.LLM_TIMEOUT_S)["choices"][0]
            result = ChatResult(choice["message"].get("content") or "", choice.get("finish_reason"))
        except CALL_ERRORS as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps({"content": result.content, "finish_reason": result.finish_reason,
                                   "request": body}))
        os.replace(tmp, path)
        return result

    def ensure_ready(self) -> bool:
        """Start the server and load the model if needed. Never raises."""
        try:
            if not self._server_up():
                if not self._lms("server", "start", timeout=60) or not _wait(self._server_up, 30):
                    return False
            if not self._model_loaded():
                loaded = self._lms(
                    "load", self.model, "--context-length", str(config.LLM_CONTEXT_LENGTH), "-y", timeout=200
                )
                if not loaded or not _wait(self._model_loaded, 30):
                    return False
            return True
        except Exception:  # noqa: BLE001 - readiness must never crash the pipeline
            return False

    def _server_up(self) -> bool:
        return _http_ok(f"{self.base_url}/models", timeout=3)

    def _model_loaded(self) -> bool:
        output = self._lms_output("ps")
        return output is not None and self.model in output

    def _lms_path(self) -> str | None:
        return str(config.LMS_BIN) if config.LMS_BIN.exists() else shutil.which("lms")

    def _lms(self, *args: str, timeout: float) -> bool:
        exe = self._lms_path()
        if not exe:
            return False
        try:
            return subprocess.run([exe, *args], capture_output=True, text=True, timeout=timeout).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def _lms_output(self, *args: str) -> str | None:
        exe = self._lms_path()
        if not exe:
            return None
        try:
            return subprocess.run([exe, *args], capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.TimeoutExpired):
            return None
