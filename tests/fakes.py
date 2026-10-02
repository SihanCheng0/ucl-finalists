"""Test doubles shared by the analyst tests."""
import json

from ucl.llm import CacheMiss


class FakeLLM:
    """Scripted replies; caches by request like the real client, so re-runs replay."""

    def __init__(self, script):
        self.script, self.requests, self.cache = list(script), [], {}

    def ensure_ready(self):
        return True

    def chat(self, messages, offline=False):
        key = json.dumps(messages, sort_keys=True)
        if key in self.cache:
            return self.cache[key]
        if offline:
            raise CacheMiss(key)
        self.requests.append(messages)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        self.cache[key] = item
        return item
