"""Pipeline events for the browser: boot-scoped ids, a replay buffer, bounded per-subscriber queues and SSE
framing (spec §4.3)."""
from __future__ import annotations

import json
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from .jsonsafe import to_jsonable

BUFFER = 500
QUEUE_MAX = 2000
HEARTBEAT_S = 15.0
TYPES = ("stage", "progress", "log", "done", "run_failed")  # never "error": EventSource uses it for the connection


@dataclass(frozen=True)
class Event:
    id: str  # "{boot}-{n}"
    run_id: str | None
    ts: float
    type: str
    stage: str | None
    data: dict

    @property
    def n(self) -> int:
        return int(self.id.rpartition("-")[2])

    def to_dict(self) -> dict:
        return {"id": self.id, "run_id": self.run_id, "ts": self.ts, "type": self.type, "stage": self.stage,
                "data": self.data}

    def sse(self) -> str:
        return f"id: {self.id}\nevent: {self.type}\ndata: {json.dumps(self.to_dict(), allow_nan=False)}\n\n"


class Subscription:
    """One client's queue: its backlog first, then live events. When it falls `maxlen` behind, the oldest go."""

    def __init__(self, backlog: list[Event], maxlen: int):
        self._items: deque[Event] = deque(backlog, maxlen=maxlen)
        self._ready = threading.Condition()
        self.closed = False

    def push(self, event: Event) -> None:
        with self._ready:
            self._items.append(event)
            self._ready.notify()

    def close(self) -> None:
        with self._ready:
            self.closed = True
            self._ready.notify_all()

    def get(self, timeout: float) -> Event | None:
        """The next event; None after `timeout` seconds with nothing new, or once closed and drained."""
        with self._ready:
            if not self._items and not self.closed:
                self._ready.wait(timeout)
            return self._items.popleft() if self._items else None


class EventBus:
    def __init__(self, boot: str | None = None, buffer: int = BUFFER, queue_max: int = QUEUE_MAX,
                 clock: Callable[[], float] = time.time):
        self.boot = boot or secrets.token_hex(4)
        self._buffer: deque[Event] = deque(maxlen=buffer)
        self._subscriptions: set[Subscription] = set()
        self._lock = threading.Lock()
        self._n = 0
        self._queue_max = queue_max
        self._clock = clock
        self.closed = False

    @property
    def subscribers(self) -> int:
        with self._lock:
            return len(self._subscriptions)

    def publish(self, type: str, data: dict, run_id: str | None = None, stage: str | None = None,
                keep: bool = True) -> Event:
        """Send an event to every subscriber. `keep=False` leaves it out of the replay buffer: progress events
        are superseded by /state, and would otherwise push log lines out of the buffer."""
        if type not in TYPES:
            raise ValueError(f"unknown event type {type!r}")
        payload = to_jsonable(data)
        json.dumps(payload, allow_nan=False)  # a payload that can't be sent fails here, not in every stream
        with self._lock:  # ids, the buffer and every queue advance together, so every client sees one order
            self._n += 1
            event = Event(f"{self.boot}-{self._n}", run_id, self._clock(), type, stage, payload)
            if keep:
                self._buffer.append(event)
            for subscription in self._subscriptions:
                subscription.push(event)
        return event

    def subscribe(self, last_event_id: str | None = None) -> Subscription:
        """A new subscription. It starts after `last_event_id` when that id is from this boot, and otherwise with
        the whole buffer. The backlog is taken under the same lock as the registration, so no event can fall
        between the two."""
        with self._lock:
            backlog = list(self._buffer)
            after = _sequence(last_event_id, self.boot)
            if after is not None:
                backlog = [event for event in backlog if event.n > after]
            subscription = Subscription(backlog, self._queue_max)
            if self.closed:
                subscription.close()
            else:
                self._subscriptions.add(subscription)
            return subscription

    def unsubscribe(self, subscription: Subscription) -> None:
        with self._lock:
            self._subscriptions.discard(subscription)

    def close(self) -> None:
        """End every stream (at shutdown). Later subscriptions get the backlog, then the end."""
        with self._lock:
            self.closed = True
            subscriptions, self._subscriptions = list(self._subscriptions), set()
        for subscription in subscriptions:
            subscription.close()

    def stream(self, subscription: Subscription, heartbeat: float = HEARTBEAT_S) -> Iterator[str]:
        """SSE text for one subscription: each event as it comes, and a ': ping' comment after `heartbeat` idle
        seconds. It never ends on `done`, only when the bus closes, and it unsubscribes however it ends."""
        try:
            while True:
                event = subscription.get(heartbeat)
                if event is not None:
                    yield event.sse()
                elif subscription.closed:
                    return
                else:
                    yield ": ping\n\n"
        finally:
            self.unsubscribe(subscription)


def _sequence(last_event_id: str | None, boot: str) -> int | None:
    """The n of a Last-Event-ID from this boot; None for no id, a malformed one, or one from an earlier server."""
    if not last_event_id:
        return None
    id_boot, _, n = last_event_id.rpartition("-")
    return int(n) if id_boot == boot and n.isdecimal() and n.isascii() else None
