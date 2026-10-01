import threading
from itertools import islice
from pathlib import Path

import pytest

from ucl.web.events import EventBus


def drain(subscription):
    """Ids of the events waiting in a subscription, without blocking."""
    ids = []
    while (event := subscription.get(0)) is not None:
        ids.append(event.id)
    return ids


def test_ids_are_boot_scoped_and_increase():
    bus = EventBus(boot="b1", clock=lambda: 5.0)
    first = bus.publish("log", {"level": "info", "text": "hi"}, run_id="b1-r1", stage="fetch")
    second = bus.publish("done", {"status": "done", "stages": []}, run_id="b1-r1")
    assert (first.id, second.id) == ("b1-1", "b1-2")
    assert first.to_dict() == {"id": "b1-1", "run_id": "b1-r1", "ts": 5.0, "type": "log", "stage": "fetch",
                               "data": {"level": "info", "text": "hi"}}


def test_an_event_is_one_sse_frame_with_id_event_and_json_data():
    bus = EventBus(boot="b1", clock=lambda: 5.0)
    event = bus.publish("stage", {"name": "fetch", "status": "running"})
    assert event.sse() == ('id: b1-1\nevent: stage\ndata: {"id": "b1-1", "run_id": null, "ts": 5.0, '
                           '"type": "stage", "stage": null, "data": {"name": "fetch", "status": "running"}}\n\n')


def test_payloads_are_made_json_safe_and_bad_ones_fail_when_published():
    bus = EventBus(boot="b1")
    assert bus.publish("progress", {"counters": {"x": float("nan")}}).data == {"counters": {"x": None}}
    with pytest.raises(ValueError):
        bus.publish("error", {})  # EventSource reserves "error" for connection errors
    with pytest.raises(TypeError):
        bus.publish("log", {"text": Path("x")})
    assert drain(bus.subscribe()) == ["b1-1"]  # the bad event took no id and reached nobody


def test_a_new_subscriber_replays_the_buffer_or_resumes_after_its_last_id():
    bus = EventBus(boot="b1", buffer=3)
    for i in range(5):
        bus.publish("log", {"text": str(i)})
    assert drain(bus.subscribe()) == ["b1-3", "b1-4", "b1-5"]  # the buffer keeps the last 3
    assert drain(bus.subscribe("b1-4")) == ["b1-5"]
    assert drain(bus.subscribe("a0-9")) == ["b1-3", "b1-4", "b1-5"]  # an id from an earlier server: replay all
    assert drain(bus.subscribe("garbage")) == ["b1-3", "b1-4", "b1-5"]
    assert drain(bus.subscribe("b1-²")) == ["b1-3", "b1-4", "b1-5"]


def test_unkept_events_reach_live_subscribers_but_are_not_replayed():
    bus = EventBus(boot="b1")
    live = bus.subscribe()
    bus.publish("progress", {"counters": {}}, keep=False)
    bus.publish("log", {"text": "kept"})
    assert drain(live) == ["b1-1", "b1-2"]
    assert drain(bus.subscribe()) == ["b1-2"]


def test_a_subscriber_that_falls_behind_loses_its_oldest_events():
    bus = EventBus(boot="b1", queue_max=2)
    subscription = bus.subscribe()
    for i in range(4):
        bus.publish("log", {"text": str(i)})
    assert drain(subscription) == ["b1-3", "b1-4"]


def test_the_stream_sends_events_then_pings_when_idle():
    bus = EventBus(boot="b1")
    bus.publish("log", {"text": "x"})
    stream = bus.stream(bus.subscribe(), heartbeat=0.01)
    assert next(stream).startswith("id: b1-1\nevent: log\n")
    assert next(stream) == ": ping\n\n"
    stream.close()


def test_a_live_event_wakes_a_waiting_stream():
    bus = EventBus(boot="b1")
    stream = bus.stream(bus.subscribe(), heartbeat=5)
    threading.Timer(0.05, lambda: bus.publish("log", {"text": "late"})).start()
    assert next(stream).startswith("id: b1-1\nevent: log\n")
    stream.close()


def test_a_stream_closed_by_its_client_unsubscribes():
    bus = EventBus(boot="b1")
    bus.publish("log", {"text": "x"})
    stream = bus.stream(bus.subscribe(), heartbeat=0.01)
    next(stream)
    assert bus.subscribers == 1
    stream.close()
    assert bus.subscribers == 0


def test_close_ends_every_stream_and_later_subscribers_end_after_the_backlog():
    bus = EventBus(boot="b1")
    bus.publish("log", {"text": "x"})
    stream = bus.stream(bus.subscribe("b1-1"), heartbeat=5)
    threading.Timer(0.05, bus.close).start()
    assert list(islice(stream, 1)) == []
    assert bus.subscribers == 0
    late = bus.stream(bus.subscribe(), heartbeat=5)
    assert [frame[:9] for frame in islice(late, 2)] == ["id: b1-1\n"]
