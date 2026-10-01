import threading

import pytest

from ucl.web.events import EventBus
from ucl.web.pipeline import CORE, BadRun, PipelineRunner, RunInProgress, StageOutcome


class Recorder:
    """Every event the bus sends, captured from before the run starts (progress events are never replayed)."""

    def __init__(self, bus):
        self._subscription, self.events = bus.subscribe(), []

    def of(self, type=None):
        while (event := self._subscription.get(0)) is not None:
            self.events.append(event)
        return [e for e in self.events if type is None or e.type == type]


def statuses(runner):
    return {s["name"]: s["status"] for s in runner.state()["stages"]}


def recorder(calls, name, outcome=None, error=None):
    def stage(ctx):
        calls.append(name)
        ctx.log(f"{name} says hi")
        if error is not None:
            raise error
        return outcome or StageOutcome("done", f"{name} ok")
    return stage


def make(overrides=None, **kwargs):
    calls = []
    registry = {name: recorder(calls, name) for name in CORE}
    registry.update(overrides or {})
    bus = EventBus(boot="b")
    return PipelineRunner(registry, bus, **kwargs), Recorder(bus), calls


def test_the_default_run_is_the_five_core_stages_in_order():
    runner, events, calls = make()
    run_id = runner.start()
    assert runner.wait(5)
    assert run_id == "b-r1" and calls == CORE
    assert set(statuses(runner).values()) == {"done"}
    assert [e.data["status"] for e in events.of("done")] == ["done"]


def test_optional_stages_run_only_when_asked_for_and_in_canonical_order():
    extra = []
    runner, events, calls = make({"players": recorder(extra, "players")})
    runner.start()
    assert runner.wait(5) and extra == []
    runner.start(["players", "report"])
    assert runner.wait(5) and extra == ["players"] and calls[-1] == "report"  # report runs before players
    assert list(statuses(runner)) == [*CORE, "players"]


def test_requested_stages_run_in_canonical_order_once_each():
    runner, events, calls = make()
    runner.start(["report", "fetch", "fetch"])
    assert runner.wait(5)
    assert calls == ["fetch", "report"]
    assert statuses(runner) == {"fetch": "done", "build": "idle", "model": "idle", "analyze": "idle",
                                "report": "done"}


def test_bad_requests_are_refused_before_anything_runs():
    def missing(names):
        return "model needs data/processed/dataset.json: run build first" if names == ["model"] else None

    runner, events, calls = make(missing_inputs=missing)
    with pytest.raises(BadRun, match="unknown stage"):
        runner.start(["fetch", "train", 3])
    with pytest.raises(BadRun, match="run build first"):
        runner.start(["model"])
    with pytest.raises(BadRun, match="no stages"):
        runner.start([])
    assert calls == [] and runner.state()["running"] is False


def test_a_failed_stage_stops_the_run_and_lists_its_errors():
    failing = lambda ctx: StageOutcome("failed", "dataset validation failed", ["2016 Roma: a", "2017 Ajax: b"])
    runner, events, calls = make({"build": failing})
    runner.start()
    assert runner.wait(5)
    assert calls == ["fetch"]
    assert [e.data for e in events.of("run_failed")] == [
        {"stage": "build", "errors": ["2016 Roma: a", "2017 Ajax: b"]}]
    assert events.of("done") == []
    assert statuses(runner)["build"] == "failed" and statuses(runner)["model"] == "idle"


def test_an_exception_fails_the_stage_with_its_message():
    runner, events, calls = make({"model": recorder([], "model", error=ValueError("bad fold"))})
    runner.start()
    assert runner.wait(5)
    model = next(s for s in runner.state()["stages"] if s["name"] == "model")
    assert (model["status"], model["message"]) == ("failed", "ValueError: bad fold")
    assert [e.data for e in events.of("run_failed")] == [{"stage": "model", "errors": ["ValueError: bad fold"]}]
    assert not runner.state()["running"]


def test_a_stage_that_returns_nonsense_fails_instead_of_hanging():
    runner, events, calls = make({"build": lambda ctx: None})
    runner.start()
    assert runner.wait(5)
    assert statuses(runner)["build"] == "failed"
    assert events.of("run_failed")[0].data["errors"][0].startswith("TypeError: stage build returned None")


def test_a_warning_continues_and_marks_the_whole_run():
    runner, events, calls = make({"analyze": lambda ctx: StageOutcome("warning", "LM Studio not ready")})
    runner.start()
    assert runner.wait(5)
    assert calls == ["fetch", "build", "model", "report"]
    done = events.of("done")[0].data
    assert done["status"] == "warning" and {"name": "analyze", "status": "warning"} in done["stages"]
    assert {"level": "warn", "text": "analyze: LM Studio not ready"} in [e.data for e in events.of("log")]


def test_skip_ai_marks_analyze_skipped_without_calling_it():
    runner, events, calls = make()
    runner.start(skip_ai=True)
    assert runner.wait(5)
    assert "analyze" not in calls and statuses(runner)["analyze"] == "skipped"


def test_two_runs_cannot_overlap_and_a_busy_runner_says_so_first():
    gate = threading.Event()
    runner, events, calls = make({"fetch": lambda ctx: StageOutcome("done") if gate.wait(5) else None},
                                 missing_inputs=lambda names: "run build first" if names == ["model"] else None)
    barrier, results = threading.Barrier(4), []

    def attempt():
        barrier.wait()
        try:
            results.append(runner.start(["fetch"]))
        except RunInProgress:
            results.append(409)

    threads = [threading.Thread(target=attempt) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(results, key=str) == [409, 409, 409, "b-r1"]
    with pytest.raises(RunInProgress):
        runner.start(["model"])  # busy beats "missing inputs"
    assert runner._thread.daemon  # a Ctrl-C never waits on a run
    gate.set()
    assert runner.wait(5)
    assert runner.start(["fetch"]) == "b-r2"  # free again once the run ended
    assert runner.wait(5)


def test_the_last_event_goes_out_after_the_run_is_marked_finished():
    seen = []

    class WatchingBus(EventBus):
        def publish(self, type, *args, **kwargs):
            if type == "done":
                seen.append(runner._running)
            return super().publish(type, *args, **kwargs)

    runner = PipelineRunner({name: (lambda ctx: StageOutcome("done")) for name in CORE}, WatchingBus(boot="b"))
    runner.start()
    assert runner.wait(5) and seen == [False]


def test_the_run_ends_cleanly_even_when_publishing_fails():
    class BrokenBus(EventBus):
        def publish(self, *args, **kwargs):
            raise RuntimeError("bus down")

    runner = PipelineRunner({name: (lambda ctx: StageOutcome("done")) for name in CORE}, BrokenBus(boot="b"))
    runner.start()
    assert runner.wait(5)
    assert set(statuses(runner).values()) == {"done"}


def test_state_is_a_deep_copy():
    runner, events, calls = make()
    state = runner.state()
    state["stages"][0]["status"] = "hacked"
    state["counters"]["requests"]["network"] = 99
    assert runner.state()["stages"][0]["status"] == "idle"
    assert runner.state()["counters"]["requests"]["network"] == 0
    assert runner.state()["boot"] == "b" and runner.state()["run_id"] is None


def test_counters_move_with_the_stages_and_reset_per_run():
    def fetch(ctx):
        ctx.bump("requests", "network", 3)
        ctx.bump("seasons")
        ctx.set("validated", True)
        return StageOutcome("done")

    runner, events, calls = make({"fetch": fetch})
    runner.start(["fetch"])
    assert runner.wait(5)
    counters = runner.state()["counters"]
    assert counters["requests"] == {"network": 3, "cache": 0}
    assert counters["seasons"] == {"done": 1, "of": 15} and counters["validated"] is True
    assert counters["folds"] == {"done": 0, "of": 15} and counters["narratives"] == {"done": 0, "of": 11}
    runner.start(["report"])
    assert runner.wait(5)
    assert runner.state()["counters"]["requests"]["network"] == 0
    assert events.of("progress")[-1].data["counters"]["requests"]["network"] == 0  # the reset was announced


def test_progress_is_coalesced_and_flushed_at_the_end_of_each_stage():
    now = [0.0]

    def fetch(ctx):
        for _ in range(3):
            ctx.bump("seasons")  # at t=0: the first publishes, the next two wait
        now[0] = 1.0
        ctx.bump("seasons")  # t=1: publishes at once, carrying all four
        ctx.bump("seasons")  # waits, then goes out when the stage ends
        return StageOutcome("done")

    runner, events, calls = make({"fetch": fetch}, clock=lambda: now[0])
    runner.start(["fetch"])
    assert runner.wait(5)
    assert [e.data["counters"]["seasons"]["done"] for e in events.of("progress")] == [0, 1, 4, 5]


def test_stage_events_carry_status_times_and_message():
    wall = iter([100.0, 101.5])
    runner, events, calls = make(wall=lambda: next(wall))
    runner.start(["fetch"])
    assert runner.wait(5)
    assert [e.data for e in events.of("stage")] == [
        {"name": "fetch", "status": "running", "started": 100.0, "finished": None, "message": ""},
        {"name": "fetch", "status": "done", "started": 100.0, "finished": 101.5, "message": "fetch ok"},
    ]
    assert [e.data for e in events.of("log")] == [{"level": "info", "text": "fetch says hi"}]
    assert {e.run_id for e in events.of()} == {"b-r1"}
