"""One pipeline run at a time, on a daemon thread. Stage statuses, counters and log lines go out through the
event bus (spec §4.3). The stages are injected (services.py has the real ones), so this file knows nothing
about UEFA, models or LM Studio.

Lock rule: stage and progress events are published while holding the runner's lock, so they leave in the same
order as the state changes. The bus, the clocks and `missing_inputs` must never call back into the runner."""
from __future__ import annotations

import copy
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .. import config
from .events import EventBus

CORE = ["fetch", "build", "model", "analyze", "report"]
ORDER = [*CORE, "live", "players"]  # canonical order; the last two are optional and never run by default
STAGE_LABELS = {"fetch": "Fetch", "build": "Build", "model": "Model", "analyze": "Analyze", "report": "Report",
                "live": "Live season", "players": "Player index"}
OUTCOMES = ("done", "warning", "failed")
PROGRESS_EVERY_S = 0.25


class RunInProgress(Exception):
    """A run is already going (HTTP 409)."""


class BadRun(Exception):
    """The requested run can't start (HTTP 422): unknown stages, none at all, or missing inputs."""


@dataclass
class StageOutcome:
    status: str  # "done" | "warning" | "failed"
    message: str = ""
    errors: list[str] = field(default_factory=list)


def fresh_counters() -> dict:
    return {
        "requests": {"network": 0, "cache": 0},
        "seasons": {"done": 0, "of": len(config.SEASONS)},
        "validated": None,
        "folds": {"done": 0, "of": len(config.SEASONS)},
        "ablation": {"done": 0, "of": 7},
        "narratives": {"done": 0, "of": 2 * len(config.TARGET_SEASONS) + 1},
        "players": {"done": 0, "of": 0},
    }


class StageContext:
    """What a running stage may do: write log lines and move counters."""

    def __init__(self, runner: PipelineRunner, run_id: str, stage: str):
        self._runner, self._run_id, self.stage = runner, run_id, stage

    def log(self, text: str, level: str = "info") -> None:
        self._runner._emit("log", {"level": level, "text": text}, self._run_id, self.stage)

    def bump(self, group: str, key: str = "done", by: int = 1) -> None:
        def change(counters: dict) -> None:
            counters[group][key] += by
        self._runner._update(change)

    def set(self, group: str, value) -> None:
        def change(counters: dict) -> None:
            counters[group] = value
        self._runner._update(change)


Stage = Callable[[StageContext], StageOutcome]


class PipelineRunner:
    def __init__(self, stages: dict[str, Stage], bus: EventBus,
                 missing_inputs: Callable[[list[str]], str | None] = lambda names: None,
                 clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time,
                 counters: Callable[[], dict] = fresh_counters):
        self._stages, self.bus, self._missing_inputs = dict(stages), bus, missing_inputs
        self._names = [name for name in ORDER if name in self._stages]
        self._clock, self._wall = clock, wall  # clock paces progress events; wall stamps stage times
        self._new_counters = counters
        self._lock = threading.Lock()
        self._running = False
        self._run_id: str | None = None
        self._runs = 0
        self._thread: threading.Thread | None = None
        self._states = self._fresh_states()
        self._counters = counters()
        self._last_progress = float("-inf")
        self._dirty = False

    @property
    def stage_names(self) -> list[str]:
        return list(self._names)

    def _fresh_states(self) -> list[dict]:
        return [{"name": name, "status": "idle", "started": None, "finished": None, "message": ""}
                for name in self._names]

    def start(self, stages: list[str] | None = None, skip_ai: bool = False) -> str:
        """Start a run of `stages` (default: the core five) in canonical order. Raises RunInProgress first,
        then BadRun."""
        requested = [name for name in CORE if name in self._stages] if stages is None else list(stages)
        unknown = [str(name) for name in requested if name not in self._stages]
        with self._lock:
            if self._running:
                raise RunInProgress(f"run {self._run_id} is still going")
            if unknown:
                raise BadRun(f"unknown stage(s): {', '.join(unknown)}")
            names = [name for name in self._names if name in requested]
            if not names:
                raise BadRun("no stages to run")
            missing = self._missing_inputs(names)
            if missing:
                raise BadRun(missing)
            self._runs += 1
            run_id = f"{self.bus.boot}-r{self._runs}"
            self._running, self._run_id = True, run_id
            self._states, self._counters = self._fresh_states(), self._new_counters()
            self._last_progress, self._dirty = float("-inf"), False
            self._emit("progress", {"counters": copy.deepcopy(self._counters)}, run_id, keep=False)
            self._thread = threading.Thread(target=self._run, args=(run_id, names, skip_ai), daemon=True,
                                            name=f"pipeline-{run_id}")
            self._thread.start()
        return run_id

    def state(self) -> dict:
        with self._lock:
            return copy.deepcopy({"running": self._running, "run_id": self._run_id, "boot": self.bus.boot,
                                  "stages": self._states, "counters": self._counters})

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for the current run to end; True when no run is going. For tests and shutdown."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
        return not self.state()["running"]

    def _call(self, name: str, run_id: str) -> StageOutcome:
        try:
            outcome = self._stages[name](StageContext(self, run_id, name))
            if not isinstance(outcome, StageOutcome) or outcome.status not in OUTCOMES:
                raise TypeError(f"stage {name} returned {outcome!r}, not a StageOutcome")
            return outcome
        except Exception as exc:  # noqa: BLE001 - any crash fails the stage and ends the run
            return StageOutcome("failed", f"{type(exc).__name__}: {exc}")

    def _run(self, run_id: str, names: list[str], skip_ai: bool) -> None:
        final: tuple[str, dict, str | None] | None = None
        current: str | None = None
        overall = "done"
        try:
            for name in names:
                current = name
                if name == "analyze" and skip_ai:
                    self._set(run_id, name, status="skipped", message="AI write-ups skipped")
                    continue
                self._set(run_id, name, status="running", started=self._wall())
                outcome = self._call(name, run_id)
                self._flush(run_id)
                self._set(run_id, name, status=outcome.status, finished=self._wall(), message=outcome.message)
                if outcome.status == "failed":
                    self._emit("log", {"level": "error", "text": f"{name} failed: {outcome.message}"}, run_id, name)
                    final = ("run_failed", {"stage": name, "errors": outcome.errors or [outcome.message]}, name)
                    return
                if outcome.status == "warning":
                    overall = "warning"
                    self._emit("log", {"level": "warn", "text": f"{name}: {outcome.message}"}, run_id, name)
            final = ("done", {"status": overall}, None)
        except Exception as exc:  # noqa: BLE001 - a fault in the runner itself still ends the run cleanly
            message = f"{type(exc).__name__}: {exc}"
            if current is not None:
                self._set(run_id, current, status="failed", finished=self._wall(), message=message)
            final = ("run_failed", {"stage": current, "errors": [message]}, current)
        finally:
            with self._lock:  # the run is over before anyone hears it is, so /state and new runs agree
                self._running = False
                if final is not None:
                    type, data, stage = final
                    if type == "done":
                        data = {**data, "stages": [{"name": s["name"], "status": s["status"]} for s in self._states]}
                    self._emit(type, data, run_id, stage)

    def _set(self, run_id: str, name: str, **changes) -> None:
        with self._lock:
            state = next(s for s in self._states if s["name"] == name)
            state.update(changes)
            self._emit("stage", dict(state), run_id, name)

    def _update(self, change: Callable[[dict], None]) -> None:
        """Apply a counter change. Publish progress now if the last one went out long enough ago, else at the next
        chance (the next change or the end of the stage); the UI also polls /state while a run is going."""
        with self._lock:
            change(self._counters)
            now = self._clock()
            if now - self._last_progress >= PROGRESS_EVERY_S:
                self._last_progress, self._dirty = now, False
                self._emit("progress", {"counters": copy.deepcopy(self._counters)}, self._run_id, keep=False)
            else:
                self._dirty = True

    def _flush(self, run_id: str) -> None:
        """Publish a counter change still waiting for its slot (at the end of each stage)."""
        with self._lock:
            if self._dirty:
                self._last_progress, self._dirty = self._clock(), False
                self._emit("progress", {"counters": copy.deepcopy(self._counters)}, run_id, keep=False)

    def _emit(self, type: str, data: dict, run_id: str | None, stage: str | None = None, keep: bool = True) -> None:
        """Publish; a bus problem must never change a stage's result."""
        try:
            self.bus.publish(type, data, run_id=run_id, stage=stage, keep=keep)
        except Exception:  # noqa: BLE001
            pass
