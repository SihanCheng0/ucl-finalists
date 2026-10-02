"""The real pipeline, wired into the web runner. Each stage calls stages.py, turns its progress into counters,
and reloads its part of the DataStore (spec §4.3)."""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .. import config, stages
from .events import EventBus
from .pipeline import PipelineRunner, Stage, StageContext, StageOutcome
from .store import DataStore

# analyst.run logs one "  {key}: {status}, {n} call(s), ..." line per narrative
NARRATIVE_LINE = re.compile(r"^\s+\S+: (?:ok|unavailable), \d+ call")
MAX_ERRORS = 20  # failed match ids listed in a run_failed event


@dataclass
class Services:
    store: DataStore
    bus: EventBus
    runner: PipelineRunner
    live: object | None = None  # LiveService (live.py)
    players: object | None = None  # PlayerService (players.py)
    llm_model: str = config.LLM_MODEL

    def live_state(self):
        return self.live.current() if self.live is not None else None

    def team_seasons(self, snapshot, live_state=None) -> pd.DataFrame:
        """The historical team-seasons, plus the live season's rows when there are any."""
        rows = snapshot.dataset.team_seasons
        if live_state is None or live_state.snapshot is None:
            return rows
        return pd.concat([rows, live_state.snapshot.rows], ignore_index=True)


def _shown(path: Path) -> str:
    try:
        return str(path.relative_to(config.ROOT))
    except ValueError:
        return str(path)


def missing_inputs(names: list[str], processed_dir: Path = config.PROCESSED_DIR,
                   out_dir: Path = config.OUT_DIR) -> str | None:
    """Why a run of `names` can't start. A stage needs an output that is neither on disk nor made by an earlier
    stage of the same run."""
    dataset_file, metrics_file = processed_dir / "dataset.json", out_dir / "metrics.json"
    needs = {
        "model": [("build", dataset_file)],
        "analyze": [("build", dataset_file), ("model", metrics_file)],
        "report": [("build", dataset_file), ("model", metrics_file)],
        "live": [("build", dataset_file)],
        "players": [("build", dataset_file)],
    }
    for name in names:
        for producer, path in needs.get(name, []):
            if producer not in names and not path.exists():
                return f"{name} needs {_shown(path)}: run {producer} first"
    return None


def _listed(errors: list[str]) -> list[str]:
    shown = errors[:MAX_ERRORS]
    return shown + [f"… and {len(errors) - MAX_ERRORS} more"] if len(errors) > MAX_ERRORS else shown


def stage_registry(store, llm_model: str = config.LLM_MODEL,
                   client_factory: Callable[[Callable], object] | None = None) -> dict[str, Stage]:
    """name -> stage for the runner. `store` needs only `reload(*parts)`; `client_factory(hook)` makes the UEFA
    client, so tests can pass a fake. The stages read and write the default config paths."""
    from .. import analyst, dataset, model

    def make_client(ctx: StageContext):
        def hook(kind: str, key: str, source: str) -> None:
            ctx.bump("requests", source)

        if client_factory is not None:
            return client_factory(hook)
        from ..uefa import UefaClient

        return UefaClient(on_request=hook)

    def fetch(ctx: StageContext) -> StageOutcome:
        result = stages.fetch(make_client(ctx), progress=lambda p: ctx.bump("seasons"),
                              log=lambda line: ctx.log(line.strip()))
        if result.failed:
            errors = [f"{match_id}: {reason}" for match_id, reason in result.failed.items()]
            return StageOutcome("failed", f"{len(errors)} match-stat requests failed; run fetch again to resume",
                                _listed(errors))
        return StageOutcome("done", "raw data cached")

    def build(ctx: StageContext) -> StageOutcome:
        try:
            ds = stages.build(make_client(ctx))
        except dataset.ValidationError as exc:
            ctx.set("validated", False)
            return StageOutcome("failed", "dataset validation failed", str(exc).splitlines())
        ctx.set("validated", True)
        store.reload("dataset")
        return StageOutcome("done", f"{len(ds.team_seasons)} team-seasons, {len(ds.features)} features")

    def fit(ctx: StageContext) -> StageOutcome:
        def progress(update: dict) -> None:
            if "fold" in update:
                ctx.set("folds", {"done": update["fold"], "of": update["of"]})
            else:
                ctx.set("ablation", {"done": update["k"], "of": update["of"]})

        results = stages.model(dataset.load(), progress=progress)
        store.reload("results")
        metrics = results.metrics
        return StageOutcome("done", f"Spearman {metrics['spearman_mean']:.2f}, AUC {metrics['auc']:.2f}")

    def analyze(ctx: StageContext) -> StageOutcome:
        ctx.log("Checking LM Studio; loading the model can take a few minutes")

        def log(line: str) -> None:
            ctx.log(line.strip())
            if NARRATIVE_LINE.match(line):
                ctx.bump("narratives")

        result = stages.analyze(dataset.load(), model.load(), llm_model, True, log=log, merge=True)
        store.reload("analysis")
        analysis = result.analysis
        if analysis.status != "ok":
            return StageOutcome("warning",
                                f"LM Studio not ready ({analysis.reason or 'unknown reason'}); earlier write-ups kept")
        if result.unavailable:
            n = len(result.unavailable)
            return StageOutcome("warning", f"{n} write-up{'s' if n != 1 else ''} could not be written this time "
                                           f"(earlier text kept where there was one): {', '.join(result.unavailable)}")
        return StageOutcome("done", f"{len(analysis.narratives)} write-ups")

    def report(ctx: StageContext) -> StageOutcome:
        path = stages.report(dataset.load(), model.load(), analyst.load())
        return StageOutcome("done", f"wrote {path.name} and report_page.html")

    return {"fetch": fetch, "build": build, "model": fit, "analyze": analyze, "report": report}


def live_stage(live) -> Stage:
    def run(ctx: StageContext) -> StageOutcome:
        ctx.log(f"Fetching the {config.season_label(config.LIVE_SEASON)} season from UEFA")
        state = live.refresh(force=True)
        if state.snapshot is None:
            return StageOutcome("failed", f"live season unavailable ({state.message or 'no data'})")
        message = f"{len(state.snapshot.rows)} teams, {state.snapshot.finished_matches} finished matches"
        if state.status == "stale":
            return StageOutcome("warning", f"{message}; UEFA couldn't be reached, so this is the last copy "
                                           f"({state.message or 'some match stats missing'})")
        return StageOutcome("done", message)
    return run


def players_stage(players, client_factory: Callable[[Callable], object] | None = None) -> Stage:
    def run(ctx: StageContext) -> StageOutcome:
        def hook(kind: str, key: str, source: str) -> None:
            ctx.bump("requests", source)

        if client_factory is not None:
            client = client_factory(hook)
        else:
            from ..uefa import UefaClient

            client = UefaClient(on_request=hook)  # so the run's request counters include the squads
        failures = players.build_index(lambda done, of: ctx.set("players", {"done": done, "of": of}), log=ctx.log,
                                       client=client)
        if failures:
            for (season, team_id), reason in failures[:MAX_ERRORS]:
                ctx.log(f"{config.season_label(season)} {team_id}: {reason}", level="warn")
            return StageOutcome("warning", f"{len(failures)} squads unavailable; their seasons are missing from "
                                           "player histories")
        return StageOutcome("done", "player index complete")
    return run


def all_pairs(store: DataStore, live) -> list[tuple[int, str]]:
    """Every (season, team_id) whose squad the player index should hold: history plus the live field."""
    pairs: list[tuple[int, str]] = []
    dataset = store.snapshot.dataset
    if dataset is not None:
        pairs += [(int(s), str(t)) for s, t in zip(dataset.team_seasons["season"], dataset.team_seasons["team_id"])]
    state = live.current() if live is not None else None
    if state is not None and state.snapshot is not None:
        rows = state.snapshot.rows
        pairs += [(int(s), str(t)) for s, t in zip(rows["season"], rows["team_id"])]
    return pairs


def build_services(processed_dir: Path = config.PROCESSED_DIR, out_dir: Path = config.OUT_DIR,
                   llm_model: str = config.LLM_MODEL, start_live: bool = True) -> Services:
    """Everything the app needs. The stages themselves always use the config paths, so tests that point this
    at tmp_path must not start real runs."""
    from ..uefa import UefaClient
    from .live import LiveService, build_snapshot
    from .players import PlayerService

    store = DataStore(processed_dir, out_dir)
    bus = EventBus()
    live_client = UefaClient()

    def build_live(force: bool):
        dataset = store.snapshot.dataset
        if dataset is None:
            raise RuntimeError("the historical dataset isn't built yet: run build first")
        return build_snapshot(live_client, dataset.team_seasons, dataset.features, force=force)

    live = LiveService(build_live)
    request_client = UefaClient(retry_delays=(), timeout=config.REQUEST_TIMEOUT_S)
    players = PlayerService(request_client, UefaClient(), lambda: all_pairs(store, live))
    registry = {**stage_registry(store, llm_model), "live": live_stage(live), "players": players_stage(players)}
    runner = PipelineRunner(registry, bus,
                            missing_inputs=lambda names: missing_inputs(names, processed_dir, out_dir))
    if start_live:
        live.refresh_in_background()
    return Services(store, bus, runner, live, players, llm_model)
