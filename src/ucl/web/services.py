"""The real pipeline, wired into the web runner. Each stage calls stages.py, turns its progress into counters,
and reloads its part of the DataStore (spec §4.3)."""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

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


def build_services(processed_dir: Path = config.PROCESSED_DIR, out_dir: Path = config.OUT_DIR,
                   llm_model: str = config.LLM_MODEL) -> Services:
    """The store, bus and runner over the given directories. The stages themselves always use the config
    paths, so tests that point this at tmp_path must not start real runs."""
    store = DataStore(processed_dir, out_dir)
    bus = EventBus()
    runner = PipelineRunner(stage_registry(store, llm_model), bus,
                            missing_inputs=lambda names: missing_inputs(names, processed_dir, out_dir))
    return Services(store, bus, runner)
