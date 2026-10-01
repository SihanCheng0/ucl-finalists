"""The pipeline's stages as plain functions, shared by the CLI and the web runner (spec §4.2).

Each stage returns its result and never prints; messages go through `log`. Module functions are called with the
positional signatures the CLI has always used, and new keyword arguments only when given, so tests that patch
those functions keep working. Modules are imported inside each stage, as in cli.py, so `ucl fetch` doesn't load
scikit-learn.

Two return types differ from the spec on purpose: `build` returns the Dataset itself (it carries the build notes)
and `analyze` returns AnalyzeResult, so the web runner can name the narratives it could not write."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from . import config

if TYPE_CHECKING:
    from .analyst import Analysis
    from .dataset import Dataset
    from .model import ModelResults
    from .uefa import UefaClient

Log = Callable[[str], None]
Progress = Callable[[dict], None]


def _given(**kwargs) -> dict:
    """Only the keyword arguments that were given."""
    return {k: v for k, v in kwargs.items() if v is not None}


@dataclass
class FetchResult:
    failed: dict[str, str]  # match id -> reason, for stats that failed after retries


@dataclass
class AnalyzeResult:
    analysis: Analysis
    unavailable: list[str] = field(default_factory=list)  # narrative keys this run could not write


def fetch(client: UefaClient, progress: Progress | None = None, log: Log = print) -> FetchResult:
    from . import dataset

    return FetchResult(dataset.fetch_all(client, config.SEASONS, log=log, **_given(progress=progress)))


def build(client: UefaClient) -> Dataset:
    """Build, validate and save the dataset. Raises dataset.ValidationError before anything is saved."""
    from . import dataset

    ds = dataset.build(client, config.SEASONS)
    dataset.validate(ds)
    dataset.save(ds)
    return ds


def model(ds: Dataset, progress: Progress | None = None) -> ModelResults:
    from . import model as modelling

    results = modelling.run(ds.team_seasons, ds.finals, ds.features, **_given(progress=progress))
    modelling.save(results)
    return results


def analyze(ds: Dataset, results: ModelResults, llm_model: str, enabled: bool, log: Log = print,
            merge: bool = False) -> AnalyzeResult:
    """Write the narratives and save analysis.json.

    Without `merge` (the CLI) every result is saved, as before. With `merge` (the web runner), a run the LLM
    could not do is never saved, and narratives this run could not write keep their previous text together
    with the facts it was written from, so the dashboard can flag them as written for earlier numbers."""
    from . import analyst

    analysis = analyst.run(ds.team_seasons, ds.finals, results, llm_model=llm_model, enabled=enabled, log=log)
    unavailable = [key for key, narrative in analysis.narratives.items() if narrative.status != "ok"]
    if merge:
        if analysis.status != "ok":
            return AnalyzeResult(analysis, unavailable)
        try:
            previous = analyst.load()
        except Exception:  # noqa: BLE001 - an unreadable old file means there is no earlier text to keep
            previous = analyst.Analysis("skipped", llm_model)
        for key in unavailable:
            old = previous.narratives.get(key)
            if old is not None and old.status == "ok":
                analysis.narratives[key] = old
                analysis.facts[key] = previous.facts.get(key, analysis.facts.get(key))
    analyst.save(analysis)
    return AnalyzeResult(analysis, unavailable)


def report(ds: Dataset, results: ModelResults, analysis: Analysis) -> Path:
    """Write out/report_page.html (the artifact-ready fragment) and out/report.html; returns the latter."""
    from . import report as reporting

    page = reporting.render(ds.team_seasons, ds.finals, results, analysis)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUT_DIR / "report_page.html").write_text(page)
    path = config.OUT_DIR / "report.html"
    path.write_text(reporting.standalone(page))
    return path
