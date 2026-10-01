"""The pipeline's outputs on disk, held as one immutable snapshot that is swapped in whole after each stage
(spec §4.3). Requests read `store.snapshot` once, so they never see half of a reload."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .. import config
from .jsonsafe import to_jsonable

if TYPE_CHECKING:
    from ..analyst import Analysis
    from ..dataset import Dataset
    from ..model import ModelResults

PARTS = ("dataset", "results", "analysis")
NAMES = {"dataset": "processed data", "results": "model outputs", "analysis": "analysis"}
MISSING = {"dataset": "processed data missing: run build", "results": "model outputs missing: run model"}


@dataclass(frozen=True)
class Snapshot:
    dataset: Dataset | None = None
    results: ModelResults | None = None
    analysis: Analysis | None = None
    built_at: datetime | None = None
    modelled_at: datetime | None = None
    analysed_at: datetime | None = None
    errors: dict[str, str] = field(default_factory=dict)  # part -> why it could not be loaded
    stale_keys: frozenset[str] = frozenset()  # narratives written from facts that no longer match the data

    @property
    def ready(self) -> bool:
        return self.dataset is not None and self.results is not None


class DataStore:
    def __init__(self, processed_dir: Path = config.PROCESSED_DIR, out_dir: Path = config.OUT_DIR):
        self.processed_dir = Path(processed_dir)
        self.out_dir = Path(out_dir)
        self._lock = threading.Lock()
        self.snapshot = Snapshot()
        self.reload(*PARTS)

    def reload(self, *parts: str) -> Snapshot:
        """Re-read `parts` from disk and swap in a new snapshot. A part that fails to load becomes None and its
        reason goes into `errors`; parts not named are carried over unchanged."""
        from .. import analyst, dataset, model

        loaders = {
            "dataset": (lambda: dataset.load(self.processed_dir), self.processed_dir / "dataset.json", "built_at"),
            "results": (lambda: model.load(self.out_dir), self.out_dir / "metrics.json", "modelled_at"),
            "analysis": (lambda: analyst.load(self.out_dir / "analysis.json"), self.out_dir / "analysis.json",
                         "analysed_at"),
        }
        with self._lock:  # one reload at a time; readers never take the lock
            changes, errors = {}, dict(self.snapshot.errors)
            for part in parts:
                load, marker, stamp = loaders[part]
                try:
                    changes[part] = load()
                    changes[stamp] = _mtime(marker)
                    errors.pop(part, None)
                except Exception as exc:  # noqa: BLE001 - a missing or corrupt output means "not ready", not a crash
                    changes[part], changes[stamp] = None, None
                    errors[part] = MISSING[part] if isinstance(exc, FileNotFoundError) and part in MISSING else (
                        f"{NAMES[part]} could not be read: {type(exc).__name__}: {exc}")
            snapshot = replace(self.snapshot, **changes, errors=errors)
            self.snapshot = replace(snapshot, stale_keys=stale_narratives(snapshot))
            return self.snapshot


def _mtime(path: Path) -> datetime | None:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc) if path.exists() else None


def _normalised(value):
    return json.loads(json.dumps(to_jsonable(value), sort_keys=True, default=str))


def stale_narratives(snapshot: Snapshot) -> frozenset[str]:
    """Narrative keys whose stored facts differ from the facts built from today's data and model (spec §4.2).
    When the two can't be lined up (a dataset newer than the model) or the stored facts are unreadable, every
    narrative is stale."""
    analysis = snapshot.analysis
    if not snapshot.ready or analysis is None or not analysis.narratives:
        return frozenset()
    from ..facts import build_facts

    try:
        current = _normalised(build_facts(snapshot.dataset.team_seasons, snapshot.dataset.finals,
                                          snapshot.results))
        stored = _normalised(analysis.facts)
        return frozenset(key for key in analysis.narratives if stored.get(key) != current.get(key))
    except Exception:  # noqa: BLE001
        return frozenset(analysis.narratives)
