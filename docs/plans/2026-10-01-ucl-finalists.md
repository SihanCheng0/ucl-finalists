# UCL Finalists Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a re-runnable pipeline that gathers 15 seasons of UEFA Champions League data, trains leave-one-season-out models of knockout depth, explains the 10 finalists of 2022–2026 with SHAP and a local LLM (Qwen 3.5 via LM Studio), and renders a self-contained HTML report.

**Architecture:** `uv run ucl all` chains five stages: fetch → build → model → analyze → report. Each stage reads the previous stage's files, so any stage can be re-run alone. Pure logic (labels, features, grounding checks, retry policy) is kept apart from I/O (the UEFA client and the LM Studio client), so every rule is unit-tested without network access.

**Tech Stack:** Python 3.13 (uv), pandas 3, scikit-learn 1.9, shap 0.52, scipy, pytest. HTTP uses stdlib `urllib`. The local LLM is LM Studio's OpenAI-compatible server.

**Spec:** `docs/specs/2026-10-01-ucl-finalists-design.md` (revision 4).
**Project root:** `~/Projects/ucl-finalists`. Every path below is relative to it, and every command runs from it.

**Refinements of spec §9.** The interfaces are unchanged; the files are smaller:
- `dataset.py` delegates to `labels.py` (rounds, stages, finals) and `features.py` (per-match rows → per-season features).
- `analyst.py` delegates to `grounding.py` (pure `check_grounding`, `validate_output`) and `llm.py` (LM Studio client and cache).
- `report.py` delegates chart markup to `charts.py` and its CSS/JS to `assets.py`. Charts are HTML/CSS rows rather than SVG, so text stays legible at phone width.
- `analyst.run` and `report.render` keep the spec signatures. They derive the active feature list from `results.drivers` via `model.feature_list`.
- `model.run(team_seasons, finals, features)` takes the finals table (for winners vs runners-up) and the build's active feature list, which can be shorter than 16 if the coverage rule drops one.

**Refinements adopted from plan review (also reflected in the spec):**
- The UEFA client returns `{id: reason}` for failed stats, checks response shapes before caching, names a corrupt cache file in its error, and `fetch` stops when every request in a season fails.
- Only 404/410 are cached as "missing" stats. Other 4xx responses fail fast and are not cached, because a 403 may be a block.
- If UEFA's stats feed uses a different id for a club than the match feed (seen for Steaua 2614166 vs FCSB 50065), and exactly one side is unmatched with exactly one stats entry left over, that entry is assigned to the unmatched side.
- P(final) from model B is rescaled within each season to sum to 2, since exactly two teams reach each final, with no value above 1. Model B trains mostly on 16-team seasons, so without this its raw probabilities run high in 24-team seasons.

**Verified facts this plan relies on (probed 2026-10-01):**
- **IDs:** match ids, team ids and coefficient member ids are strings, and per-match stat values are strings such as `"119.69"`.
- **Coefficients:** `overallRanking.totalValue` holds the coefficient. Ranking 2011 exists (439 clubs). Girona, Bologna and Stuttgart are absent from ranking 2024.
- **Standings:** `https://standings.uefa.com/v1/standings?competitionId=1&seasonYear={Y}` gives `points` and `goalDifference` per team. Example: Arsenal 2026 has 24 points and +19.
- **Libraries:** uv resolves pandas 3.0.6, scikit-learn 1.9.1, scipy 1.18.1 and shap 0.52.0. TreeSHAP on this exact `GradientBoostingRegressor` is additive to about 1e-15. LightGBM and XGBoost fail to import (no libomp).
- **LM Studio:** model key `qwen/qwen3.5-35b-a3b`. `reasoning_effort: "none"` → 0 reasoning tokens; without it the answers are often empty.

**Conventions:** keep comments sparse and only where the *why* is non-obvious. Prefer pure functions; every function that touches the network or a subprocess takes an injectable callable so tests can fake it.

---

## Chunk 1: Foundation

### Task 1: Project scaffold and config

**Files:**
- Create: `pyproject.toml`, `uv.lock` (by `uv sync`), `.python-version`, `.gitignore`, `src/ucl/__init__.py`, `src/ucl/config.py`, `tests/test_config.py`, `tasks/todo.md`

- [ ] **Step 1: Initialise the repo and write the packaging files**

```bash
git init -b main
mkdir -p src/ucl tests tasks scripts
```

`pyproject.toml`:

```toml
[project]
name = "ucl-finalists"
version = "0.1.0"
description = "Why the best Champions League teams win: UEFA data, validated models and a local-LLM analyst."
requires-python = ">=3.13,<3.14"
dependencies = [
    "numpy>=2.1",
    "pandas>=3.0",
    "scipy>=1.14",
    "scikit-learn>=1.9",
    "shap>=0.52",
]

[project.scripts]
ucl = "ucl.cli:main"

[dependency-groups]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/ucl"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`.python-version`:

```
3.13
```

`.gitignore`:

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.DS_Store
.claude/settings.local.json
data/raw/
data/llm_cache/
```

`src/ucl/__init__.py`:

```python
"""UCL finalists: why the best Champions League teams win."""
```

`tasks/todo.md` (the user's CLAUDE.md asks for a checkable task list in the project):

```markdown
# UCL Finalists — task list

Plan: docs/plans/2026-10-01-ucl-finalists.md
Spec: docs/specs/2026-10-01-ucl-finalists-design.md

- [ ] Chunk 1: scaffold, config, UEFA client
- [ ] Chunk 2: labels, features, dataset
- [ ] Chunk 3: fetch + build on real data, models, drivers, ablation, finals comparison
- [ ] Chunk 4: grounding checks, LM Studio client
- [ ] Chunk 5: fact sheets, analyst, analyze on the real data
- [ ] Chunk 6: charts and the report
- [ ] Chunk 7: end-to-end run, verification, publish

## Review
(filled in at the end)
```

- [ ] **Step 2: Write the failing config test**

`tests/test_config.py`:

```python
from ucl import config


def test_sixteen_features_have_metadata_and_sources():
    assert len(config.FEATURES) == 16
    assert set(config.FEATURE_META) == set(config.FEATURES)
    assert set(config.FEATURE_SOURCES) == set(config.FEATURES)
    assert set(config.FEATURE_GROUP) == set(config.FEATURES)


def test_season_helpers():
    assert config.season_label(2026) == "2025-26"
    assert config.season_label(2012) == "2011-12"
    assert (config.field_size(2024), config.field_size(2025)) == (32, 36)
    assert (config.phase_matches(2024), config.phase_matches(2025)) == (6, 8)
    assert (config.ko_size(2024), config.ko_size(2025)) == (16, 24)


def test_expected_finals_cover_the_target_seasons():
    assert sorted(config.EXPECTED_FINALS) == config.TARGET_SEASONS


def test_every_round_name_has_a_depth():
    assert config.ROUND_DEPTH["League Phase"] == config.ROUND_DEPTH["Group stage"] == 0
    assert config.ROUND_DEPTH["Knockout Phase Play-Offs"] == config.ROUND_DEPTH["Knock-out Play-off"] == 1
    assert config.ROUND_DEPTH["Final"] == 5
```

- [ ] **Step 3: Install and run the test to see it fail**

Run: `uv sync && uv run pytest tests/test_config.py -q`
Expected: FAIL with `ImportError: cannot import name 'config'`. uv downloads Python 3.13 if needed.

- [ ] **Step 4: Write `src/ucl/config.py`**

```python
"""Constants shared by every pipeline stage."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
LLM_CACHE_DIR = ROOT / "data" / "llm_cache"
OUT_DIR = ROOT / "out"

# UEFA's seasonYear is the calendar year a season ends: 2026 = 2025-26.
SEASONS = list(range(2012, 2027))
TARGET_SEASONS = [2022, 2023, 2024, 2025, 2026]


def season_label(season: int) -> str:
    return f"{season - 1}-{season % 100:02d}"


def is_league_format(season: int) -> bool:
    """2024-25 onward: one 36-team league phase instead of eight groups of four."""
    return season >= 2025


def field_size(season: int) -> int:
    return 36 if is_league_format(season) else 32


def phase_matches(season: int) -> int:
    return 8 if is_league_format(season) else 6


def ko_size(season: int) -> int:
    return 24 if is_league_format(season) else 16


# The five target finals, by UEFA team id (spec §3).
EXPECTED_FINALS = {
    2022: {"winner_id": "50051", "runner_up_id": "7889"},   # Real Madrid 1-0 Liverpool
    2023: {"winner_id": "52919", "runner_up_id": "50138"},  # Man City 1-0 Inter
    2024: {"winner_id": "50051", "runner_up_id": "52758"},  # Real Madrid 2-0 Dortmund
    2025: {"winner_id": "52747", "runner_up_id": "50138"},  # PSG 5-0 Inter
    2026: {"winner_id": "52747", "runner_up_id": "52280"},  # PSG 1-1 Arsenal, 4-3 on penalties
}

# Friendlier names than UEFA's internationalName ("Paris", "B. Dortmund").
DISPLAY_NAMES = {
    "50051": "Real Madrid",
    "7889": "Liverpool",
    "52919": "Manchester City",
    "50138": "Inter",
    "52758": "Borussia Dortmund",
    "52747": "Paris Saint-Germain",
    "52280": "Arsenal",
}

# Round names differ between seasons (spec §4); depth orders them.
ROUND_DEPTH = {
    "Group stage": 0,
    "League Phase": 0,
    "Knockout Phase Play-Offs": 1,
    "Knock-out Play-off": 1,
    "Round of 16": 2,
    "Quarter-finals": 3,
    "Semi-finals": 4,
    "Final": 5,
}
FEATURE_GROUPS = {
    "pedigree": ["coef_log"],
    "results": ["points_pg", "goal_diff_pg"],
    "style": [
        "shots_pg", "shot_accuracy", "conversion", "attacks_pg",
        "shots_against_pg", "on_target_against_pg", "save_pct",
        "possession_pct", "pass_accuracy", "passes_pg", "long_pass_share",
        "distance_km_pg", "fouls_pg",
    ],
}
FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]
FEATURE_GROUP = {f: group for group, fs in FEATURE_GROUPS.items() for f in fs}

# feature -> (plain-English label, display decimals, stored as a 0-1 fraction shown as %)
FEATURE_META = {
    "coef_log": ("Pre-season UEFA club coefficient (points)", 1, False),
    "points_pg": ("Points per game", 2, False),
    "goal_diff_pg": ("Goal difference per game", 2, False),
    "shots_pg": ("Shots per game", 1, False),
    "shot_accuracy": ("Shot accuracy (% on target)", 1, True),
    "conversion": ("Shot conversion (% of shots scored)", 1, True),
    "attacks_pg": ("Attacks per game", 1, False),
    "shots_against_pg": ("Opponent shots per game", 1, False),
    "on_target_against_pg": ("Opponent shots on target per game", 1, False),
    "save_pct": ("Save rate (% of shots on target faced)", 1, True),
    "possession_pct": ("Possession (%)", 1, False),
    "pass_accuracy": ("Pass accuracy (%)", 1, True),
    "passes_pg": ("Passes per game", 0, False),
    "long_pass_share": ("Long passes (% of passes)", 1, True),
    "distance_km_pg": ("Distance covered per game (km)", 1, False),
    "fouls_pg": ("Fouls committed per game", 1, False),
}
LOWER_IS_BETTER = {"shots_against_pg", "on_target_against_pg", "fouls_pg"}

# Per-match columns each feature needs, for the coverage rule (spec §5.3).
FEATURE_SOURCES = {
    "coef_log": [],
    "points_pg": [],
    "goal_diff_pg": [],
    "shots_pg": ["attempts_on_target", "attempts_off_target"],
    "shot_accuracy": ["attempts_on_target", "attempts_off_target"],
    "conversion": ["goals", "attempts_on_target", "attempts_off_target"],
    "attacks_pg": ["attacks"],
    "shots_against_pg": ["opp_attempts_on_target", "opp_attempts_off_target"],
    "on_target_against_pg": ["opp_attempts_on_target"],
    "save_pct": ["saves", "opp_attempts_on_target"],
    "possession_pct": ["ball_possession"],
    "pass_accuracy": ["passes_completed", "passes_attempted"],
    "passes_pg": ["passes_attempted"],
    "long_pass_share": ["passes_long_attempted", "passes_attempted"],
    "distance_km_pg": ["distance_covered"],
    "fouls_pg": ["fouls_committed"],
}
OWN_STATS = [
    "goals", "attempts_on_target", "attempts_off_target", "ball_possession", "passes_attempted",
    "passes_completed", "passes_long_attempted", "attacks", "distance_covered", "fouls_committed", "saves",
]
OPP_STATS = ["attempts_on_target", "attempts_off_target"]

COVERAGE_MIN = 0.90
COEF_MATCH_MIN = 0.75
MIN_MATCHES_WITH_STATS = 4

GBR_PARAMS = {
    "n_estimators": 250, "learning_rate": 0.03, "max_depth": 2,
    "min_samples_leaf": 8, "subsample": 0.8, "random_state": 42,
}
LOGIT_PARAMS = {"C": 0.3, "max_iter": 2000}
TOP_DRIVERS = 6
ROBUST_SHARE = 0.8  # same sign in >= 80% of LOSO fits = 12 of 15 (spec §6)

MATCHES_URL = (
    "https://match.uefa.com/v5/matches?competitionId=1&seasonYear={season}"
    "&phase=TOURNAMENT&limit=500&offset=0&order=ASC"
)
MATCH_STATS_URL = "https://matchstats.uefa.com/v2/team-statistics/{match_id}"
COEF_PAGE_SIZE = 500
COEF_URL = (
    "https://comp.uefa.com/v2/coefficients?coefficientRange=OVERALL&coefficientType=MEN_CLUB"
    f"&language=EN&page={{page}}&pagesize={COEF_PAGE_SIZE}&seasonYear={{season}}"
)
STANDINGS_URL = "https://standings.uefa.com/v1/standings?competitionId=1&seasonYear={season}"
HTTP_TIMEOUT_S = 25
HTTP_RETRY_DELAYS_S = (1, 2, 4, 8)

LLM_BASE_URL = "http://localhost:1234/v1"
LLM_MODEL = "qwen/qwen3.5-35b-a3b"
LLM_PARAMS = {"temperature": 0.3, "max_tokens": 8000, "reasoning_effort": "none"}
LLM_TIMEOUT_S = 300
LLM_CONTEXT_LENGTH = 16384
LMS_BIN = Path.home() / ".lmstudio" / "bin" / "lms"
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_config.py -q`
Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: scaffold ucl-finalists project with shared config"
```

### Task 2: UEFA client (cached, retrying, concurrent)

**Files:**
- Create: `src/ucl/uefa.py`
- Test: `tests/test_uefa.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_uefa.py`:

```python
import http.client
import json
import threading
import time
import urllib.error

import pytest

from ucl import config
from ucl.uefa import MISSING_MARKER, PermanentHTTPError, UefaClient


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://example.test", code, "error", hdrs=None, fp=None)


class FakeFetch:
    """Scripted responses per URL. Each call consumes one item; the last item repeats."""

    def __init__(self, responses: dict[str, list]):
        self.responses = {url: list(items) for url, items in responses.items()}
        self.calls: list[str] = []

    def __call__(self, url: str, timeout: float):
        self.calls.append(url)
        queue = self.responses[url]
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item


def make_client(tmp_path, responses):
    fetch = FakeFetch(responses)
    return UefaClient(cache_dir=tmp_path, fetch=fetch, sleep=lambda _: None), fetch


def test_matches_are_cached(tmp_path):
    url = config.MATCHES_URL.format(season=2026)
    client, fetch = make_client(tmp_path, {url: [[{"id": "1"}]]})
    assert client.matches(2026) == [{"id": "1"}]
    assert client.matches(2026) == [{"id": "1"}]
    assert fetch.calls == [url]


@pytest.mark.parametrize("error", [
    http_error(408), http_error(429), http_error(503), urllib.error.URLError("reset"), TimeoutError(),
    json.JSONDecodeError("bad", "", 0), http.client.IncompleteRead(b""),
])
def test_transient_errors_are_retried(tmp_path, error):
    url = config.MATCHES_URL.format(season=2026)
    client, fetch = make_client(tmp_path, {url: [error, [{"id": "1"}]]})
    assert client.matches(2026) == [{"id": "1"}]
    assert len(fetch.calls) == 2


def test_backoff_is_one_two_four_eight_seconds_plus_jitter(tmp_path):
    url = config.MATCHES_URL.format(season=2026)
    sleeps: list[float] = []
    client = UefaClient(cache_dir=tmp_path, fetch=FakeFetch({url: [http_error(500)]}), sleep=sleeps.append)
    with pytest.raises(RuntimeError, match="giving up"):
        client.matches(2026)
    assert len(sleeps) == 4
    assert all(d <= s <= d + 0.5 for s, d in zip(sleeps, (1, 2, 4, 8)))


def test_gives_up_after_five_attempts_and_caches_nothing(tmp_path):
    url = config.MATCHES_URL.format(season=2026)
    client, fetch = make_client(tmp_path, {url: [http_error(500)]})
    with pytest.raises(RuntimeError, match="giving up"):
        client.matches(2026)
    assert len(fetch.calls) == 5
    assert not list(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("code", [404, 410])
def test_not_found_on_stats_is_cached_as_missing(tmp_path, code):
    url = config.MATCH_STATS_URL.format(match_id="42")
    client, fetch = make_client(tmp_path, {url: [http_error(code)]})
    assert client.team_match_stats("42") is None
    assert client.team_match_stats("42") is None
    assert len(fetch.calls) == 1
    assert json.loads((tmp_path / "stats" / "42.json").read_text()) == MISSING_MARKER


def test_forbidden_on_stats_fails_fast_and_is_not_cached(tmp_path):
    # a 403 may be a block, not a missing resource: never cache it as "missing"
    url = config.MATCH_STATS_URL.format(match_id="43")
    client, fetch = make_client(tmp_path, {url: [http_error(403)]})
    with pytest.raises(PermanentHTTPError) as exc:
        client.team_match_stats("43")
    assert exc.value.code == 403
    assert len(fetch.calls) == 1
    assert not list(tmp_path.rglob("*.json"))


def test_permanent_4xx_on_matches_raises_after_one_call(tmp_path):
    url = config.MATCHES_URL.format(season=2026)
    client, fetch = make_client(tmp_path, {url: [http_error(404)]})
    with pytest.raises(PermanentHTTPError) as exc:
        client.matches(2026)
    assert exc.value.code == 404 and len(fetch.calls) == 1


def test_empty_stats_list_means_missing(tmp_path):
    url = config.MATCH_STATS_URL.format(match_id="7")
    client, _ = make_client(tmp_path, {url: [[]]})
    assert client.team_match_stats("7") is None


def test_unexpected_response_shapes_are_not_cached(tmp_path):
    url = config.MATCHES_URL.format(season=2026)
    client, _ = make_client(tmp_path, {url: [[]]})
    with pytest.raises(RuntimeError, match="unexpected response"):
        client.matches(2026)
    assert not list(tmp_path.rglob("*.json"))


def test_corrupt_cache_file_names_the_file(tmp_path):
    (tmp_path / "matches").mkdir()
    (tmp_path / "matches" / "2026.json").write_text("{truncated")
    client, fetch = make_client(tmp_path, {})
    with pytest.raises(RuntimeError, match="2026.json"):
        client.matches(2026)
    assert fetch.calls == []


def test_coefficients_follow_pages(tmp_path):
    def page(n):
        members = [{"member": {"id": str(i)}, "overallRanking": {"totalValue": 1.0}} for i in range(n)]
        return {"data": {"members": members}}

    p1 = config.COEF_URL.format(season=2025, page=1)
    p2 = config.COEF_URL.format(season=2025, page=2)
    client, fetch = make_client(tmp_path, {p1: [page(config.COEF_PAGE_SIZE)], p2: [page(3)]})
    assert len(client.coefficients(2025)) == config.COEF_PAGE_SIZE + 3
    assert fetch.calls == [p1, p2]


def test_many_reports_failures_with_reasons(tmp_path):
    ok = config.MATCH_STATS_URL.format(match_id="1")
    bad = config.MATCH_STATS_URL.format(match_id="2")
    entry = [{"teamId": "9", "statistics": []}]
    client, _ = make_client(tmp_path, {ok: [entry], bad: [http_error(503)]})
    results, failed = client.team_match_stats_many(["1", "2"])
    assert results == {"1": entry}
    assert list(failed) == ["2"] and "giving up" in failed["2"]


def test_many_fetches_each_unique_id_once_with_at_most_four_in_flight(tmp_path):
    lock, state, calls = threading.Lock(), {"now": 0, "peak": 0}, []

    def fetch(url, timeout):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
            calls.append(url)
        time.sleep(0.01)
        with lock:
            state["now"] -= 1
        return [{"teamId": "1", "statistics": []}]

    client = UefaClient(cache_dir=tmp_path, fetch=fetch, sleep=lambda _: None)
    results, failed = client.team_match_stats_many([str(i) for i in range(30)] * 2)
    assert len(results) == 30 and not failed
    assert len(calls) == 30 and state["peak"] <= 4
    assert not list(tmp_path.rglob("*.tmp"))
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_uefa.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ucl.uefa'`

- [ ] **Step 3: Write `src/ucl/uefa.py`**

```python
"""Cached, retrying client for UEFA's public JSON APIs (spec §4, §10)."""
from __future__ import annotations

import http.client
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable

from . import config

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
# Cached in place of a 404/410 so builds stay offline and deterministic.
MISSING_MARKER = {"__missing__": True}
MISSING_CODES = {404, 410}
RETRY_CODES = {408, 429}
MAX_COEF_PAGES = 20
# OSError covers URLError, TimeoutError, ConnectionError and ssl.SSLError.
TRANSIENT_ERRORS = (OSError, http.client.HTTPException, json.JSONDecodeError)

Fetch = Callable[[str, float], Any]
Check = Callable[[Any], bool]


class PermanentHTTPError(Exception):
    """A 4xx that retrying will not fix."""

    def __init__(self, message: str, code: int):
        super().__init__(message)
        self.code = code


def http_get_json(url: str, timeout: float) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


def _is_match_list(data: Any) -> bool:
    return isinstance(data, list) and len(data) > 0


def _is_coefficient_page(data: Any) -> bool:
    return isinstance(data, dict) and isinstance((data.get("data") or {}).get("members"), list)


class UefaClient:
    def __init__(
        self,
        cache_dir: Path = config.RAW_DIR,
        max_workers: int = 4,
        fetch: Fetch = http_get_json,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.cache_dir = Path(cache_dir)
        self.max_workers = max_workers
        self._fetch = fetch
        self._sleep = sleep

    def matches(self, season: int) -> list[dict]:
        return self._cached(f"matches/{season}", config.MATCHES_URL.format(season=season), check=_is_match_list)

    def team_match_stats(self, match_id: str) -> list[dict] | None:
        """The two teams' stats for a match, or None if UEFA has none."""
        data = self._cached(
            f"stats/{match_id}", config.MATCH_STATS_URL.format(match_id=match_id), missing_ok=True
        )
        return data or None

    def team_match_stats_many(
        self, match_ids: Iterable[str]
    ) -> tuple[dict[str, list[dict] | None], dict[str, str]]:
        """Fetch concurrently. Returns (results, {failed id: reason})."""

        def one(match_id: str):
            try:
                return match_id, self.team_match_stats(match_id), None
            except Exception as exc:  # noqa: BLE001 - reported to the caller with its reason
                return match_id, None, exc

        results: dict[str, list[dict] | None] = {}
        failed: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for match_id, data, error in pool.map(one, list(dict.fromkeys(match_ids))):
                if error is None:
                    results[match_id] = data
                else:
                    failed[match_id] = f"{type(error).__name__}: {error}"
        return results, failed

    def coefficients(self, season: int) -> list[dict]:
        """All members of the 5-year club ranking for `season`, across pages."""
        members: list[dict] = []
        for page in range(1, MAX_COEF_PAGES + 1):
            data = self._cached(
                f"coefficients/{season}_p{page}",
                config.COEF_URL.format(season=season, page=page),
                check=_is_coefficient_page,
            )
            batch = data["data"]["members"]
            members.extend(batch)
            if len(batch) < config.COEF_PAGE_SIZE:
                return members
        raise RuntimeError(f"coefficient ranking {season} did not end within {MAX_COEF_PAGES} pages")

    def _cached(self, key: str, url: str, missing_ok: bool = False, check: Check | None = None) -> Any:
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"corrupt cache file {path}: delete it and run `uv run ucl fetch` again") from exc
        else:
            try:
                data = self._get(url)
            except PermanentHTTPError as exc:
                # Only "not found" means missing; a 403 or 400 may be a block and must not poison the cache.
                if not (missing_ok and exc.code in MISSING_CODES):
                    raise
                data = MISSING_MARKER
            if check is not None and data != MISSING_MARKER and not check(data):
                raise RuntimeError(f"unexpected response shape from {url}; nothing was cached")
            _atomic_write_json(path, data)
        return None if data == MISSING_MARKER else data

    def _get(self, url: str) -> Any:
        delays = config.HTTP_RETRY_DELAYS_S
        last: Exception | None = None
        for attempt in range(len(delays) + 1):
            try:
                return self._fetch(url, config.HTTP_TIMEOUT_S)
            except urllib.error.HTTPError as exc:
                if 400 <= exc.code < 500 and exc.code not in RETRY_CODES:
                    raise PermanentHTTPError(f"HTTP {exc.code} for {url}", exc.code) from exc
                last = exc
            except TRANSIENT_ERRORS as exc:
                last = exc
            if attempt < len(delays):
                self._sleep(delays[attempt] + random.uniform(0, 0.5))
        raise RuntimeError(f"giving up on {url} after {len(delays) + 1} attempts: {last}") from last
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_uefa.py -q`
Expected: `20 passed`

- [ ] **Step 5: Tick Chunk 1 in `tasks/todo.md` and commit**

```bash
git add src/ucl/uefa.py tests/test_uefa.py tasks/todo.md
git commit -m "feat: cached, retrying UEFA API client"
```

---

## Chunk 2: Dataset

### Task 3: Labels — rounds, stages, finals

**Files:**
- Create: `src/ucl/labels.py`
- Test: `tests/test_labels.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_labels.py`:

```python
import math

import pandas as pd
import pytest

from ucl import labels


def raw_match(mid, round_name, home, away, hg, ag, winner=None, pens=None, reason="WIN_REGULAR", city="Budapest"):
    match = {
        "id": mid,
        "round": {"metaData": {"name": round_name}},
        "homeTeam": {"id": home, "internationalName": f"Team {home}"},
        "awayTeam": {"id": away, "internationalName": f"Team {away}"},
        "score": {"total": {"home": hg, "away": ag}},
        "stadium": {"city": {"translations": {"name": {"EN": city}}}},
    }
    if pens:
        match["score"]["penalty"] = {"home": pens[0], "away": pens[1]}
    if winner:
        match["winner"] = {"match": {"reason": reason, "team": {"id": winner}}}
    return match


def test_unknown_round_name_raises():
    with pytest.raises(ValueError, match="Play-in"):
        labels.match_rows([raw_match("1", "Play-in", "A", "B", 1, 0)], 2026)


def test_penalty_final_2026_style():
    raw = raw_match("9", "Final", "52747", "52280", 1, 1, winner="52747", pens=(4, 3), reason="WIN_ON_PENALTIES")
    final = labels.finals_table(labels.match_rows([raw], 2026)).iloc[0]
    assert (final.winner_id, final.runner_up_id) == ("52747", "52280")
    assert (final.winner_goals, final.runner_up_goals, final.winner_pens, final.runner_up_pens) == (1, 1, 4, 3)
    assert final.city == "Budapest"


def test_draw_reason_final_with_away_winner_2012_style():
    raw = raw_match("8", "Final", "50037", "52914", 1, 1, winner="52914", pens=(3, 4), reason="DRAW")
    final = labels.finals_table(labels.match_rows([raw], 2012)).iloc[0]
    assert (final.winner_id, final.runner_up_id) == ("52914", "50037")
    assert (final.winner_goals, final.runner_up_goals, final.winner_pens, final.runner_up_pens) == (1, 1, 4, 3)


def test_regular_final_is_oriented_to_the_winner():
    # 2022 is stored as Liverpool 0-1 Real Madrid
    raw = raw_match("7", "Final", "7889", "50051", 0, 1, winner="50051")
    final = labels.finals_table(labels.match_rows([raw], 2022)).iloc[0]
    assert (final.winner_id, final.winner_goals, final.runner_up_goals) == ("50051", 1, 0)
    assert pd.isna(final.winner_pens)


def test_stages_league_format():
    raw = [
        raw_match("1", "League Phase", "A", "G", 1, 0),
        raw_match("2", "League Phase", "B", "C", 1, 0),
        raw_match("3", "League Phase", "D", "E", 1, 0),
        raw_match("4", "League Phase", "F", "A", 1, 0),
        raw_match("5", "Knock-out Play-off", "A", "B", 0, 1, winner="B"),
        raw_match("6", "Round of 16", "B", "C", 0, 1, winner="C"),
        raw_match("7", "Quarter-finals", "C", "D", 0, 1, winner="D"),
        raw_match("8", "Semi-finals", "D", "E", 0, 1, winner="E"),
        raw_match("9", "Final", "E", "F", 0, 1, winner="F"),
    ]
    s = labels.stages(labels.match_rows(raw, 2026)).set_index("team_id")
    assert not s.loc["G", "in_ko"]
    assert math.isnan(s.loc["G", "ko_stage"])
    assert s.loc["G", "stage_label"] == "League phase"
    assert (s.loc["A", "ko_stage"], s.loc["A", "stage_label"]) == (0, "Knockout play-off")
    assert (s.loc["B", "ko_stage"], s.loc["B", "stage_label"]) == (0, "Round of 16")
    assert (s.loc["C", "ko_stage"], s.loc["D", "ko_stage"]) == (1, 2)
    assert (s.loc["E", "ko_stage"], s.loc["E", "stage_label"]) == (3, "Runner-up")
    assert (s.loc["F", "ko_stage"], s.loc["F", "stage_label"]) == (4, "Winner")


def test_group_stage_label_before_2025():
    s = labels.stages(labels.match_rows([raw_match("1", "Group stage", "A", "B", 0, 0)], 2024))
    assert set(s["stage_label"]) == {"Group stage"}


def test_stages_group_format_ko_levels():
    raw = [
        raw_match("1", "Group stage", "A", "B", 1, 0),
        raw_match("2", "Group stage", "C", "D", 1, 0),
        raw_match("3", "Group stage", "E", "F", 1, 0),
        raw_match("4", "Group stage", "G", "A", 1, 0),
        raw_match("5", "Round of 16", "A", "B", 0, 1, winner="B"),
        raw_match("6", "Quarter-finals", "B", "C", 0, 1, winner="C"),
        raw_match("7", "Semi-finals", "C", "D", 0, 1, winner="D"),
        raw_match("8", "Final", "D", "E", 2, 1, winner="D"),
    ]
    s = labels.stages(labels.match_rows(raw, 2024)).set_index("team_id")["ko_stage"]
    assert s[["A", "B", "C", "D", "E"]].tolist() == [0, 1, 2, 4, 3]
    assert math.isnan(s["F"]) and math.isnan(s["G"])
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_labels.py -q`
Expected: FAIL with `ImportError: cannot import name 'labels'`

- [ ] **Step 3: Write `src/ucl/labels.py`**

```python
"""Round names → depth, how far each team went, and the finals table (spec §5.2, §5.4)."""
from __future__ import annotations

import pandas as pd

from . import config


def _city(match: dict) -> str | None:
    stadium = match.get("stadium") or {}
    return (((stadium.get("city") or {}).get("translations") or {}).get("name") or {}).get("EN")


def match_rows(raw_matches: list[dict], season: int) -> pd.DataFrame:
    """One row per match, keeping only the fields the pipeline uses."""
    rows = []
    for m in raw_matches:
        round_name = m["round"]["metaData"]["name"]
        if round_name not in config.ROUND_DEPTH:
            raise ValueError(f"unknown round name {round_name!r} in season {season}")
        score = m.get("score") or {}
        total = score.get("total") or {}
        pens = score.get("penalty") or {}
        winner = ((m.get("winner") or {}).get("match") or {}).get("team") or {}
        rows.append({
            "season": season,
            "match_id": str(m["id"]),
            "round": round_name,
            "depth": config.ROUND_DEPTH[round_name],
            "home_id": str(m["homeTeam"]["id"]),
            "home": m["homeTeam"]["internationalName"],
            "away_id": str(m["awayTeam"]["id"]),
            "away": m["awayTeam"]["internationalName"],
            "home_goals": total.get("home"),
            "away_goals": total.get("away"),
            "home_pens": pens.get("home"),
            "away_pens": pens.get("away"),
            "winner_id": str(winner["id"]) if winner.get("id") is not None else None,
            "city": _city(m),
        })
    return pd.DataFrame(rows)


def ko_stage(depth: int, won_final: bool) -> int | None:
    """0 out before QF, 1 QF, 2 SF, 3 runner-up, 4 winner; None outside the knockouts."""
    if depth == 0:
        return None
    if depth in (1, 2):
        return 0
    if depth == 5:
        return 4 if won_final else 3
    return depth - 2


def stage_label(depth: int, won_final: bool, season: int) -> str:
    if depth == 0:
        return "League phase" if config.is_league_format(season) else "Group stage"
    if depth == 5:
        return "Winner" if won_final else "Runner-up"
    return {1: "Knockout play-off", 2: "Round of 16", 3: "Quarter-finals", 4: "Semi-finals"}[depth]


def stages(matches: pd.DataFrame) -> pd.DataFrame:
    """Furthest round per (season, team): in_ko, ko_stage (NaN outside the knockouts), stage_label."""
    sides = [
        matches[["season", "depth", f"{side}_id", side]].rename(columns={f"{side}_id": "team_id", side: "team"})
        for side in ("home", "away")
    ]
    furthest = (
        pd.concat(sides, ignore_index=True)
        .sort_values("depth", kind="stable")
        .groupby(["season", "team_id"], as_index=False)
        .last()
    )
    final_winner = matches.loc[matches["round"] == "Final"].set_index("season")["winner_id"].to_dict()
    rows = []
    for r in furthest.itertuples(index=False):
        depth = int(r.depth)
        won = depth == 5 and final_winner.get(r.season) == r.team_id
        rows.append({
            "season": int(r.season),
            "team_id": r.team_id,
            "team": r.team,
            "in_ko": depth >= 1,
            "ko_stage": ko_stage(depth, won),
            "stage_label": stage_label(depth, won, int(r.season)),
        })
    out = pd.DataFrame(rows)
    out["ko_stage"] = out["ko_stage"].astype("float")
    return out


def finals_table(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per season with scores oriented to the winner (home/away in a final is nominal)."""
    rows = []
    for m in matches.loc[matches["round"] == "Final"].itertuples(index=False):
        if m.winner_id not in (m.home_id, m.away_id):
            raise ValueError(f"season {m.season}: final winner {m.winner_id!r} is not one of the finalists")
        w, r = ("home", "away") if m.winner_id == m.home_id else ("away", "home")
        rows.append({
            "season": int(m.season),
            "winner_id": getattr(m, f"{w}_id"),
            "winner": getattr(m, w),
            "runner_up_id": getattr(m, f"{r}_id"),
            "runner_up": getattr(m, r),
            "winner_goals": getattr(m, f"{w}_goals"),
            "runner_up_goals": getattr(m, f"{r}_goals"),
            "winner_pens": getattr(m, f"{w}_pens"),
            "runner_up_pens": getattr(m, f"{r}_pens"),
            "city": m.city,
        })
    return pd.DataFrame(rows).sort_values("season").reset_index(drop=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_labels.py -q`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add src/ucl/labels.py tests/test_labels.py
git commit -m "feat: round→stage labels and winner-oriented finals table"
```

### Task 4: Features — per-match rows to per-season features

**Files:**
- Create: `src/ucl/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_features.py`:

```python
import numpy as np
import pandas as pd
import pytest

from ucl import features


def stats(team_id, **values):
    return {"teamId": team_id, "statistics": [{"name": k, "value": str(v)} for k, v in values.items()]}


BASE = dict(goals=0, attempts_on_target=0, attempts_off_target=0, ball_possession=50, passes_attempted=400,
            passes_completed=320, passes_long_attempted=40, attacks=40, distance_covered=110,
            fouls_committed=10, saves=0)


@pytest.fixture
def two_matches():
    matches = pd.DataFrame([
        {"season": 2026, "match_id": "m1", "round": "League Phase", "depth": 0, "home_id": "1", "home": "One",
         "away_id": "2", "away": "Two", "home_goals": 2, "away_goals": 1},
        {"season": 2026, "match_id": "m2", "round": "League Phase", "depth": 0, "home_id": "3", "home": "Three",
         "away_id": "1", "away": "One", "home_goals": 0, "away_goals": 0},
        {"season": 2026, "match_id": "k1", "round": "Round of 16", "depth": 2, "home_id": "1", "home": "One",
         "away_id": "3", "away": "Three", "home_goals": 5, "away_goals": 0},
    ])
    stat_map = {
        "m1": [
            stats("1", **{**BASE, "goals": 2, "attempts_on_target": 5, "attempts_off_target": 5,
                          "ball_possession": 60, "passes_attempted": 500, "passes_completed": 450,
                          "passes_long_attempted": 50, "saves": 2}),
            stats("2", **{**BASE, "goals": 1, "attempts_on_target": 3, "attempts_off_target": 4}),
        ],
        # team 1's own stats are missing for m2; the opponent's are present
        "m2": [stats("3", **{**BASE, "attempts_on_target": 2, "attempts_off_target": 2})],
    }
    return matches, stat_map


def test_team_match_rows_use_only_phase_matches(two_matches):
    tm = features.team_match_rows(*two_matches)
    assert set(tm["match_id"]) == {"m1", "m2"}
    assert len(tm) == 4


def test_aliased_stats_team_id_is_matched_to_the_unmatched_side():
    # real case: the match feed has Steaua as 2614166, the stats feed as 50065 (FCSB)
    matches = pd.DataFrame([{"season": 2014, "match_id": "a1", "round": "Group stage", "depth": 0,
                             "home_id": "1", "home": "One", "away_id": "2614166", "away": "Steaua",
                             "home_goals": 1, "away_goals": 1}])
    stat_map = {"a1": [stats("1", **{**BASE, "attempts_on_target": 4}),
                       stats("50065", **{**BASE, "attempts_on_target": 6})]}
    tm = features.team_match_rows(matches, stat_map).set_index("team_id")
    assert tm.loc["2614166", "has_stats"]
    assert tm.loc["2614166", "attempts_on_target"] == 6
    assert tm.loc["1", "opp_attempts_on_target"] == 6


def test_season_features_follow_missing_stat_rules(two_matches):
    tm = features.team_match_rows(*two_matches)
    row = features.season_features(tm).set_index("team_id").loc["1"]
    assert (row.n_matches, row.n_with_stats) == (2, 1)
    assert row.points_pg == pytest.approx(2.0)          # win + draw, from the score
    assert row.goal_diff_pg == pytest.approx(0.5)
    assert row.shots_pg == pytest.approx(10.0)          # only m1 has own stats
    assert row.shot_accuracy == pytest.approx(0.5)
    assert row.conversion == pytest.approx(0.2)
    assert row.shots_against_pg == pytest.approx(5.5)   # opponents: 7 in m1, 4 in m2
    assert row.on_target_against_pg == pytest.approx(2.5)
    assert row.save_pct == pytest.approx(2 / 3)         # m2 excluded: own saves missing
    assert row.pass_accuracy == pytest.approx(0.9)
    assert row.long_pass_share == pytest.approx(0.1)


def test_covered_features_drop_sparse_stats():
    tm = pd.DataFrame({
        "season": [2025] * 4 + [2026] * 4,
        "attacks": [1, 2, 3, 4, 1, np.nan, np.nan, 4],
        "fouls_committed": [1] * 8,
    })
    kept, worst = features.covered_features(tm, ["attacks_pg", "fouls_pg", "points_pg"])
    assert kept == ["fouls_pg", "points_pg"]
    assert worst["attacks_pg"] == pytest.approx(0.5)


def test_coefficients_shift_year_and_fill_debutants():
    coefs = features.coefficient_table({2025: [
        {"member": {"id": "1"}, "overallRanking": {"totalValue": 100.0}},
        {"member": {"id": "2"}, "overallRanking": {"totalValue": 50.0}},
    ]})
    ts = pd.DataFrame({"season": [2026, 2026, 2026], "team_id": ["1", "2", "3"]})
    out, rate = features.attach_coefficients(ts, coefs)
    out = out.set_index("team_id")
    assert out.loc["3", "coef"] == 50.0 and out.loc["3", "coef_imputed"]
    assert not out.loc["1", "coef_imputed"]
    assert out.loc["1", "coef_log"] == pytest.approx(np.log1p(100.0))
    assert rate == {2026: pytest.approx(2 / 3)}


def test_impute_uses_the_season_median():
    df = pd.DataFrame({"season": [1, 1, 1, 2], "save_pct": [0.5, np.nan, 0.7, 0.9]})
    out, n = features.impute_season_median(df, ["save_pct"])
    assert n == 1
    assert out.loc[1, "save_pct"] == pytest.approx(0.6)


def test_zscores_and_percentiles_are_within_season():
    df = pd.DataFrame({"season": [1, 1, 1, 2, 2, 2], "shots_pg": [10.0, 12.0, 14.0, 1.0, 2.0, 9.0]})
    z = features.zscore_within_season(df, ["shots_pg"])
    assert z.groupby(df["season"])["z_shots_pg"].mean().abs().max() < 1e-12
    pct = features.add_percentiles(df, ["shots_pg"])
    assert pct.loc[2, "pct_season_shots_pg"] == 100 and pct.loc[5, "pct_season_shots_pg"] == 100
    assert pct.loc[5, "pct_all_shots_pg"] == 50  # 9.0 is the 3rd smallest of 6


def test_display_value_formats():
    row = pd.Series({"coef": 148.0, "coef_log": 5.0, "save_pct": 0.7123, "passes_pg": 523.4})
    assert features.display_value(row, "coef_log") == 148.0
    assert features.display_value(row, "save_pct") == 71.2
    assert features.display_value(row, "passes_pg") == 523
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_features.py -q`
Expected: FAIL with `ImportError: cannot import name 'features'`

- [ ] **Step 3: Write `src/ucl/features.py`**

```python
"""Per-match rows → per-season features, normalisation and percentiles (spec §5.3)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _stat_values(entry: dict) -> dict[str, float]:
    values: dict[str, float] = {}
    for stat in entry.get("statistics", []):
        try:
            values[stat["name"]] = float(stat["value"])
        except (KeyError, TypeError, ValueError):
            continue
    return values


def team_match_rows(matches: pd.DataFrame, stats: dict[str, list[dict] | None]) -> pd.DataFrame:
    """One row per team per group/league-phase match: result, own stats, opponent shot stats."""
    rows = []
    for m in matches.loc[matches["depth"] == 0].itertuples(index=False):
        by_team = {str(e["teamId"]): _stat_values(e) for e in (stats.get(m.match_id) or [])}
        # The stats feed occasionally uses another id for a club than the match feed (Steaua 2614166
        # vs FCSB 50065). With one side unmatched and one entry left over, that entry is the unmatched side's.
        unmatched = [t for t in (m.home_id, m.away_id) if t not in by_team]
        leftover = [t for t in by_team if t not in (m.home_id, m.away_id)]
        if len(unmatched) == 1 and len(leftover) == 1:
            by_team[unmatched[0]] = by_team.pop(leftover[0])
        for side, other in (("home", "away"), ("away", "home")):
            team_id, opp_id = getattr(m, f"{side}_id"), getattr(m, f"{other}_id")
            gf, ga = int(getattr(m, f"{side}_goals")), int(getattr(m, f"{other}_goals"))
            own, opp = by_team.get(team_id), by_team.get(opp_id)
            row = {
                "season": int(m.season), "match_id": m.match_id, "team_id": team_id, "opp_id": opp_id,
                "gf": gf, "ga": ga, "points": 3 if gf > ga else 1 if gf == ga else 0,
                "has_stats": bool(own),
            }
            for name in config.OWN_STATS:
                row[name] = own.get(name, np.nan) if own else np.nan
            for name in config.OPP_STATS:
                row[f"opp_{name}"] = opp.get(name, np.nan) if opp else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _ratio(numerator: pd.Series, denominator: pd.Series) -> float:
    """Ratio of sums over the matches where both are present."""
    both = numerator.notna() & denominator.notna()
    total = denominator[both].sum()
    return float(numerator[both].sum() / total) if total > 0 else np.nan


def _season_row(g: pd.DataFrame) -> dict[str, float]:
    shots = g["attempts_on_target"] + g["attempts_off_target"]
    opp_shots = g["opp_attempts_on_target"] + g["opp_attempts_off_target"]
    return {
        "n_matches": len(g),
        "n_with_stats": int(g["has_stats"].sum()),
        "points_pg": g["points"].mean(),
        "goal_diff_pg": (g["gf"] - g["ga"]).mean(),
        "shots_pg": shots.mean(),
        "shot_accuracy": _ratio(g["attempts_on_target"], shots),
        "conversion": _ratio(g["goals"], shots),
        "attacks_pg": g["attacks"].mean(),
        "shots_against_pg": opp_shots.mean(),
        "on_target_against_pg": g["opp_attempts_on_target"].mean(),
        "save_pct": _ratio(g["saves"], g["opp_attempts_on_target"]),
        "possession_pct": g["ball_possession"].mean(),
        "pass_accuracy": _ratio(g["passes_completed"], g["passes_attempted"]),
        "passes_pg": g["passes_attempted"].mean(),
        "long_pass_share": _ratio(g["passes_long_attempted"], g["passes_attempted"]),
        "distance_km_pg": g["distance_covered"].mean(),
        "fouls_pg": g["fouls_committed"].mean(),
    }


def season_features(team_matches: pd.DataFrame) -> pd.DataFrame:
    """Per-match averages skip matches where the stat is missing (pandas mean skips NaN)."""
    rows = [
        {"season": int(season), "team_id": team_id, **_season_row(g)}
        for (season, team_id), g in team_matches.groupby(["season", "team_id"], sort=True)
    ]
    return pd.DataFrame(rows)


def covered_features(team_matches: pd.DataFrame, features: list[str]) -> tuple[list[str], dict[str, float]]:
    """Keep features whose source stats are present in >= COVERAGE_MIN of team-matches in every season."""
    kept: list[str] = []
    worst: dict[str, float] = {}
    for f in features:
        sources = config.FEATURE_SOURCES[f]
        if not sources:
            kept.append(f)
            continue
        present = team_matches[sources].notna().all(axis=1)
        worst[f] = float(present.groupby(team_matches["season"]).mean().min())
        if worst[f] >= config.COVERAGE_MIN:
            kept.append(f)
    return kept, worst


def coefficient_table(rankings: dict[int, list[dict]]) -> pd.DataFrame:
    """Ranking year Y applies to season Y + 1 (pre-season strength, spec §4)."""
    rows = [
        {"season": year + 1, "team_id": str(m["member"]["id"]), "coef": float(m["overallRanking"]["totalValue"])}
        for year, members in rankings.items()
        for m in members
    ]
    return pd.DataFrame(rows, columns=["season", "team_id", "coef"]).drop_duplicates(["season", "team_id"])


def attach_coefficients(team_seasons: pd.DataFrame, coefs: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, float]]:
    """Join by team id. Debutants absent from ranking Y-1 get the season minimum."""
    out = team_seasons.merge(coefs, on=["season", "team_id"], how="left")
    match_rate = {int(s): float(r) for s, r in out["coef"].notna().groupby(out["season"]).mean().items()}
    out["coef_imputed"] = out["coef"].isna()
    out["coef"] = out["coef"].fillna(out.groupby("season")["coef"].transform("min"))
    out["coef_log"] = np.log1p(out["coef"])
    return out, match_rate


def impute_season_median(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    n_missing = int(out[features].isna().sum().sum())
    out[features] = out[features].fillna(out.groupby("season")[features].transform("median"))
    return out, n_missing


def zscore_within_season(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Population z-scores against each season's field. Uses feature values only, never labels."""
    deviation = df[features] - df.groupby("season")[features].transform("mean")
    std = (deviation**2).groupby(df["season"]).transform("mean") ** 0.5
    return (deviation / std).add_prefix("z_")


def add_percentiles(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df.copy()
    for f in features:
        out[f"pct_season_{f}"] = (out.groupby("season")[f].rank(pct=True) * 100).round()
        out[f"pct_all_{f}"] = (out[f].rank(pct=True) * 100).round()
    return out


def display_value(row: pd.Series, feature: str) -> float | int:
    """Human-facing value: the raw coefficient for coef_log, percentages for 0-1 fractions."""
    _, decimals, is_fraction = config.FEATURE_META[feature]
    value = float(row["coef"] if feature == "coef_log" else row[feature])
    if is_fraction:
        value *= 100
    return int(round(value)) if decimals == 0 else round(value, decimals)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_features.py -q`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add src/ucl/features.py tests/test_features.py
git commit -m "feat: per-season features with missing-stat, coefficient and normalisation rules"
```

### Task 5: Dataset — build, validate, save, load

**Files:**
- Create: `src/ucl/dataset.py`, `tests/synthetic.py`, `tests/conftest.py`
- Test: `tests/test_dataset.py`

- [ ] **Step 1: Write the synthetic-dataset helper and the failing tests**

`tests/synthetic.py` (a dataset with the real structure, reused by the model and analyst tests):

```python
"""A synthetic dataset with real field sizes, knockout counts and the expected finals."""
import numpy as np
import pandas as pd

from ucl import config, features
from ucl.dataset import Dataset


def make_synthetic_dataset(seasons=tuple(config.SEASONS), seed: int = 7) -> Dataset:
    rng = np.random.default_rng(seed)
    team_rows, final_matches, finals = [], [], []
    for season in seasons:
        ids = [f"{season}{i:02d}" for i in range(config.field_size(season))]
        expected = config.EXPECTED_FINALS.get(season)
        if expected:
            ids[0], ids[1] = expected["winner_id"], expected["runner_up_id"]
        n_ko = config.ko_size(season)
        stages = [4, 3, 2, 2, 1, 1, 1, 1] + [0] * (n_ko - 8) + [None] * (len(ids) - n_ko)
        for team_id, stage in zip(ids, stages):
            strength = rng.normal() + (0.6 * stage if stage is not None else -0.8)
            row = {
                "season": season, "team_id": team_id, "team": f"Team {team_id}",
                "in_ko": stage is not None, "ko_stage": np.nan if stage is None else float(stage),
                "stage_label": "synthetic", "n_matches": config.phase_matches(season),
                "n_with_stats": config.phase_matches(season), "coef_imputed": False,
            }
            for f in config.FEATURES:
                row[f] = strength * rng.uniform(0.3, 1.0) + rng.normal(scale=0.7)
            row["coef"] = round(abs(row["coef_log"]) * 30, 1)
            team_rows.append(row)
        final_matches.append({"season": season, "match_id": f"F{season}", "round": "Final", "depth": 5,
                              "home_id": ids[1], "away_id": ids[0], "winner_id": ids[0]})
        finals.append({"season": season, "winner_id": ids[0], "winner": f"Team {ids[0]}",
                       "runner_up_id": ids[1], "runner_up": f"Team {ids[1]}", "winner_goals": 2,
                       "runner_up_goals": 1, "winner_pens": np.nan, "runner_up_pens": np.nan,
                       "city": "Testville"})
    ts = pd.DataFrame(team_rows)
    ts = pd.concat([ts, features.zscore_within_season(ts, config.FEATURES)], axis=1)
    ts = features.add_percentiles(ts, config.FEATURES)
    ts["complete"] = True
    ts["reached_final"] = ts["ko_stage"].ge(3)
    ts["is_target"] = ts["season"].isin(config.TARGET_SEASONS) & ts["reached_final"]
    ts["team_display"] = ts["team"]
    return Dataset(
        matches=pd.DataFrame(final_matches),
        team_match_stats=pd.DataFrame({"season": [], "match_id": [], "team_id": [], "has_stats": []}),
        team_seasons=ts,
        finals=pd.DataFrame(finals),
        features=list(config.FEATURES),
        notes={"coef_match_rate": {s: 1.0 for s in seasons}},
    )
```

`tests/conftest.py`:

```python
import pytest
from synthetic import make_synthetic_dataset


@pytest.fixture
def synthetic_ds():
    return make_synthetic_dataset()


@pytest.fixture(scope="session")
def built():
    """Six synthetic seasons with model results, shared by the facts, analyst and report tests."""
    from ucl import model  # imported lazily: model.py is written after this file

    ds = make_synthetic_dataset(seasons=tuple(range(2021, 2027)))
    return ds, model.run(ds.team_seasons, ds.finals, ds.features)
```

`tests/test_dataset.py`:

```python
import pandas as pd
import pytest
from synthetic import make_synthetic_dataset

from ucl import dataset


def test_valid_dataset_passes(synthetic_ds):
    dataset.validate(synthetic_ds)


def test_missing_team_is_reported(synthetic_ds):
    ts = synthetic_ds.team_seasons
    synthetic_ds.team_seasons = ts.drop(ts.index[ts["season"] == 2013][-1])
    with pytest.raises(dataset.ValidationError, match="2013: field has 31 teams"):
        dataset.validate(synthetic_ds)


def test_wrong_final_is_reported(synthetic_ds):
    synthetic_ds.finals.loc[synthetic_ds.finals["season"] == 2026, "winner_id"] = "999"
    with pytest.raises(dataset.ValidationError, match="2026: final"):
        dataset.validate(synthetic_ds)


def test_incomplete_target_finalist_is_reported(synthetic_ds):
    ts = synthetic_ds.team_seasons
    ts.loc[ts["is_target"] & (ts["season"] == 2025), "complete"] = False
    with pytest.raises(dataset.ValidationError, match="target finalists without enough stats"):
        dataset.validate(synthetic_ds)


def test_bad_knockout_levels_are_reported(synthetic_ds):
    ts = synthetic_ds.team_seasons
    ts.loc[ts.index[(ts["season"] == 2020) & (ts["ko_stage"] == 1)][0], "ko_stage"] = 0.0
    with pytest.raises(dataset.ValidationError, match="2020: teams at ko_stage"):
        dataset.validate(synthetic_ds)


def test_low_coefficient_match_rate_is_reported(synthetic_ds):
    synthetic_ds.notes["coef_match_rate"][2019] = 0.5
    with pytest.raises(dataset.ValidationError, match="2019: only 50% of clubs matched"):
        dataset.validate(synthetic_ds)


def test_save_and_load_roundtrip_keeps_id_strings(tmp_path):
    ds = make_synthetic_dataset(seasons=(2026,))
    ds.team_match_stats = pd.DataFrame(
        {"season": [2026], "match_id": ["0042"], "team_id": ["007"], "has_stats": [False]}
    )
    dataset.save(ds, tmp_path)
    loaded = dataset.load(tmp_path)
    assert loaded.team_match_stats.loc[0, "team_id"] == "007"
    assert loaded.team_match_stats.loc[0, "match_id"] == "0042"
    assert loaded.features == ds.features
    assert loaded.team_seasons["in_ko"].dtype == bool
    assert (tmp_path / "missing_stats.csv").exists()
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_dataset.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ucl.dataset'`

- [ ] **Step 3: Write `src/ucl/dataset.py`**

```python
"""Build, validate, save and load the analysis dataset (spec §5)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from . import config
from . import features as feat
from . import labels
from .uefa import UefaClient

ID_COLUMNS = ["team_id", "opp_id", "match_id", "home_id", "away_id", "winner_id", "runner_up_id"]


class ValidationError(Exception):
    """The built dataset breaks one of the spec §5.5 rules."""


@dataclass
class Dataset:
    matches: pd.DataFrame
    team_match_stats: pd.DataFrame
    team_seasons: pd.DataFrame
    finals: pd.DataFrame
    features: list[str]
    notes: dict = field(default_factory=dict)


def fetch_all(client: UefaClient, seasons: list[int] = config.SEASONS,
              log: Callable[[str], None] = print) -> dict[str, str]:
    """Fill the raw cache. Returns {match id: reason} for stats that failed after retries."""
    failed_all: dict[str, str] = {}
    for season in seasons:
        matches = labels.match_rows(client.matches(season), season)
        client.coefficients(season - 1)
        phase_ids = matches.loc[matches["depth"] == 0, "match_id"].tolist()
        stats, failed = client.team_match_stats_many(phase_ids)
        missing = sum(1 for v in stats.values() if v is None)
        log(f"{season}: {len(matches)} matches, {len(phase_ids)} phase matches, "
            f"stats missing {missing}, failed {len(failed)}")
        if failed:
            match_id, reason = next(iter(failed.items()))
            log(f"  e.g. match {match_id}: {reason}")
        failed_all.update(failed)
        if phase_ids and len(failed) == len(phase_ids):
            log("  every request failed for this season; stopping. Fix the cause above, then run `uv run ucl fetch` again.")
            break
    return failed_all


def build(client: UefaClient, seasons: list[int] = config.SEASONS) -> Dataset:
    matches = pd.concat([labels.match_rows(client.matches(s), s) for s in seasons], ignore_index=True)
    stats, failed = client.team_match_stats_many(matches.loc[matches["depth"] == 0, "match_id"].tolist())
    if failed:
        match_id, reason = next(iter(failed.items()))
        raise RuntimeError(f"{len(failed)} match-stat requests failed (e.g. {match_id}: {reason}); "
                           "run `uv run ucl fetch` again first")
    tm = feat.team_match_rows(matches, stats)
    kept, coverage = feat.covered_features(tm, config.FEATURES)

    ts = labels.stages(matches).merge(feat.season_features(tm), on=["season", "team_id"], how="left")
    coefs = feat.coefficient_table({s - 1: client.coefficients(s - 1) for s in seasons})
    ts, coef_rates = feat.attach_coefficients(ts, coefs)
    ts, n_imputed = feat.impute_season_median(ts, kept)
    ts = pd.concat([ts, feat.zscore_within_season(ts, kept)], axis=1)
    ts = feat.add_percentiles(ts, kept)
    ts["complete"] = ts["n_with_stats"] >= config.MIN_MATCHES_WITH_STATS
    ts["reached_final"] = ts["ko_stage"].ge(3)
    ts["is_target"] = ts["season"].isin(config.TARGET_SEASONS) & ts["reached_final"]
    ts["team_display"] = ts["team_id"].map(config.DISPLAY_NAMES).fillna(ts["team"])
    ts = ts.sort_values(["season", "team_id"]).reset_index(drop=True)

    notes = {
        "coverage": coverage,
        "dropped_features": [f for f in config.FEATURES if f not in kept],
        "imputed_values": n_imputed,
        "coef_match_rate": coef_rates,
        "coef_imputed": int(ts["coef_imputed"].sum()),
        "team_matches_without_stats": int((~tm["has_stats"]).sum()),
    }
    return Dataset(matches, tm, ts, labels.finals_table(matches), kept, notes)


def validate(ds: Dataset) -> None:
    """Raise ValidationError listing every broken spec §5.5 rule."""
    errors: list[str] = []
    ts, matches, finals = ds.team_seasons, ds.matches, ds.finals
    for season, g in ts.groupby("season"):
        season = int(season)
        if len(g) != config.field_size(season):
            errors.append(f"{season}: field has {len(g)} teams, expected {config.field_size(season)}")
        wrong = g[g["n_matches"] != config.phase_matches(season)]
        if len(wrong):
            errors.append(f"{season}: {len(wrong)} teams lack {config.phase_matches(season)} group/league matches")
        ko = g[g["in_ko"]]
        if len(ko) != config.ko_size(season):
            errors.append(f"{season}: knockout population is {len(ko)}, expected {config.ko_size(season)}")
        levels = [int((ko["ko_stage"] >= k).sum()) for k in (1, 2, 3, 4)]
        if levels != [8, 4, 2, 1]:
            errors.append(f"{season}: teams at ko_stage >= 1..4 are {levels}, expected [8, 4, 2, 1]")
    final_counts = matches.loc[matches["round"] == "Final"].groupby("season").size()
    for season in sorted(ts["season"].unique()):
        if int(final_counts.get(season, 0)) != 1:
            errors.append(f"{season}: expected exactly one final match")
    for season, expected in config.EXPECTED_FINALS.items():
        row = finals.loc[finals["season"] == season]
        if season not in set(ts["season"]):
            continue
        if row.empty or (row.iloc[0]["winner_id"], row.iloc[0]["runner_up_id"]) != (
            expected["winner_id"], expected["runner_up_id"]
        ):
            errors.append(f"{season}: final does not match the expected winner/runner-up ids")
    targets = ts.loc[ts["is_target"]]
    expected_targets = 2 * len([s for s in config.TARGET_SEASONS if s in set(ts["season"])])
    if len(targets) != expected_targets:
        errors.append(f"found {len(targets)} target finalists, expected {expected_targets}")
    incomplete = targets.loc[~targets["complete"]]
    if len(incomplete):
        names = ", ".join(f"{r.team} {r.season}" for r in incomplete.itertuples())
        errors.append(f"target finalists without enough stats: {names}")
    zcols = [f"z_{f}" for f in ds.features]
    if ts[zcols].isna().any().any():
        errors.append("NaN in model features after imputation")
    for season, rate in ds.notes.get("coef_match_rate", {}).items():
        if rate < config.COEF_MATCH_MIN:
            errors.append(f"{season}: only {rate:.0%} of clubs matched a coefficient")
    if errors:
        raise ValidationError("\n".join(errors))


def save(ds: Dataset, directory: Path = config.PROCESSED_DIR) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ds.matches.to_csv(directory / "matches.csv", index=False)
    ds.team_match_stats.to_csv(directory / "team_match_stats.csv", index=False)
    ds.team_seasons.to_csv(directory / "team_seasons.csv", index=False)
    ds.finals.to_csv(directory / "finals.csv", index=False)
    tm = ds.team_match_stats
    tm.loc[~tm["has_stats"].astype(bool), ["season", "match_id", "team_id"]].to_csv(
        directory / "missing_stats.csv", index=False
    )
    meta = {"features": ds.features, "notes": ds.notes}
    (directory / "dataset.json").write_text(json.dumps(meta, indent=2, default=str))


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV keeping id columns as strings (ids like '007' must survive)."""
    header = pd.read_csv(path, nrows=0).columns
    return pd.read_csv(path, dtype={c: str for c in ID_COLUMNS if c in header})


def load(directory: Path = config.PROCESSED_DIR) -> Dataset:
    directory = Path(directory)
    meta = json.loads((directory / "dataset.json").read_text())
    return Dataset(
        matches=read_csv(directory / "matches.csv"),
        team_match_stats=read_csv(directory / "team_match_stats.csv"),
        team_seasons=read_csv(directory / "team_seasons.csv"),
        finals=read_csv(directory / "finals.csv"),
        features=meta["features"],
        notes=meta["notes"],
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_dataset.py -q`
Expected: `7 passed`

- [ ] **Step 5: Tick Chunk 2 in `tasks/todo.md` (`- [ ]` → `- [x]`) and commit**

```bash
git add src/ucl/dataset.py tests/synthetic.py tests/conftest.py tests/test_dataset.py tasks/todo.md
git commit -m "feat: dataset build, spec validations and CSV persistence"
```

---

## Chunk 3: Real data and models

### Task 6: CLI fetch/build and the real data

**Files:**
- Create: `src/ucl/cli.py`, `scripts/check_standings.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing CLI test**

`tests/test_cli.py`:

```python
import pytest

from ucl import cli


def test_unknown_command_exits_with_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["nope"])
    assert exc.value.code == 2


def test_commands_run_in_pipeline_order():
    assert list(cli.COMMANDS)[:2] == ["fetch", "build"]
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_cli.py -q`
Expected: FAIL with `ImportError: cannot import name 'cli'`

- [ ] **Step 3: Write `src/ucl/cli.py`**

```python
"""Command line: uv run ucl fetch | build | model | analyze | report | all."""
from __future__ import annotations

import argparse
import sys

from . import config


def cmd_fetch(args: argparse.Namespace) -> int:
    from .dataset import fetch_all
    from .uefa import UefaClient

    failed = fetch_all(UefaClient(), config.SEASONS)
    if failed:
        print(f"{len(failed)} match-stat requests failed; run `uv run ucl fetch` again to resume.", file=sys.stderr)
        return 1
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    from . import dataset
    from .uefa import UefaClient

    ds = dataset.build(UefaClient(), config.SEASONS)
    try:
        dataset.validate(ds)
    except dataset.ValidationError as exc:
        print(f"dataset validation failed:\n{exc}", file=sys.stderr)
        return 2
    dataset.save(ds)
    ts, notes = ds.team_seasons, ds.notes
    print(f"built {len(ts)} team-seasons ({int(ts['in_ko'].sum())} knockout), {len(ds.features)} features")
    if notes["dropped_features"]:
        print("dropped (coverage < 90%):", ", ".join(notes["dropped_features"]))
    print(f"imputed values: {notes['imputed_values']}, coefficients imputed: {notes['coef_imputed']}, "
          f"team-matches without stats: {notes['team_matches_without_stats']}")
    return 0


COMMANDS = {"fetch": cmd_fetch, "build": cmd_build}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ucl", description="Why the best Champions League teams win.")
    parser.add_argument("command", choices=[*COMMANDS, "all"])
    parser.add_argument("--no-ai", action="store_true", help="skip the local-LLM analysis")
    parser.add_argument("--llm-model", default=config.LLM_MODEL, help="LM Studio model key")
    args = parser.parse_args(argv)
    names = list(COMMANDS) if args.command == "all" else [args.command]
    for name in names:
        code = COMMANDS[name](args)
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q`
Expected: all tests pass (`48 passed`).

- [ ] **Step 5: Write `scripts/check_standings.py`** (independent check of our points and goal difference)

```python
"""Compare the 10 finalists' computed group/league-phase points and goal difference with UEFA standings."""
import json
import urllib.request

from ucl import config, dataset
from ucl.uefa import USER_AGENT


def standings_item(season: int, team_id: str) -> dict | None:
    request = urllib.request.Request(config.STANDINGS_URL.format(season=season), headers={"User-Agent": USER_AGENT})
    groups = json.loads(urllib.request.urlopen(request, timeout=25).read())
    return next((item for g in groups for item in g["items"] if str(item["team"]["id"]) == team_id), None)


def main() -> int:
    ts = dataset.load().team_seasons
    bad = 0
    for r in ts.loc[ts["is_target"]].itertuples():
        label = f"{config.season_label(int(r.season))} {r.team_display}"
        item = standings_item(int(r.season), r.team_id)
        if item is None:
            print(f"BAD {label}: team id {r.team_id} not found in UEFA standings")
            bad += 1
            continue
        ours = (round(r.points_pg * r.n_matches), round(r.goal_diff_pg * r.n_matches))
        theirs = (int(item["points"]), int(item["goalDifference"]))
        bad += ours != theirs
        print(f"{'OK ' if ours == theirs else 'BAD'} {label}: ours {ours[0]} pts {ours[1]:+d} GD | "
              f"UEFA {theirs[0]} pts {theirs[1]:+d} GD")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Fetch the real data**

Run: `uv run ucl fetch`
Expected: 15 lines, from `2012: 125 matches, 96 phase matches, stats missing 0, failed 0` to `2026: 189 matches, 144 phase matches, ...`, then exit 0. 2020 shows 119 matches because its quarter- and semi-finals were single legs. About 1,536 stats requests at 4 workers take roughly 3–8 minutes. If it exits 1, run it again; it resumes from cache. An unmapped round name raises here, in `labels.match_rows`. Add the name to `config.ROUND_DEPTH`.

If the same ids fail on every run (a persistent 5xx or a 403), stop and inspect one of them directly before going on; don't loop. If "stats missing" is large for a season, look at a cached marker. Markers come only from 404/410, so a large count means UEFA really has no stats for those matches. Record that in `tasks/todo.md`.

- [ ] **Step 7: Build and validate**

Run: `uv run ucl build`
Expected: `built 488 team-seasons (256 knockout), 16 features`, followed by the imputation counts.
- A stat missing in an early season shows up as fewer than 16 features, with a "dropped (coverage < 90%)" line. It is not a validation failure. Record it in `tasks/todo.md`.
- If validation fails, read the listed rules, fix the root cause in the module concerned, run `uv run pytest -q` and re-run the build.
- A few "coefficients imputed" beyond the debutants are expected: clubs with aliased ids, such as Steaua/FCSB, look like debutants to the coefficient join.

- [ ] **Step 8: Cross-check the 10 finalists against UEFA standings**

Run: `uv run python scripts/check_standings.py`
Expected: 10 lines starting with `OK `. One should read `OK  2025-26 Arsenal: ours 24 pts +19 GD | UEFA 24 pts +19 GD`. Exit 0.

- [ ] **Step 9: Commit (processed data included; the raw cache is git-ignored)**

`git add -A` also picks up any root-cause fix made in Step 7.

```bash
git add -A
git commit -m "feat: fetch/build CLI; real dataset built and cross-checked against UEFA standings"
```

### Task 7: Leave-one-season-out models and metrics

**Files:**
- Create: `src/ucl/model.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_model.py`:

```python
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from synthetic import make_synthetic_dataset

from ucl import model


@pytest.fixture(scope="module")
def small():
    ds = make_synthetic_dataset(seasons=tuple(range(2021, 2027)))
    return ds, model.knockout_population(ds.team_seasons)


def test_one_prediction_and_shap_row_per_knockout_team(small):
    ds, ko = small
    out = model.loso(ko, ds.features)
    assert len(out.predictions) == len(ko) == len(out.shap)
    assert not out.predictions.duplicated(["season", "team_id"]).any()
    assert set(out.predictions["rank_in_season"].groupby(out.predictions["season"]).min()) == {1}
    # two teams reach each final, so each season's probabilities add up to 2
    np.testing.assert_allclose(out.predictions.groupby("season")["p_final"].sum(), 2.0)
    assert out.predictions["p_final"].max() <= 1.0


def test_scale_to_two_sums_to_two_and_caps_at_one():
    assert model.scale_to_two([0.1, 0.3, 0.2, 0.4]) == pytest.approx([0.2, 0.6, 0.4, 0.8])
    assert model.scale_to_two([0.9, 0.05, 0.05]) == pytest.approx([1.0, 0.5, 0.5])


def test_evaluate_matches_hand_computed_values():
    pred = pd.DataFrame({
        "season": [1] * 5 + [2] * 5,
        "ko_stage": [4, 3, 2, 1, 0] * 2,
        "exp_stage": [5, 1, 4, 3, 2, 1, 5, 4, 3, 2],
        "reached_final": [True, True, False, False, False] * 2,
        "p_final": [0.6, 0.1, 0.3, 0.2, 0.05, 0.4, 0.5, 0.45, 0.1, 0.05],
        "base_rate": [0.4] * 10,
    })
    m = model.evaluate(pred)
    # rho: 1 - 6*12/120 = 0.4 and 1 - 6*20/120 = 0.0; one finalist per season ranks 5th by expected stage;
    # AUC = 19.5 / 24; Brier = 1.9275 / 10; base-rate Brier = (4*0.36 + 6*0.16) / 10
    assert m["spearman_by_season"] == {1: pytest.approx(0.4), 2: pytest.approx(0.0, abs=1e-12)}
    assert m["spearman_mean"] == pytest.approx(0.2)
    assert m["finalists_in_top4"] == pytest.approx(0.5)
    assert m["auc"] == pytest.approx(0.8125)
    assert m["brier"] == pytest.approx(0.19275)
    assert m["brier_base_rate"] == pytest.approx(0.24)
    assert (m["n_knockout"], m["n_finalists"], m["n_seasons"]) == (10, 4, 2)


def test_loso_never_trains_on_held_out_season_or_incomplete_rows(small, monkeypatch):
    ds, ko = small
    ko = ko.copy()
    ko.loc[0, "complete"] = False
    seen = []
    real_fit = GradientBoostingRegressor.fit

    def spy(self, X, y, *args, **kwargs):
        seen.append(X.index)
        return real_fit(self, X, y, *args, **kwargs)

    monkeypatch.setattr(GradientBoostingRegressor, "fit", spy)
    out = model.loso(ko, ds.features, with_shap=False)
    assert len(seen) == ko["season"].nunique()  # one fit per fold, so the loop below is not vacuous
    for season, index in zip(sorted(ko["season"].unique()), seen):
        train = ko.loc[index]
        assert season not in set(train["season"])
        assert train["complete"].all()
    assert len(out.predictions) == len(ko)  # the incomplete row is still predicted


def test_runs_are_deterministic(small):
    ds, ko = small
    a, b = model.loso(ko, ds.features), model.loso(ko, ds.features)
    pd.testing.assert_frame_equal(a.predictions, b.predictions)
    pd.testing.assert_frame_equal(a.shap, b.shap)


def test_shap_values_add_up_to_the_prediction(small):
    ds, ko = small
    out = model.loso(ko, ds.features)
    total = out.shap["base_value"] + out.shap[[f"shap_{f}" for f in ds.features]].sum(axis=1)
    np.testing.assert_allclose(total.to_numpy(), out.predictions["exp_stage"].to_numpy(), atol=1e-6)


def test_evaluate_reports_the_spec_metrics(small):
    ds, ko = small
    metrics = model.evaluate(model.loso(ko, ds.features, with_shap=False).predictions)
    assert set(metrics) >= {"spearman_mean", "spearman_by_season", "finalists_in_top4", "auc", "brier",
                            "brier_base_rate", "n_knockout", "n_finalists", "n_seasons"}
    assert metrics["n_finalists"] == 12 and metrics["n_seasons"] == 6
    assert 0.0 <= metrics["auc"] <= 1.0
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_model.py -q`
Expected: FAIL with `ImportError: cannot import name 'model'`

- [ ] **Step 3: Write `src/ucl/model.py` (core)**

```python
"""Leave-one-season-out models, SHAP drivers, ablation and winners-vs-runners-up (spec §6)."""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.stats import binomtest, spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from . import config


@dataclass
class LosoOutput:
    predictions: pd.DataFrame
    shap: pd.DataFrame | None
    fold_coefs: list[np.ndarray]


@dataclass
class ModelResults:
    predictions: pd.DataFrame
    shap: pd.DataFrame
    drivers: pd.DataFrame
    ablation: pd.DataFrame
    metrics: dict
    finals_compare: pd.DataFrame


def knockout_population(team_seasons: pd.DataFrame) -> pd.DataFrame:
    ko = team_seasons.loc[team_seasons["in_ko"].astype(bool)].copy()
    ko["ko_stage"] = ko["ko_stage"].astype(int)
    ko["reached_final"] = ko["reached_final"].astype(bool)
    return ko.sort_values(["season", "team_id"]).reset_index(drop=True)


def scale_to_two(p) -> np.ndarray:
    """Rescale one season's P(final) to sum to 2 (two finalists) with none above 1.

    Model B trains mostly on 16-team seasons, so its raw probabilities run high in 24-team seasons.
    """
    p = np.asarray(p, dtype=float)
    capped = np.zeros(len(p), dtype=bool)
    while True:
        scaled = np.where(capped, 1.0, p * (2.0 - capped.sum()) / p[~capped].sum())
        newly = (scaled > 1.0) & ~capped
        if not newly.any():
            return scaled
        capped |= newly


def loso(ko: pd.DataFrame, features: list[str], with_shap: bool = True) -> LosoOutput:
    """Each season predicted by models trained on the complete rows of the other seasons."""
    cols = [f"z_{f}" for f in features]
    predictions, shap_frames, coefs = [], [], []
    for season in sorted(ko["season"].unique()):
        train = ko.loc[(ko["season"] != season) & ko["complete"].astype(bool)]
        test = ko.loc[ko["season"] == season]
        gbr = GradientBoostingRegressor(**config.GBR_PARAMS).fit(train[cols], train["ko_stage"])
        logit = LogisticRegression(**config.LOGIT_PARAMS).fit(train[cols], train["reached_final"].astype(int))
        coefs.append(logit.coef_[0].copy())
        fold = pd.DataFrame({
            "season": test["season"].to_numpy(),
            "team_id": test["team_id"].to_numpy(),
            "ko_stage": test["ko_stage"].to_numpy(),
            "reached_final": test["reached_final"].to_numpy(),
            "exp_stage": gbr.predict(test[cols]),
            "p_final": scale_to_two(logit.predict_proba(test[cols])[:, 1]),
        })
        fold["rank_in_season"] = fold["p_final"].rank(ascending=False, method="min").astype(int)
        fold["ko_size"] = len(test)
        fold["base_rate"] = 2 / len(test)
        predictions.append(fold)
        if with_shap:
            explainer = shap.TreeExplainer(gbr)
            values = pd.DataFrame(explainer.shap_values(test[cols]), columns=[f"shap_{f}" for f in features])
            values.insert(0, "base_value", float(np.ravel(explainer.expected_value)[0]))
            values.insert(0, "team_id", test["team_id"].to_numpy())
            values.insert(0, "season", test["season"].to_numpy())
            shap_frames.append(values)
    return LosoOutput(
        predictions=pd.concat(predictions, ignore_index=True),
        shap=pd.concat(shap_frames, ignore_index=True) if with_shap else None,
        fold_coefs=coefs,
    )


def evaluate(pred: pd.DataFrame) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # constant-input warnings in degenerate seasons
        rho = {int(s): float(spearmanr(g["exp_stage"], g["ko_stage"]).statistic) for s, g in pred.groupby("season")}
    finalists = pred["reached_final"].astype(bool)
    stage_rank = pred.groupby("season")["exp_stage"].rank(ascending=False, method="min")
    y = finalists.astype(int)
    return {
        "spearman_mean": float(np.nanmean(list(rho.values()))),
        "spearman_by_season": rho,
        "finalists_in_top4": float((stage_rank[finalists] <= 4).mean()),
        "auc": float(roc_auc_score(y, pred["p_final"])),
        "brier": float(brier_score_loss(y, pred["p_final"])),
        "brier_base_rate": float(brier_score_loss(y, pred["base_rate"])),
        "n_knockout": int(len(pred)),
        "n_finalists": int(y.sum()),
        "n_seasons": int(pred["season"].nunique()),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_model.py -q`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add src/ucl/model.py tests/test_model.py
git commit -m "feat: leave-one-season-out gradient boosting + logistic models with TreeSHAP"
```

### Task 8: Drivers, ablation, winners-vs-runners-up, persistence

**Files:**
- Modify: `src/ucl/model.py` (append)
- Test: `tests/test_model_analysis.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_model_analysis.py`:

```python
import numpy as np
import pandas as pd
import pytest
from synthetic import make_synthetic_dataset

from ucl import config, model


@pytest.fixture(scope="module")
def results():
    ds = make_synthetic_dataset(seasons=tuple(range(2019, 2027)))
    return ds, model.run(ds.team_seasons, ds.finals, ds.features)


def test_robust_needs_sign_agreement_in_twelve_of_fifteen_folds():
    ko = pd.DataFrame({"season": [1, 1, 2, 2], "team_id": ["a", "b", "c", "d"],
                       "z_shots_pg": [-1.0, 1.0, -0.5, 0.5], "z_possession_pct": [-1.0, 1.0, -0.5, 0.5]})
    shap_df = ko[["season", "team_id"]].assign(
        base_value=0.0, shap_shots_pg=[-0.2, 0.2, -0.1, 0.1], shap_possession_pct=[-0.1, 0.1, -0.05, 0.05])
    coefs = [np.array([1.0 if i < 12 else -1.0, 1.0 if i < 11 else -1.0]) for i in range(15)]
    d = model.drivers(ko, shap_df, coefs, ["shots_pg", "possession_pct"]).set_index("feature")
    assert (d.loc["shots_pg", "sign_agree_folds"], d.loc["shots_pg", "label"]) == (12, "robust")
    assert (d.loc["possession_pct", "sign_agree_folds"], d.loc["possession_pct", "label"]) == (11, "model-dependent")


def test_sign_counts_treat_ties():
    higher, lower, tied, p = model.sign_counts(np.array([0.5, -0.2, 0.0, 0.0, 1.0]))
    assert (higher, lower, tied) == (2, 1, 2)
    assert p == pytest.approx(1.0)  # binomial test on the 3 non-tied finals


def test_holm_adjustment():
    assert model.holm([0.01, 0.04, 0.03]).tolist() == pytest.approx([0.03, 0.06, 0.06])


def test_driver_labels_only_on_the_top_six(results):
    _, res = results
    d = res.drivers
    assert list(d["rank"]) == list(range(1, len(d) + 1))
    assert (d.loc[d["rank"] > config.TOP_DRIVERS, "label"] == "").all()
    assert set(d.loc[d["rank"] <= config.TOP_DRIVERS, "label"]) <= {"robust", "model-dependent"}
    assert d["importance"].is_monotonic_decreasing


def test_ablation_has_the_seven_feature_sets(results):
    _, res = results
    assert list(res.ablation["feature_set"]) == [
        "pedigree", "results", "style", "all", "all minus pedigree", "all minus results", "all minus style",
    ]


def test_finals_compare_counts_every_final(results):
    ds, res = results
    fc = res.finals_compare
    assert len(fc) == len(ds.features)
    assert ((fc["higher"] + fc["lower"] + fc["tied"]) == len(ds.finals)).all()
    assert {f"diff_{s}" for s in config.TARGET_SEASONS} <= set(fc.columns)
    assert (fc["p_holm"] >= fc["p"] - 1e-12).all()


def test_predictions_carry_display_columns(results):
    _, res = results
    assert {"team_display", "is_target", "complete"} <= set(res.predictions.columns)
    assert int(res.predictions["is_target"].sum()) == 10


def test_feature_list_follows_config_order(results):
    ds, res = results
    assert model.feature_list(res) == ds.features


def test_save_and_load_roundtrip(results, tmp_path):
    _, res = results
    model.save(res, tmp_path)
    loaded = model.load(tmp_path)
    assert loaded.metrics["n_seasons"] == res.metrics["n_seasons"]
    assert len(loaded.shap) == len(res.shap)
    assert loaded.predictions["team_id"].dtype != np.int64
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_model_analysis.py -q`
Expected: FAIL/ERROR with `AttributeError: module 'ucl.model' has no attribute ...` (`drivers`, `sign_counts`, `holm`, `run`)

- [ ] **Step 3: Append to `src/ucl/model.py`**

```python
def feature_sets(features: list[str]) -> dict[str, list[str]]:
    groups = {g: [f for f in fs if f in features] for g, fs in config.FEATURE_GROUPS.items()}
    sets: dict[str, list[str]] = dict(groups)
    sets["all"] = list(features)
    for name, fs in groups.items():
        sets[f"all minus {name}"] = [f for f in features if f not in fs]
    return {name: fs for name, fs in sets.items() if fs}


def ablation(ko: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for name, fs in feature_sets(features).items():
        metrics = evaluate(loso(ko, fs, with_shap=False).predictions)
        rows.append({"feature_set": name, "n_features": len(fs),
                     "spearman": metrics["spearman_mean"], "auc": metrics["auc"]})
    return pd.DataFrame(rows)


def drivers(ko: pd.DataFrame, shap_df: pd.DataFrame, fold_coefs: list[np.ndarray],
            features: list[str]) -> pd.DataFrame:
    merged = shap_df.merge(ko[["season", "team_id", *[f"z_{f}" for f in features]]], on=["season", "team_id"])
    coefs = np.vstack(fold_coefs)
    rows = []
    for i, f in enumerate(features):
        values = merged[f"shap_{f}"].to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = spearmanr(merged[f"z_{f}"].to_numpy(), values).statistic
        direction = 0 if np.isnan(corr) or corr == 0 else int(np.sign(corr))
        rows.append({
            "feature": f,
            "group": config.FEATURE_GROUP[f],
            "label_text": config.FEATURE_META[f][0],
            "importance": float(np.mean(np.abs(values))),
            "direction": direction,
            "logit_coef_mean": float(coefs[:, i].mean()),
            "sign_agree_folds": int((np.sign(coefs[:, i]) == direction).sum()) if direction else 0,
        })
    out = pd.DataFrame(rows).sort_values("importance", ascending=False, kind="stable").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    top = out["rank"] <= config.TOP_DRIVERS
    robust = out["sign_agree_folds"] >= math.ceil(config.ROBUST_SHARE * len(fold_coefs))
    out["label"] = np.where(top & robust, "robust", np.where(top, "model-dependent", ""))
    return out


def holm(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def sign_counts(diffs: np.ndarray, tol: float = 1e-9) -> tuple[int, int, int, float]:
    """(higher, lower, tied, two-sided sign-test p on the non-tied finals)."""
    higher, lower = int((diffs > tol).sum()), int((diffs < -tol).sum())
    n = higher + lower
    p = float(binomtest(higher, n, 0.5).pvalue) if n else 1.0
    return higher, lower, len(diffs) - n, p


def finals_compare(team_seasons: pd.DataFrame, finals: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Winner minus runner-up z-scores across every final, with sign tests and Holm adjustment."""
    by_key = team_seasons.set_index(["season", "team_id"])
    diffs: dict[str, dict[int, float]] = {f: {} for f in features}
    for fr in finals.itertuples(index=False):
        winner = by_key.loc[(fr.season, fr.winner_id)]
        runner_up = by_key.loc[(fr.season, fr.runner_up_id)]
        for f in features:
            diffs[f][int(fr.season)] = float(winner[f"z_{f}"] - runner_up[f"z_{f}"])
    rows = []
    for f in features:
        values = np.array(list(diffs[f].values()))
        higher, lower, tied, p = sign_counts(values)
        row = {"feature": f, "label_text": config.FEATURE_META[f][0], "higher": higher, "lower": lower,
               "tied": tied, "mean_diff": float(values.mean()), "p": p}
        row.update({f"diff_{s}": diffs[f].get(s, np.nan) for s in config.TARGET_SEASONS})
        rows.append(row)
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"])
    return out


def run(team_seasons: pd.DataFrame, finals: pd.DataFrame, features: list[str]) -> ModelResults:
    ko = knockout_population(team_seasons)
    main = loso(ko, features)
    display = team_seasons[["season", "team_id", "team_display", "is_target", "complete"]]
    return ModelResults(
        predictions=main.predictions.merge(display, on=["season", "team_id"], how="left"),
        shap=main.shap,
        drivers=drivers(ko, main.shap, main.fold_coefs, features),
        ablation=ablation(ko, features),
        metrics=evaluate(main.predictions),
        finals_compare=finals_compare(team_seasons, finals, features),
    )


def feature_list(results: ModelResults) -> list[str]:
    """Active features in config order (drivers are sorted by importance)."""
    active = set(results.drivers["feature"])
    return [f for f in config.FEATURES if f in active]


FRAMES = {
    "predictions": "predictions.csv",
    "shap": "shap.csv",
    "drivers": "drivers.csv",
    "ablation": "ablation.csv",
    "finals_compare": "finals_compare.csv",
}


def save(results: ModelResults, directory: Path = config.OUT_DIR) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for attr, name in FRAMES.items():
        getattr(results, attr).to_csv(directory / name, index=False)
    (directory / "metrics.json").write_text(json.dumps(results.metrics, indent=2))


def load(directory: Path = config.OUT_DIR) -> ModelResults:
    from .dataset import read_csv

    directory = Path(directory)
    frames = {attr: read_csv(directory / name) for attr, name in FRAMES.items()}
    frames["drivers"]["label"] = frames["drivers"]["label"].fillna("")
    return ModelResults(metrics=json.loads((directory / "metrics.json").read_text()), **frames)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_model.py tests/test_model_analysis.py -q`
Expected: `16 passed`

- [ ] **Step 5: Commit**

```bash
git add src/ucl/model.py tests/test_model_analysis.py
git commit -m "feat: SHAP drivers with robustness labels, ablation and winners-vs-runners-up tests"
```

### Task 9: CLI `model` stage on the real data

**Files:**
- Modify: `src/ucl/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Update the CLI order test (failing)**

In `tests/test_cli.py`, replace `test_commands_run_in_pipeline_order` with:

```python
def test_commands_run_in_pipeline_order():
    assert list(cli.COMMANDS)[:3] == ["fetch", "build", "model"]
```

Run: `uv run pytest tests/test_cli.py -q`
Expected: FAIL (`['fetch', 'build'] != ['fetch', 'build', 'model']`)

- [ ] **Step 2: Add `cmd_model` to `src/ucl/cli.py`** (above `COMMANDS`) and register it

```python
def cmd_model(args: argparse.Namespace) -> int:
    from . import dataset, model

    ds = dataset.load()
    results = model.run(ds.team_seasons, ds.finals, ds.features)
    model.save(results)
    m = results.metrics
    print(f"Spearman {m['spearman_mean']:.2f} | AUC {m['auc']:.2f} | Brier {m['brier']:.3f} "
          f"(base rate {m['brier_base_rate']:.3f}) | finalists in top 4: {m['finalists_in_top4']:.0%}")
    print(results.drivers.head(config.TOP_DRIVERS)[["feature", "importance", "direction", "label"]]
          .to_string(index=False))
    print(results.ablation.to_string(index=False))
    return 0
```

```python
COMMANDS = {"fetch": cmd_fetch, "build": cmd_build, "model": cmd_model}
```

- [ ] **Step 3: Run the tests**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 4: Run the model on the real data**

Run: `uv run ucl model`
Expected: a metrics line, the top-6 drivers table and the 7-row ablation table, finishing in under a minute. Sanity checks:
- AUC is between 0.5 and 1.
- Brier is at or below the base-rate Brier. If not, note it honestly in `tasks/todo.md` rather than tuning.

Then check the real-data invariants:

```bash
uv run python -c "
import json, pandas as pd
p = pd.read_csv('out/predictions.csv'); s = pd.read_csv('out/shap.csv'); m = json.load(open('out/metrics.json'))
assert len(p) == len(s) == m['n_knockout'] == 256, (len(p), len(s), m['n_knockout'])
assert (m['n_seasons'], m['n_finalists']) == (15, 30), m
assert int(p['is_target'].sum()) == 10
assert ((p.groupby('season')['p_final'].sum() - 2).abs() < 1e-9).all()
print('invariants ok')
"
```

Expected: `invariants ok`.

- [ ] **Step 5: Tick Chunk 3 in `tasks/todo.md` (`- [ ]` → `- [x]`) and commit**

```bash
git add -A
git commit -m "feat: model stage; out-of-fold predictions, drivers and ablation on the real data"
```

---

## Chunk 4: Grounding checks and the LM Studio client

### Task 10: Grounding and output-validity checks (pure)

**Files:**
- Create: `src/ucl/grounding.py`
- Test: `tests/test_grounding.py`, `tests/test_output_validity.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_grounding.py`:

```python
import numpy as np

from ucl.grounding import check_grounding, fact_numbers


def test_percent_matches_a_fraction_or_a_percent():
    assert check_grounding("They had a 31% chance.", {"p": 0.31}) == []
    assert check_grounding("They had a 31% chance.", {"p": 31}) == []
    assert check_grounding("A rate of 0.31 per game.", {"p": 31}) == []


def test_rounding_tolerance_follows_the_text_precision():
    assert check_grounding("17.4 shots a game", {"shots": 18.43, "other": 17.38}) == []
    assert check_grounding("18.4 shots", {"shots": 18.43}) == []
    assert check_grounding("18.5 shots", {"shots": 18.45}) == []


def test_invented_numbers_are_flagged():
    assert check_grounding("AUC of 0.88", {"auc": 0.78}) == ["0.88"]
    assert check_grounding("a 47% chance", {"p": 0.31}) == ["47"]


def test_years_seasons_and_small_integers_are_ignored():
    text = "In 2025-26 and 2025–26 (and 2024) they made the top 3 after 2025/26, as in 2011 and 2027."
    assert check_grounding(text, {}) == []


def test_numbers_just_outside_the_year_range_are_checked():
    assert check_grounding("2010 and 2028", {}) == ["2010", "2028"]


def test_numbers_inside_fact_strings_and_keys_count():
    facts = {"Actual result": "Winner: beat Inter 5-0 in the final", "Teams since 2011-12 (488)": 1}
    assert check_grounding("one of 488 teams", facts) == []
    assert 488.0 in fact_numbers(facts)


def test_numpy_numbers_count_as_facts():
    assert check_grounding("ranked 24th", {"n": np.int64(24)}) == []


def test_signs_are_ignored_when_matching():
    assert check_grounding("down by -0.42 and up +1.83", {"a": 0.42, "b": -1.83}) == []


def test_each_unsupported_number_is_reported_once():
    assert check_grounding("47 and 47 again", {}) == ["47"]
```

`tests/test_output_validity.py`:

```python
from ucl.grounding import normalize_headings, strip_think, validate_output

HEADINGS = ["How they got there", "Weak spots"]
GOOD = "## How they got there\nText.\n\n## Weak spots\nMore text."


def test_good_answer_is_valid():
    assert validate_output(GOOD, "stop", HEADINGS) is None


def test_length_cutoff_is_invalid():
    assert "cut off" in validate_output(GOOD, "length", HEADINGS)


def test_empty_answer_is_invalid():
    assert "empty" in validate_output("   ", "stop", HEADINGS)


def test_unclosed_think_is_invalid():
    assert "unclosed" in validate_output("<think>still thinking about " + GOOD, "stop", HEADINGS)


def test_closed_and_stray_think_tags_are_stripped():
    assert strip_think("<think>plan</think>\n" + GOOD) == GOOD
    assert strip_think("leaked reasoning</think>\n" + GOOD) == GOOD
    assert validate_output("<think>plan</think>\n" + GOOD, "stop", HEADINGS) is None


def test_missing_heading_is_invalid():
    reason = validate_output("## How they got there\nText only.", "stop", HEADINGS)
    assert "Weak spots" in reason


def test_heading_variants_are_accepted_and_normalised():
    text = "## how they got there:\nx\n**2. Weak spots**\ny"
    assert validate_output(text, "stop", HEADINGS) is None
    assert normalize_headings(text, HEADINGS) == "## How they got there\nx\n## Weak spots\ny"


def test_crlf_answers_are_valid():
    assert validate_output("## How they got there\r\nx\r\n## Weak spots\r\ny", "stop", HEADINGS) is None
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_grounding.py tests/test_output_validity.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ucl.grounding'`

- [ ] **Step 3: Write `src/ucl/grounding.py`**

```python
"""Pure checks on LLM output: answer validity, heading normalisation and numeric grounding (spec §7)."""
from __future__ import annotations

import math
import numbers
import re

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)
_SEASON = re.compile(r"\b(?:19|20)\d{2}\s*[-–—/]\s*\d{2,4}\b")
_NUMBER = re.compile(r"(?<![\w.])[-−+]?\d+(?:,\d{3})*(?:\.\d+)?")


def strip_think(text: str | None) -> str:
    body = _THINK_BLOCK.sub("", text or "")
    if "</think>" in body and "<think>" not in body:
        body = body.rsplit("</think>", 1)[1]  # reasoning leaked without its opening tag
    return body.strip()


def _heading_key(line: str) -> str:
    """A line reduced to bare text: no #s, numbering, bold/italic markers or trailing colon."""
    core = line.strip().lstrip("#").strip().strip("*_").strip()
    core = re.sub(r"^\d+[.)]\s*", "", core).strip("*_").strip()
    return core.rstrip(":.").strip().strip("*_").strip().casefold()


def _is_heading(line: str, heading: str) -> bool:
    """True when the line is only the heading (#s, numbering, bold and a trailing colon allowed)."""
    return len(line) <= 200 and _heading_key(line) == heading.casefold()


def validate_output(text: str | None, finish_reason: str | None, required_headings: list[str]) -> str | None:
    """Why an answer is unusable, or None if it is fine."""
    if finish_reason == "length":
        return "the answer was cut off at the token limit"
    body = strip_think(text)
    if "<think>" in body:
        return "the answer contains an unclosed <think> block"
    if not body:
        return "the answer was empty"
    lines = body.splitlines()
    missing = [h for h in required_headings if not any(_is_heading(line, h) for line in lines)]
    if missing:
        return "missing heading(s): " + "; ".join(missing)
    return None


def normalize_headings(text: str, headings: list[str]) -> str:
    """Rewrite each recognised heading line as '## Heading' so later parsing is uniform."""
    out = []
    for line in text.splitlines():
        match = next((h for h in headings if _is_heading(line, h)), None)
        out.append(f"## {match}" if match else line)
    return "\n".join(out)


def _value(token: str) -> float:
    return float(token.replace(",", "").replace("−", "-").lstrip("+"))


def _decimals(token: str) -> int:
    return len(token.split(".", 1)[1]) if "." in token else 0


def fact_numbers(facts) -> list[float]:
    """Every number in a facts structure, including numbers inside strings and keys."""
    found: list[float] = []

    def walk(obj) -> None:
        if isinstance(obj, bool):
            return
        if isinstance(obj, numbers.Real):
            if math.isfinite(float(obj)):
                found.append(float(obj))
        elif isinstance(obj, str):
            found.extend(_value(t) for t in _NUMBER.findall(obj))
        elif isinstance(obj, dict):
            for key, value in obj.items():
                walk(key)
                walk(value)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                walk(value)

    walk(facts)
    return found


def _ignored(token: str) -> bool:
    if "." in token:
        return False
    value = abs(_value(token))
    return value <= 10 or 2011 <= value <= 2027


def check_grounding(text: str, facts) -> list[str]:
    """Numbers in `text` that no fact supports, using the spec §7 tolerance rule."""
    values = [abs(v) for v in fact_numbers(facts)]
    candidates = [c for v in values for c in (v, v * 100, v / 100)]
    unsupported: list[str] = []
    for token in _NUMBER.findall(_SEASON.sub(" ", text)):
        if _ignored(token):
            continue
        target, tolerance = abs(_value(token)), 0.5 * 10 ** -_decimals(token) + 1e-9
        if not any(abs(target - c) <= tolerance for c in candidates):
            clean = token.lstrip("+-−")
            if clean not in unsupported:
                unsupported.append(clean)
    return unsupported
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_grounding.py tests/test_output_validity.py -q`
Expected: `17 passed`

- [ ] **Step 5: Commit**

```bash
git add src/ucl/grounding.py tests/test_grounding.py tests/test_output_validity.py
git commit -m "feat: pure output-validity, heading normalisation and numeric grounding checks"
```

### Task 11: LM Studio client with response cache

**Files:**
- Create: `src/ucl/llm.py`
- Test: `tests/test_llm.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_llm.py`:

```python
import http.client
import urllib.error

import pytest

from ucl.llm import ChatResult, LLMError, LMStudio


class FakePost:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.bodies = response, error, []

    def __call__(self, url, body, timeout):
        self.bodies.append(body)
        if self.error:
            raise self.error
        return self.response


def reply(content, finish="stop"):
    return {"choices": [{"message": {"content": content, "reasoning_content": "hidden"}, "finish_reason": finish}]}


def test_chat_returns_content_and_finish_reason(tmp_path):
    llm = LMStudio(cache_dir=tmp_path, post=FakePost(reply("## Hi\nThere")))
    assert llm.chat([{"role": "user", "content": "x"}]) == ChatResult("## Hi\nThere", "stop")


def test_request_disables_reasoning_and_uses_the_model(tmp_path):
    post = FakePost(reply("ok"))
    LMStudio(model="m/key", cache_dir=tmp_path, post=post).chat([{"role": "user", "content": "x"}])
    body = post.bodies[0]
    assert body["model"] == "m/key" and body["reasoning_effort"] == "none" and body["max_tokens"] == 8000


def test_identical_requests_are_served_from_cache(tmp_path):
    post = FakePost(reply("ok"))
    llm = LMStudio(cache_dir=tmp_path, post=post)
    llm.chat([{"role": "user", "content": "x"}])
    llm.chat([{"role": "user", "content": "x"}])
    llm.chat([{"role": "user", "content": "y"}])
    assert len(post.bodies) == 2


def test_cache_key_includes_the_model(tmp_path):
    post = FakePost(reply("ok"))
    LMStudio(model="a/one", cache_dir=tmp_path, post=post).chat([{"role": "user", "content": "x"}])
    LMStudio(model="b/two", cache_dir=tmp_path, post=post).chat([{"role": "user", "content": "x"}])
    assert len(post.bodies) == 2


def test_invalid_responses_are_cached_too(tmp_path):
    post = FakePost(reply("", finish="length"))
    llm = LMStudio(cache_dir=tmp_path, post=post)
    assert llm.chat([{"role": "user", "content": "x"}]).finish_reason == "length"
    assert llm.chat([{"role": "user", "content": "x"}]).finish_reason == "length"
    assert len(post.bodies) == 1


@pytest.mark.parametrize("error", [urllib.error.URLError("refused"), http.client.IncompleteRead(b""), TimeoutError()])
def test_failures_raise_llm_error_and_are_not_cached(tmp_path, error):
    llm = LMStudio(cache_dir=tmp_path, post=FakePost(error=error))
    with pytest.raises(LLMError):
        llm.chat([{"role": "user", "content": "x"}])
    assert not list(tmp_path.glob("*.json"))


def test_ensure_ready_never_raises(tmp_path, monkeypatch):
    llm = LMStudio(base_url="http://127.0.0.1:9/v1", cache_dir=tmp_path)
    monkeypatch.setattr(llm, "_lms", lambda *a, **k: False)
    assert llm.ensure_ready() is False


def test_ensure_ready_when_already_up_does_not_call_lms(tmp_path, monkeypatch):
    llm = LMStudio(cache_dir=tmp_path)
    monkeypatch.setattr(llm, "_server_up", lambda: True)
    monkeypatch.setattr(llm, "_model_loaded", lambda: True)

    def no_lms(*args, **kwargs):
        raise AssertionError("lms must not run")

    monkeypatch.setattr(llm, "_lms", no_lms)
    assert llm.ensure_ready() is True


def test_ensure_ready_loads_the_model_with_its_context_length(tmp_path, monkeypatch):
    llm = LMStudio(model="m/key", cache_dir=tmp_path)
    state, calls = {"loaded": False}, []

    def fake_lms(*args, timeout):
        calls.append(args)
        state["loaded"] = True
        return True

    monkeypatch.setattr(llm, "_server_up", lambda: True)
    monkeypatch.setattr(llm, "_model_loaded", lambda: state["loaded"])
    monkeypatch.setattr(llm, "_lms", fake_lms)
    assert llm.ensure_ready() is True
    assert calls == [("load", "m/key", "--context-length", "16384", "-y")]


def test_ensure_ready_gives_up_when_the_load_command_fails(tmp_path, monkeypatch):
    llm = LMStudio(cache_dir=tmp_path)
    checks = []
    monkeypatch.setattr(llm, "_server_up", lambda: True)
    monkeypatch.setattr(llm, "_model_loaded", lambda: checks.append(1) or False)
    monkeypatch.setattr(llm, "_lms", lambda *args, timeout: False)
    assert llm.ensure_ready() is False
    assert len(checks) == 1  # gave up at once instead of polling
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_llm.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ucl.llm'`

- [ ] **Step 3: Write `src/ucl/llm.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_llm.py -q`
Expected: `12 passed` (10 test functions; the failure test runs 3 cases). All run in well under a second.

- [ ] **Step 5: Tick Chunk 4 in `tasks/todo.md` (`- [ ]` → `- [x]`) and commit**

```bash
git add src/ucl/llm.py tests/test_llm.py tasks/todo.md
git commit -m "feat: LM Studio client with request-keyed response cache and readiness checks"
```

---

## Chunk 5: Fact sheets and the analyst

### Task 12: Fact sheets

**Files:**
- Create: `src/ucl/facts.py`
- Test: `tests/test_facts.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_facts.py` (uses the session-scoped `built` fixture from `tests/conftest.py`):

```python
import json

from ucl import facts


def test_result_text_handles_penalties_and_orientation():
    final = {"winner_id": "52747", "winner": "Paris", "runner_up_id": "52280", "runner_up": "Arsenal",
             "winner_goals": 1, "runner_up_goals": 1, "winner_pens": 4.0, "runner_up_pens": 3.0}
    assert facts.result_text("52747", final) == (
        "Winner: beat Arsenal on penalties in the final (1-1, 4-3 on penalties)")
    assert facts.result_text("52280", final) == (
        "Runner-up: lost the final to Paris Saint-Germain on penalties (1-1, 3-4 on penalties)")


def test_team_sheets_use_plain_labels_and_mark_lower_is_better(built):
    ds, res = built
    sheets = facts.build_facts(ds.team_seasons, ds.finals, res)
    team_keys = [k for k in sheets if k != "synthesis"]
    assert len(team_keys) == 10
    sheet = sheets[team_keys[0]]
    assert "Model probability of reaching the final (%)" in sheet
    stats = sheet["All league/group-phase stats"]
    assert "Opponent shots per game (lower is better)" in stats
    assert "Possession (%)" in stats
    json.dumps(sheets, allow_nan=False)  # raises if any NaN slipped into the facts


def test_synthesis_facts_include_context_counts_and_thresholds(built):
    ds, res = built
    synth = facts.build_facts(ds.team_seasons, ds.finals, res)["synthesis"]
    assert synth["Context"]["Finals"] == 6 and synth["Context"]["Finalists"] == 12
    assert len(synth["The 10 finalists"]) == 10
    assert synth["Significance threshold for Holm-adjusted p"] == 0.05
    assert all("number of stats" in row for row in synth["Feature-set comparison (Spearman / AUC)"])
```

Note: runner-up sentences quote the score from that team's side ("1-1, 3-4"), which is the natural reading in a scouting report. The report's finals list stays winner-oriented.

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_facts.py -q`
Expected: FAIL with `ImportError: cannot import name 'facts'`

- [ ] **Step 3: Write `src/ucl/facts.py`**

```python
"""Plain-English fact sheets the local LLM writes from (spec §7)."""
from __future__ import annotations

import math

import pandas as pd

from . import config
from .features import display_value
from .model import ModelResults, feature_list

PCT_SEASON = "Percentile vs that season's teams"
PCT_ALL = "Percentile vs all Champions League teams since 2011-12"
SIGNIFICANCE = 0.05
DIRECTION = {1: "higher helps", -1: "lower helps", 0: "no clear direction"}


def stat_label(feature: str) -> str:
    label = config.FEATURE_META[feature][0]
    return f"{label} (lower is better)" if feature in config.LOWER_IS_BETTER else label


def _name(team_id: str, fallback: str) -> str:
    return config.DISPLAY_NAMES.get(team_id, fallback)


def result_text(team_id: str, final) -> str:
    """How a finalist's final went, phrased from that team's side."""
    wg, rg = int(final["winner_goals"]), int(final["runner_up_goals"])
    winner = _name(final["winner_id"], final["winner"])
    runner_up = _name(final["runner_up_id"], final["runner_up"])
    if pd.notna(final["winner_pens"]):
        wp, rp = int(final["winner_pens"]), int(final["runner_up_pens"])
        if team_id == final["winner_id"]:
            return f"Winner: beat {runner_up} on penalties in the final ({wg}-{rg}, {wp}-{rp} on penalties)"
        return f"Runner-up: lost the final to {winner} on penalties ({rg}-{wg}, {rp}-{wp} on penalties)"
    if team_id == final["winner_id"]:
        return f"Winner: beat {runner_up} {wg}-{rg} in the final"
    return f"Runner-up: lost the final to {winner} {rg}-{wg}"


def _stat_fact(row: pd.Series, feature: str) -> dict:
    return {
        "value": display_value(row, feature),
        PCT_SEASON: int(row[f"pct_season_{feature}"]),
        PCT_ALL: int(row[f"pct_all_{feature}"]),
    }


def team_fact_sheet(row: pd.Series, pred: pd.Series, shap_row: pd.Series, final, features: list[str]) -> dict:
    contributions = sorted(((f, float(shap_row[f"shap_{f}"])) for f in features), key=lambda kv: kv[1], reverse=True)

    def driver(f: str, c: float) -> dict:
        return {"stat": stat_label(f), **_stat_fact(row, f), "SHAP contribution (knockout stages)": round(c, 2)}

    return {
        "Club": row["team_display"],
        "Season": config.season_label(int(row["season"])),
        "Actual result": result_text(row["team_id"], final),
        "Knockout teams that season": int(pred["ko_size"]),
        "Base rate: chance a random knockout team reaches the final (%)": round(100 * float(pred["base_rate"]), 1),
        "Model probability of reaching the final (%)": round(100 * float(pred["p_final"]), 1),
        "Rank by that probability among the season's knockout teams": int(pred["rank_in_season"]),
        "Model's expected knockout stage (0 = out before quarter-finals, 1 = quarter-finals, "
        "2 = semi-finals, 3 = lost final, 4 = won final)": round(float(pred["exp_stage"]), 2),
        "Stats that pushed the prediction up most": [driver(f, c) for f, c in contributions if c > 0][:4],
        "Stats that pushed the prediction down most": [driver(f, c) for f, c in reversed(contributions) if c < 0][:3],
        "All league/group-phase stats": {stat_label(f): _stat_fact(row, f) for f in features},
    }


def synthesis_facts(team_seasons: pd.DataFrame, results: ModelResults) -> dict:
    m = results.metrics
    needed = math.ceil(config.ROBUST_SHARE * m["n_seasons"])
    finalists = results.predictions.loc[results.predictions["is_target"].astype(bool)].sort_values(
        ["season", "ko_stage"], ascending=[True, False]
    )
    return {
        "Context": {
            "Seasons analysed": m["n_seasons"],
            "Finals": m["n_seasons"],
            "Finalists": m["n_finalists"],
            "Group/league-phase team-seasons": int(len(team_seasons)),
            "Knockout team-seasons": m["n_knockout"],
        },
        "Model quality (each season scored by a model trained on the other seasons)": {
            "Mean within-season Spearman correlation, predicted vs actual stage": round(m["spearman_mean"], 2),
            "Share of actual finalists in the model's top 4 of their season (%)": round(100 * m["finalists_in_top4"], 1),
            "AUC for reaching the final": round(m["auc"], 2),
            "Brier score": round(m["brier"], 3),
            "Brier score of the base-rate guess": round(m["brier_base_rate"], 3),
        },
        "Robustness rule": f"robust = the logistic model agrees on the direction in at least {needed} of "
                           f"{m['n_seasons']} seasons; only the top {config.TOP_DRIVERS} stats get a robustness label",
        "What drives deep runs, most important first": [
            {"stat": stat_label(r.feature), "group": r.group,
             "importance (mean |SHAP|, knockout stages)": round(r.importance, 3),
             "direction": DIRECTION[int(r.direction)], "robustness label": r.label or f"not in top {config.TOP_DRIVERS}"}
            for r in results.drivers.head(8).itertuples()
        ],
        "Feature-set comparison (Spearman / AUC)": [
            {"feature set": r.feature_set, "number of stats": int(r.n_features),
             "Spearman": round(r.spearman, 2), "AUC": round(r.auc, 2)}
            for r in results.ablation.itertuples()
        ],
        "The 10 finalists": [
            {"club": r.team_display, "season": config.season_label(int(r.season)),
             "result": "Winner" if int(r.ko_stage) == 4 else "Runner-up",
             "model probability of reaching the final (%)": round(100 * r.p_final, 1),
             "rank among the season's knockout teams": int(r.rank_in_season), "knockout teams": int(r.ko_size)}
            for r in finalists.itertuples()
        ],
        "Significance threshold for Holm-adjusted p": SIGNIFICANCE,
        "Winners vs runners-up across all finals (z-score difference, winner minus runner-up)": [
            {"stat": stat_label(r.feature), "finals where the winner was higher": int(r.higher),
             "finals where the winner was lower": int(r.lower), "ties": int(r.tied),
             "mean difference (z)": round(r.mean_diff, 2), "Holm-adjusted p": round(r.p_holm, 3)}
            for r in results.finals_compare.sort_values("p_holm", kind="stable").itertuples()
        ],
    }


def build_facts(team_seasons: pd.DataFrame, finals: pd.DataFrame, results: ModelResults) -> dict[str, dict]:
    """One sheet per target finalist (keyed '{season}-{team_id}') plus 'synthesis'."""
    features = feature_list(results)
    rows = team_seasons.set_index(["season", "team_id"], drop=False)  # sheets read row["season"]
    preds = results.predictions.set_index(["season", "team_id"])
    shap_rows = results.shap.set_index(["season", "team_id"])
    finals_by_season = finals.set_index("season")
    sheets: dict[str, dict] = {}
    for season, team_id in preds.index[preds["is_target"].astype(bool)]:
        sheets[f"{season}-{team_id}"] = team_fact_sheet(
            rows.loc[(season, team_id)], preds.loc[(season, team_id)], shap_rows.loc[(season, team_id)],
            finals_by_season.loc[season], features,
        )
    sheets["synthesis"] = synthesis_facts(team_seasons, results)
    return sheets
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_facts.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/ucl/facts.py tests/test_facts.py
git commit -m "feat: plain-English fact sheets for the local LLM"
```

### Task 13: Analyst — prompts, retry policy, persistence

**Files:**
- Create: `src/ucl/analyst.py`, `tests/fakes.py`
- Test: `tests/test_retry.py`, `tests/test_analyst.py`

- [ ] **Step 1: Write the fake LLM and the failing tests**

`tests/fakes.py`:

```python
"""Test doubles shared by the analyst tests."""
import json


class FakeLLM:
    """Scripted replies; caches by request like the real client, so re-runs replay."""

    def __init__(self, script):
        self.script, self.requests, self.cache = list(script), [], {}

    def ensure_ready(self):
        return True

    def chat(self, messages):
        key = json.dumps(messages, sort_keys=True)
        if key in self.cache:
            return self.cache[key]
        self.requests.append(messages)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        self.cache[key] = item
        return item
```

`tests/test_retry.py`:

```python
from fakes import FakeLLM

from ucl.analyst import TEAM_HEADINGS, write_narrative
from ucl.llm import ChatResult, LLMError

FACTS = {"Shots per game": 17.4}
GOOD = "## How they got there\nThey averaged 17.4 shots.\n## Would the model have picked them?\nYes.\n## Weak spots\nFew."
UNGROUNDED_1 = GOOD.replace("Few.", "A 47% risk.")
UNGROUNDED_1B = GOOD.replace("Few.", "A 48% risk.")
UNGROUNDED_2 = GOOD.replace("Few.", "A 47% risk and 88 problems.")
INVALID = ChatResult("", "length")
MESSAGES = [{"role": "user", "content": "write"}]


def ok(text):
    return ChatResult(text, "stop")


def test_valid_grounded_answer_takes_one_call():
    n = write_narrative(FakeLLM([ok(GOOD)]), "k", MESSAGES, TEAM_HEADINGS, FACTS)
    assert (n.status, n.calls, n.unsupported) == ("ok", 1, [])


def test_invalid_then_valid_takes_two_distinct_calls():
    llm = FakeLLM([INVALID, ok(GOOD)])
    n = write_narrative(llm, "k", MESSAGES, TEAM_HEADINGS, FACTS)
    assert (n.status, n.calls) == ("ok", 2)
    assert llm.requests[0] != llm.requests[1]
    assert "cut off" in llm.requests[1][-1]["content"]


def test_validity_retry_carries_a_truncated_excerpt():
    llm = FakeLLM([ok("x" * 5000), ok(GOOD)])
    write_narrative(llm, "k", MESSAGES, TEAM_HEADINGS, FACTS)
    assert len(llm.requests[1][-2]["content"]) == 1500


def test_rerun_is_served_entirely_from_cache():
    llm = FakeLLM([INVALID, ok(GOOD)])
    first = write_narrative(llm, "k", MESSAGES, TEAM_HEADINGS, FACTS)
    second = write_narrative(llm, "k", MESSAGES, TEAM_HEADINGS, FACTS)
    assert len(llm.requests) == 2
    assert first.text == second.text


def test_two_invalid_answers_make_the_narrative_unavailable():
    n = write_narrative(FakeLLM([INVALID, INVALID]), "k", MESSAGES, TEAM_HEADINGS, FACTS)
    assert (n.status, n.calls, n.text) == ("unavailable", 2, None)


def test_grounding_retry_fixes_numbers():
    n = write_narrative(FakeLLM([ok(UNGROUNDED_1), ok(GOOD)]), "k", MESSAGES, TEAM_HEADINGS, FACTS)
    assert (n.status, n.calls, n.unsupported) == ("ok", 2, [])


def test_grounding_retry_with_as_many_problems_keeps_the_earlier_text():
    n = write_narrative(FakeLLM([ok(UNGROUNDED_1), ok(UNGROUNDED_1B)]), "k", MESSAGES, TEAM_HEADINGS, FACTS)
    assert (n.text, n.unsupported) == (UNGROUNDED_1, ["47"])


def test_never_more_than_three_calls_and_keep_best():
    llm = FakeLLM([INVALID, ok(UNGROUNDED_1), ok(UNGROUNDED_2)])
    n = write_narrative(llm, "k", MESSAGES, TEAM_HEADINGS, FACTS)
    assert n.calls == 3 and len(llm.requests) == 3
    assert n.text == UNGROUNDED_1 and n.unsupported == ["47"]


def test_invalid_grounding_retry_keeps_the_earlier_text():
    n = write_narrative(FakeLLM([ok(UNGROUNDED_1), INVALID]), "k", MESSAGES, TEAM_HEADINGS, FACTS)
    assert (n.status, n.calls, n.text, n.unsupported) == ("ok", 2, UNGROUNDED_1, ["47"])


def test_timeout_on_grounding_retry_keeps_the_earlier_text():
    n = write_narrative(FakeLLM([ok(UNGROUNDED_1), LLMError("timeout")]), "k", MESSAGES, TEAM_HEADINGS, FACTS)
    assert (n.status, n.text, n.unsupported) == ("ok", UNGROUNDED_1, ["47"])


def test_first_call_failure_is_unavailable():
    n = write_narrative(FakeLLM([LLMError("refused")]), "k", MESSAGES, TEAM_HEADINGS, FACTS)
    assert n.status == "unavailable" and "refused" in n.reason
```

`tests/test_analyst.py`:

```python
from fakes import FakeLLM

from ucl import analyst
from ucl.llm import ChatResult


def team_reply():
    # bold headings instead of '##': accepted, then normalised
    return ChatResult("**How they got there**\nx\n**Would the model have picked them?**\ny\n**Weak spots**\nz", "stop")


def synth_reply():
    return ChatResult("## Why the best teams win\nx\n## Winners vs runners-up\ny", "stop")


def test_run_skipped_and_unavailable_paths(built):
    ds, res = built
    assert analyst.run(ds.team_seasons, ds.finals, res, enabled=False).status == "skipped"

    class Down(FakeLLM):
        def ensure_ready(self):
            return False

    assert analyst.run(ds.team_seasons, ds.finals, res, llm=Down([])).status == "unavailable"


def test_run_writes_eleven_normalised_narratives(built, tmp_path):
    ds, res = built
    llm = FakeLLM([team_reply()] * 10 + [synth_reply()])
    analysis = analyst.run(ds.team_seasons, ds.finals, res, llm=llm, log=lambda _: None)
    assert analysis.status == "ok"
    assert len(analysis.narratives) == 11 and analysis.narratives["synthesis"].status == "ok"
    team = next(n for k, n in analysis.narratives.items() if k != "synthesis")
    assert team.text.startswith("## How they got there\n")
    analyst.save(analysis, tmp_path / "analysis.json")
    loaded = analyst.load(tmp_path / "analysis.json")
    assert loaded.narratives["synthesis"].text == analysis.narratives["synthesis"].text


def test_load_without_a_file_is_skipped(tmp_path):
    assert analyst.load(tmp_path / "missing.json").status == "skipped"
```

Every team gets the same scripted reply text. The FakeLLM cache is keyed by request, and each fact sheet differs, so the 11 requests are distinct and consume the 11 replies in order.

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_retry.py tests/test_analyst.py -q`
Expected: FAIL with `ImportError: cannot import name 'analyst'`

- [ ] **Step 3: Write `src/ucl/analyst.py`**

```python
"""Prompts, the narrative retry policy and persistence for the local-LLM analyst (spec §7)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from . import config
from .facts import build_facts
from .grounding import check_grounding, normalize_headings, strip_think, validate_output
from .llm import LLMError, LMStudio
from .model import ModelResults

TEAM_HEADINGS = ["How they got there", "Would the model have picked them?", "Weak spots"]
SYNTH_HEADINGS = ["Why the best teams win", "Winners vs runners-up"]
RETRY_EXCERPT_CHARS = 1500

SYSTEM_PROMPT = (
    "You are a football performance analyst writing for a curious, data-literate reader. "
    "Use ONLY the facts provided. Quote every number exactly as it appears in the facts, with the same rounding. "
    "Do not introduce players, managers, matches, events or numbers that are not in the facts. "
    "Percentiles compare a team with other Champions League teams: 90 means higher than 90% of them. "
    "For stats marked '(lower is better)', a LOW percentile is good. "
    "SHAP contributions are measured in knockout stages: +0.30 means the stat pushed the model's expected "
    "stage up by 0.30 of a round. Write plain markdown without tables."
)
VALIDITY_FEEDBACK = (
    "Your previous answer could not be used: {reason}. Write the complete answer again, following every "
    "instruction and using exactly the required headings."
)
GROUNDING_FEEDBACK = (
    "These numbers in your answer do not appear in the facts: {numbers}. Rewrite the complete answer with the "
    "same headings, quoting only numbers that appear in the facts (same rounding), or describe those points "
    "without numbers."
)
TEAM_INSTRUCTIONS = (
    "Write a scouting report on this Champions League finalist: 180-250 words of markdown with exactly these "
    "three headings, in this order:\n## How they got there\n## Would the model have picked them?\n## Weak spots\n\n"
    "Under 'Would the model have picked them?', compare the model's probability with the base rate and the rank. "
    "The model never saw this season but was trained on the other seasons, including later ones, so describe "
    "whether it would have picked them, not a forecast made at the time.\n\nFacts (JSON):\n"
)
SYNTH_INSTRUCTIONS = (
    "Write 350-450 words of markdown with exactly these two headings, in this order:\n"
    "## Why the best teams win\n## Winners vs runners-up\n\n"
    "Explain which stats drive deep knockout runs and whether pedigree, results or style matters most, using the "
    "feature-set comparison. Only call a stat 'robust' if its robustness label is robust. Never call a "
    "winners-vs-runners-up difference significant unless its Holm-adjusted p is below the significance "
    "threshold in the facts.\n\nFacts (JSON):\n"
)


@dataclass
class Narrative:
    key: str
    status: str  # "ok" | "unavailable"
    text: str | None = None
    unsupported: list[str] = field(default_factory=list)
    calls: int = 0
    reason: str | None = None


@dataclass
class Analysis:
    status: str  # "ok" | "unavailable" | "skipped"
    model: str
    narratives: dict[str, Narrative] = field(default_factory=dict)
    facts: dict[str, dict] = field(default_factory=dict)


def write_narrative(llm, key: str, messages: list[dict], headings: list[str], facts: dict) -> Narrative:
    """At most 3 calls: initial, one validity retry, one grounding retry (spec §7)."""
    calls = 0
    try:
        answer = llm.chat(messages)
        calls += 1
    except LLMError as exc:
        return Narrative(key, "unavailable", calls=calls, reason=f"LLM call failed: {exc}")
    reason = validate_output(answer.content, answer.finish_reason, headings)
    if reason:
        messages = messages + [
            {"role": "assistant", "content": (answer.content or "")[:RETRY_EXCERPT_CHARS]},
            {"role": "user", "content": VALIDITY_FEEDBACK.format(reason=reason)},
        ]
        try:
            answer = llm.chat(messages)
            calls += 1
        except LLMError as exc:
            return Narrative(key, "unavailable", calls=calls, reason=f"LLM call failed: {exc}")
        reason = validate_output(answer.content, answer.finish_reason, headings)
        if reason:
            return Narrative(key, "unavailable", calls=calls, reason=reason)
    text = normalize_headings(strip_think(answer.content), headings)
    unsupported = check_grounding(text, facts)
    if unsupported:
        retry = messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": GROUNDING_FEEDBACK.format(numbers=", ".join(unsupported))},
        ]
        try:
            second = llm.chat(retry)
            calls += 1
            if validate_output(second.content, second.finish_reason, headings) is None:
                second_text = normalize_headings(strip_think(second.content), headings)
                second_unsupported = check_grounding(second_text, facts)
                if len(second_unsupported) < len(unsupported):
                    text, unsupported = second_text, second_unsupported
        except LLMError:
            pass  # keep the earlier valid text with its flags
    return Narrative(key, "ok", text=text, unsupported=unsupported, calls=calls)


def _messages(instructions: str, sheet: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instructions + json.dumps(sheet, indent=1, ensure_ascii=False)},
    ]


def run(
    team_seasons: pd.DataFrame,
    finals: pd.DataFrame,
    results: ModelResults,
    llm_model: str = config.LLM_MODEL,
    enabled: bool = True,
    llm=None,
    log: Callable[[str], None] = print,
) -> Analysis:
    if not enabled:
        return Analysis("skipped", llm_model)
    llm = llm or LMStudio(model=llm_model)
    if not llm.ensure_ready():
        return Analysis("unavailable", llm_model)
    sheets = build_facts(team_seasons, finals, results)
    narratives: dict[str, Narrative] = {}
    for key, sheet in sheets.items():
        synth = key == "synthesis"
        messages = _messages(SYNTH_INSTRUCTIONS if synth else TEAM_INSTRUCTIONS, sheet)
        narrative = write_narrative(llm, key, messages, SYNTH_HEADINGS if synth else TEAM_HEADINGS, sheet)
        narratives[key] = narrative
        log(f"  {key}: {narrative.status}, {narrative.calls} call(s), {len(narrative.unsupported)} unsupported")
    return Analysis("ok", llm_model, narratives, sheets)


def save(analysis: Analysis, path: Path = config.OUT_DIR / "analysis.json") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "status": analysis.status,
        "model": analysis.model,
        "narratives": {k: asdict(n) for k, n in analysis.narratives.items()},
        "facts": analysis.facts,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def load(path: Path = config.OUT_DIR / "analysis.json") -> Analysis:
    path = Path(path)
    if not path.exists():
        return Analysis("skipped", config.LLM_MODEL)
    data = json.loads(path.read_text())
    narratives = {k: Narrative(**n) for k, n in data["narratives"].items()}
    return Analysis(data["status"], data["model"], narratives, data["facts"])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_retry.py tests/test_analyst.py -q`
Expected: `14 passed`

- [ ] **Step 5: Commit**

```bash
git add src/ucl/analyst.py tests/fakes.py tests/test_retry.py tests/test_analyst.py
git commit -m "feat: analyst prompts, 3-call retry policy and persistence"
```

### Task 14: CLI `analyze` stage with the real local model

**Files:**
- Modify: `src/ucl/cli.py`, `tests/test_cli.py`

- [ ] **Step 1: Update the CLI order test (failing)**

```python
def test_commands_run_in_pipeline_order():
    assert list(cli.COMMANDS)[:4] == ["fetch", "build", "model", "analyze"]
```

Run: `uv run pytest tests/test_cli.py -q`
Expected: FAIL

- [ ] **Step 2: Add `cmd_analyze` (above `COMMANDS`) and register it**

```python
def cmd_analyze(args: argparse.Namespace) -> int:
    from . import analyst, dataset, model

    ds = dataset.load()
    analysis = analyst.run(ds.team_seasons, ds.finals, model.load(), llm_model=args.llm_model,
                           enabled=not args.no_ai)
    analyst.save(analysis)
    flagged = sum(len(n.unsupported) for n in analysis.narratives.values())
    unavailable = sum(n.status != "ok" for n in analysis.narratives.values())
    print(f"analysis {analysis.status} with {args.llm_model}: {len(analysis.narratives)} narratives "
          f"({unavailable} unavailable), {flagged} figure(s) not found in the data")
    return 0
```

```python
COMMANDS = {"fetch": cmd_fetch, "build": cmd_build, "model": cmd_model, "analyze": cmd_analyze}
```

- [ ] **Step 3: Run the whole suite**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 4: Run the analyst against LM Studio**

Run: `uv run ucl analyze`
Expected:
- 11 progress lines (10 finalists plus `synthesis`), ideally each `ok, 1 call(s), 0 unsupported`.
- Then `analysis ok with qwen/qwen3.5-35b-a3b: 11 narratives, …`.
- The model loads in about 10 s if it isn't loaded already. Each call takes roughly 5–20 s with reasoning off.

Read `out/analysis.json`. For each narrative, check:
- the required headings are present
- nothing about players, managers or matches is invented
- `lower is better` stats aren't praised for high values
- flagged numbers are listed in `unsupported`

If several narratives say something false that the facts would have prevented, fix the prompt in `src/ucl/analyst.py`, or the labels in `src/ucl/facts.py`, then run `uv run pytest -q` and re-run. Changed prompts are new cache keys.

- [ ] **Step 5: Confirm re-runs replay from cache**

LM Studio must still be running, because `ensure_ready` checks it first.

```bash
ls data/llm_cache | wc -l
cp out/analysis.json out/analysis.prev.json
uv run ucl analyze
ls data/llm_cache | wc -l
cmp out/analysis.json out/analysis.prev.json && rm out/analysis.prev.json && echo replayed
```

Expected: the same cache-file count before and after, and `replayed`.

- [ ] **Step 6: Tick Chunk 5 in `tasks/todo.md` (`- [ ]` → `- [x]`) and commit**

`git add -A` includes any prompt or label fixes from Step 4.

```bash
git add -A
git commit -m "feat: analyze stage; local Qwen scouting reports with grounding checks"
```

---

## Chunk 6: Charts and the report

Design decisions for this chunk follow the dataviz skill and the artifact page contract:
- **Charts:** HTML/CSS rows rather than SVG, so text stays legible at phone width.
  - Bars are 14px with a 4px rounded data-end and a square baseline end.
  - Dots are 12px with a 2px surface ring.
  - Axes are hairlines, and chart text uses ink tokens.
  - Mark rules are scoped under `.track`, so they never hit legend swatches.
- **Interaction and accessibility:** a legend for two or more series, a keyboard-reachable table twin per chart, focusable rows with `textContent` tooltips, and 4.5:1 contrast for small informational text.
- **Colours (validated 2026-10-01):** factor groups use palette slots 1–3 (blue, orange, aqua), which pass the all-pairs check in both themes; aqua's sub-3:1 light contrast is covered by value labels and table twins. The blue/red diverging pair passes in both modes.
- **Page:** a fragment with `<title>` first, colour tokens for light, dark and `data-theme`, Archivo plus Public Sans from Google Fonts with system fallbacks, a 16px gutter, and no horizontal scroll even with every table twin open.
- **Files:** CSS and JS live in `assets.py`, so `report.py` holds only renderers.

### Task 15: Chart builders

**Files:**
- Create: `src/ucl/charts.py`
- Test: `tests/test_charts.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_charts.py`:

```python
from ucl import charts


def row(label, value, **extra):
    return {"label": label, "value": value, "color": "var(--s1)", "tip": f"{value}\n{label}", **extra}


def test_labels_and_tips_are_escaped():
    html = charts.bar_chart([row('<b>"x"</b>', 1.0, tip='a "quoted" <tip>')])
    assert "<b>" not in html and "&lt;b&gt;" in html
    assert 'data-tip="a &quot;quoted&quot; &lt;tip&gt;"' in html


def test_bar_widths_scale_to_the_largest_value():
    html = charts.bar_chart([row("a", 1.0), row("b", 0.5)])
    assert "--w:100.0%" in html and "--w:50.0%" in html
    assert "diverging" not in html


def test_values_below_the_baseline_diverge():
    html = charts.bar_chart([row("a", 0.6), row("b", 0.4)], baseline=0.5)
    assert 'class="bars diverging"' in html
    assert '<span class="half neg"><span class="bar"' in html


def test_dot_chart_positions_dot_base_rate_and_axis():
    rows = [{"label": "A", "sub": "s", "p": 0.25, "base": 0.125, "value_text": "25%", "tip": "t"}]
    html = charts.dot_chart(rows, extent=0.5)
    assert "--x:50.0%" in html and "--x:25.0%" in html
    assert ">0%<" in html and ">25%<" in html and ">50%<" in html


def test_stage_ladder_marks_reached_rungs_and_expected_stage():
    html = charts.stage_ladder(3, 1.5)
    assert html.count("reached") == 4 and html.count(" here") == 1
    assert "--x:40.0%" in html


def test_legend_mirrors_the_mark_shape():
    html = charts.legend([("Model", "var(--s1)", "dot"), ("Base rate", "var(--muted)", "tick")])
    assert 'class="swatch dot"' in html and 'class="swatch tick"' in html


def test_table_twin_marks_numeric_columns():
    html = charts.table_twin("Show", ["Name", "Value"], [["A", "1.0"]])
    assert '<td class="num">1.0</td>' in html and "<td>A</td>" in html
    assert html.startswith('<details class="twin">')
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_charts.py -q`
Expected: FAIL with `ImportError: cannot import name 'charts'`

- [ ] **Step 3: Write `src/ucl/charts.py`**

```python
"""HTML/CSS chart builders for the report.

Dataviz specs: bars <= 24px with a 4px rounded data-end and a square baseline end, dots >= 8px with a
2px surface ring, hairline axes, text in ink tokens (never the series colour), a legend for two or more
series and a table twin for every chart. Each row is focusable and carries its tooltip in data-tip.
"""
from __future__ import annotations

from html import escape

GROUP_COLORS = {"pedigree": "var(--s1)", "results": "var(--s2)", "style": "var(--s3)"}
STAGES = ["KO", "QF", "SF", "Final", "Won"]


def _attr(text: str) -> str:
    return escape(text, quote=True)


def legend(items: list[tuple[str, str, str]]) -> str:
    """items: (label, css colour, shape), shape 'bar', 'dot' or 'tick' to mirror the marks."""
    keys = "".join(
        f'<span class="key"><span class="swatch {shape}" style="--c:{color}"></span>{escape(label)}</span>'
        for label, color, shape in items
    )
    return f'<div class="legend">{keys}</div>'


def bar_chart(rows: list[dict], *, baseline: float = 0.0, fmt: str = "{:.2f}", caption: str = "") -> str:
    """Horizontal bars from `baseline`; if any value is below it the bars diverge left and right.

    Rows need label, value, color and tip; tag (chip after the label) and note (under the value) are optional.
    """
    extent = max((abs(r["value"] - baseline) for r in rows), default=0.0) or 1.0
    diverging = any(r["value"] < baseline for r in rows)
    html = []
    for r in rows:
        delta = r["value"] - baseline
        bar = f'<span class="bar" style="--w:{abs(delta) / extent * 100:.1f}%;--c:{r["color"]}"></span>'
        if diverging:
            neg, pos = (bar, "") if delta < 0 else ("", bar)
            track = (f'<span class="track split"><span class="half neg">{neg}</span>'
                     f'<span class="half pos">{pos}</span></span>')
        else:
            track = f'<span class="track">{bar}</span>'
        tag = f'<span class="chip">{escape(r["tag"])}</span>' if r.get("tag") else ""
        note = f'<span class="note">{escape(r["note"])}</span>' if r.get("note") else ""
        html.append(
            f'<div class="bar-row" tabindex="0" data-tip="{_attr(r["tip"])}">'
            f'<span class="bar-label">{escape(r["label"])}{tag}</span>{track}'
            f'<span class="bar-value">{escape(fmt.format(r["value"]))}{note}</span></div>'
        )
    cap = f'<p class="chart-caption">{escape(caption)}</p>' if caption else ""
    kind = "bars diverging" if diverging else "bars"
    return f'<div class="{kind}">{"".join(html)}</div>{cap}'


def dot_chart(rows: list[dict], *, extent: float) -> str:
    """One row per item on a 0..extent scale: a base-rate tick and a dot for the value.

    Rows need label, sub, p, base, value_text and tip.
    """
    html = []
    for r in rows:
        x_dot = min(r["p"] / extent, 1.0) * 100
        x_tick = min(r["base"] / extent, 1.0) * 100
        html.append(
            f'<div class="dot-row" tabindex="0" data-tip="{_attr(r["tip"])}">'
            f'<span class="bar-label">{escape(r["label"])}<small>{escape(r["sub"])}</small></span>'
            f'<span class="track dots"><span class="tick" style="--x:{x_tick:.1f}%"></span>'
            f'<span class="dot" style="--x:{x_dot:.1f}%"></span></span>'
            f'<span class="bar-value">{escape(r["value_text"])}</span></div>'
        )
    ticks = "".join(f'<span style="--x:{i * 50}%">{extent * i / 2:.0%}</span>' for i in range(3))
    axis = f'<div class="axis-row" aria-hidden="true"><span></span><span class="axis">{ticks}</span><span></span></div>'
    return f'<div class="dots">{"".join(html)}{axis}</div>'


def stage_ladder(actual: int, expected: float) -> str:
    """Five rungs from 'knockouts' to 'won'; a caret marks the model's expected stage (0-4)."""
    rungs = "".join(
        f'<span class="rung{" reached" if i <= actual else ""}{" here" if i == actual else ""}">{name}</span>'
        for i, name in enumerate(STAGES)
    )
    x = (min(max(expected, 0.0), 4.0) + 0.5) / 5 * 100
    label = f"Reached: {STAGES[actual]}. Model's expected stage: {expected:.2f} of 4."
    return (f'<div class="ladder" role="img" aria-label="{_attr(label)}">{rungs}'
            f'<span class="expect" style="--x:{x:.1f}%"></span></div>')


def table(headers: list[str], rows: list[list[str]], *, numeric_from: int = 1, label: str = "Table") -> str:
    def cell(tag: str, i: int, value) -> str:
        cls = ' class="num"' if i >= numeric_from else ""
        return f"<{tag}{cls}>{escape(str(value))}</{tag}>"

    head = "".join(cell("th", i, h) for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(cell("td", i, v) for i, v in enumerate(row)) + "</tr>" for row in rows)
    return (f'<div class="table-wrap" tabindex="0" role="region" aria-label="{_attr(label)}">'
            f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>")


def table_twin(summary: str, headers: list[str], rows: list[list[str]], *, numeric_from: int = 1) -> str:
    """The table view every chart needs: tooltips enhance, they never gate."""
    return (f'<details class="twin"><summary>{escape(summary)}</summary>'
            f"{table(headers, rows, numeric_from=numeric_from, label=summary)}</details>")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_charts.py -q`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add src/ucl/charts.py tests/test_charts.py
git commit -m "feat: accessible HTML/CSS chart builders"
```

### Task 16: Report renderer and page assets

**Files:**
- Create: `src/ucl/assets.py`, `src/ucl/report.py`
- Modify: `src/ucl/facts.py`: rename `_beats` to `beats` (public; the report shows the same "teams beaten" figures as the fact sheets) and update its call sites
- Test: `tests/test_report.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_report.py` (uses the session-scoped `built` fixture from `tests/conftest.py`):

```python
import re
from datetime import date

from ucl import assets, report
from ucl.analyst import Analysis, Narrative

TEAM_TEXT = ("## How they got there\nThey had **47%** possession.\n"
             "## Would the model have picked them?\nYes.\n## Weak spots\n- Few\n- None")
SYNTH_TEXT = "## Why the best teams win\nResults matter.\n## Winners vs runners-up\nNothing significant at 47."


def analysis_for(results, unsupported=(), synth_unsupported=(), failed_key=None):
    preds = results.predictions
    keys = [f"{int(r.season)}-{r.team_id}" for r in preds[preds["is_target"]].itertuples()]
    narratives = {k: Narrative(k, "ok", TEAM_TEXT, list(unsupported), 1) for k in keys}
    if failed_key is not None:
        narratives[keys[failed_key]] = Narrative(keys[failed_key], "unavailable", reason="timeout")
    narratives["synthesis"] = Narrative("synthesis", "ok", SYNTH_TEXT, list(synth_unsupported), 1)
    return Analysis("ok", "test/model", narratives, {})


def render(built, analysis, **kwargs):
    ds, res = built
    return report.render(ds.team_seasons, ds.finals, res, analysis, generated=date(2026, 10, 1), **kwargs)


def test_page_has_every_section(built):
    page = render(built, analysis_for(built[1]))
    for heading in ["The 10 finalists", "Would the model have picked them?", "What drives deep runs",
                    "Pedigree, results or style?", "Winners vs runners-up", "Method and caveats"]:
        assert heading in page
    assert page.count('<article class="card"') == 10
    assert "Results matter." in page and "Nothing significant at" in page
    assert "Generated 2026-10-01" in page


def test_grounded_narratives_get_the_good_badge(built):
    page = render(built, analysis_for(built[1]))
    assert page.count("All figures found in the data") == 12  # 10 cards + 2 synthesis halves


def test_unsupported_numbers_are_highlighted_and_badged(built):
    page = render(built, analysis_for(built[1], unsupported=["47"]))
    assert '<mark class="unverified" title="Not found in the data">47</mark>' in page
    assert "1 figure not found in the data" in page


def test_synthesis_flags_are_counted_per_half(built):
    page = render(built, analysis_for(built[1], synth_unsupported=["47"]))
    drivers, winners = page.split('id="winners-h"')
    assert "1 figure not found in the data" not in drivers.split('id="drivers-h"')[1]
    assert "1 figure not found in the data" in winners


def test_a_failed_narrative_shows_the_unavailable_badge(built):
    page = render(built, analysis_for(built[1], failed_key=0))
    assert page.count("AI write-up unavailable") == 1


def test_skipped_and_unavailable_runs_show_notices_without_ai_claims(built):
    for status, notice in [("skipped", "AI write-ups were skipped"), ("unavailable", "LM Studio wasn’t reachable")]:
        page = render(built, Analysis(status, "test/model"))
        assert notice in page
        assert "All figures found in the data" not in page
        assert "running locally in LM Studio" not in page
        assert page.count("No AI write-up for this run.") == 10


def test_unavailable_notice_gives_the_reason(built):
    page = render(built, Analysis("unavailable", "test/model", reason="lms not found"))
    assert "(lms not found)" in page


def test_method_reports_intervals_and_chance(built):
    page = render(built, analysis_for(built[1]))
    assert "95% interval" in page and "for a random ranking" in page and "Brier skill" in page


def test_penalty_finals_are_shown(built):
    ds, res = built
    finals = ds.finals.copy()
    finals.loc[finals["season"] == 2026, ["winner_pens", "runner_up_pens"]] = [4, 3]
    page = report.render(ds.team_seasons, finals, res, analysis_for(res), generated=date(2026, 10, 1))
    assert "4–3 on penalties" in page


def test_markdown_subset_is_escaped():
    html = report.md_to_html("## Head\nPara with **bold** and <script>x</script>\n\n- a\n- b")
    assert "<h4>Head</h4>" in html and "<strong>bold</strong>" in html
    assert "<ul><li>a</li><li>b</li></ul>" in html
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_split_sections_keys_by_heading():
    parts = report.split_sections(SYNTH_TEXT)
    assert parts == {"why the best teams win": "Results matter.",
                     "winners vs runners-up": "Nothing significant at 47."}


def test_fragment_is_artifact_ready_and_standalone_is_a_document(built):
    page = render(built, analysis_for(built[1]))
    assert page.startswith("<title>")
    lowered = page.lower()
    assert "<html" not in lowered and "<body" not in lowered and "<!doctype" not in lowered
    doc = report.standalone(page)
    assert doc.startswith("<!doctype html>")
    assert doc.index("<title>") < doc.index("</head>") < doc.index('<main class="page">')


def test_css_covers_both_themes_and_scopes_the_marks():
    css = assets.CSS
    assert "@media (prefers-color-scheme: dark)" in css
    assert ':root[data-theme="dark"]' in css and ':root:not([data-theme="light"])' in css
    # mark rules must not leak onto legend swatches ("swatch dot", "swatch tick", "swatch bar")
    assert not re.search(r"(?m)^\.(dot|tick|bar)\s*\{", css)
    assert ".twin { min-width: 0;" in css


def test_ordinal_suffixes():
    assert [report.ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 95, 100)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "95th", "100th"]
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_report.py -q`
Expected: FAIL with `ImportError: cannot import name 'assets'`

- [ ] **Step 3: Write `src/ucl/assets.py`**

```python
"""The report page's stylesheet and tooltip script (artifact page contract, dataviz specs)."""

CSS = """
/* Layout: one 46rem reading column; each chart is a surface card that reflows by its own width. */
:root {
  --bg: #f3f5f8; --surface: #fcfcfd; --ink: #0d1422; --ink-2: #4a5263; --muted: #656b79;
  --hair: #e1e5eb; --rule: #c3c9d3; --accent: #23408e; --on-accent: #ffffff;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --neg: #e34948; --mark: #fde9ad;
  --good: #0ca30c; --warn: #fab219; --on-good: #ffffff; --on-warn: #3a2a00;
  --font-display: "Archivo", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-body: "Public Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0b0f16; --surface: #131923; --ink: #f2f4f8; --ink-2: #b8bfcb; --muted: #8b919d;
    --hair: #252d3a; --rule: #3a4456; --accent: #a9bbff; --on-accent: #0b0f16;
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --neg: #e66767; --mark: #5c4a12;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #0b0f16; --surface: #131923; --ink: #f2f4f8; --ink-2: #b8bfcb; --muted: #8b919d;
  --hair: #252d3a; --rule: #3a4456; --accent: #a9bbff; --on-accent: #0b0f16;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --neg: #e66767; --mark: #5c4a12;
  color-scheme: dark;
}
[hidden] { display: none !important; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 400 16px/1.6 var(--font-body); }
.page { max-width: 46rem; margin: 0 auto; padding-inline: 16px; padding-block: 48px 72px; display: grid; gap: 64px; }
section, .masthead { display: grid; gap: 16px; min-width: 0; }
h1, h2, h3 { margin: 0; font-family: var(--font-display); line-height: 1.12; text-wrap: balance; }
h1 { font-size: clamp(2.1rem, 1.5rem + 3vw, 3.3rem); font-weight: 800; font-stretch: 112%; letter-spacing: -0.015em; }
h2 { font-size: 1.6rem; font-weight: 750; font-stretch: 112%; }
h3 { font-size: 1.15rem; font-weight: 700; font-stretch: 106%; }
h4 { margin: 8px 0 0; font: 600 0.78rem/1.3 var(--font-body); letter-spacing: 0.07em; text-transform: uppercase; color: var(--ink-2); }
p { margin: 0; max-width: 65ch; }
code { padding: 1px 5px; font: 0.9em ui-monospace, "SF Mono", Menlo, monospace; background: var(--hair); border-radius: 4px; }
.eyebrow { font-size: 0.78rem; font-weight: 600; letter-spacing: 0.09em; text-transform: uppercase; color: var(--ink-2); }
.dek { font-size: 1.12rem; color: var(--ink-2); }
.notice { padding: 12px 16px; color: var(--ink-2); background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; }
.final-list { margin: 0; padding: 0; list-style: none; border-top: 2px solid var(--ink); }
.final-list li { display: grid; grid-template-columns: 5.5rem minmax(0, 1fr) auto; gap: 4px 16px; align-items: baseline; padding-block: 14px; border-bottom: 1px solid var(--hair); }
.final-list .season, .final-list .city { font-size: 0.88rem; color: var(--ink-2); font-variant-numeric: tabular-nums; }
.final-list .tie { min-width: 0; }
.final-list .score { display: inline-block; margin-inline: 10px; font-family: var(--font-display); font-weight: 800; font-stretch: 125%; }
.final-list .pens { display: block; font-size: 0.82rem; color: var(--ink-2); }
.chart { container-type: inline-size; display: grid; gap: 12px; min-width: 0; padding: 16px; background: var(--surface); border: 1px solid var(--hair); border-radius: 10px; }
.chart h4 { margin: 0; }
.pair { display: grid; grid-template-columns: repeat(auto-fit, minmax(17rem, 1fr)); gap: 16px; min-width: 0; }
.bars, .dots { display: grid; gap: 2px; min-width: 0; }
.bar-row, .dot-row, .axis-row { display: grid; grid-template-columns: minmax(0, 14rem) minmax(0, 1fr) 7rem; gap: 4px 12px; align-items: center; }
.bar-row, .dot-row { min-height: 30px; margin-inline: -6px; padding: 3px 6px; border-radius: 6px; }
.bar-row:hover, .dot-row:hover { background: color-mix(in srgb, var(--ink) 5%, transparent); }
.bar-row:focus-visible, .dot-row:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.bar-label { min-width: 0; font-size: 0.86rem; overflow-wrap: anywhere; }
.bar-label small { display: block; font-size: 0.76rem; color: var(--muted); }
.bar-value { font-size: 0.84rem; color: var(--ink-2); text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
.note { display: block; font-size: 0.72rem; color: var(--muted); white-space: normal; }
.chip { display: inline-block; margin-left: 8px; padding: 0 7px; font-size: 0.7rem; line-height: 1.6; color: var(--ink-2); white-space: nowrap; border: 1px solid var(--rule); border-radius: 999px; }
.track { position: relative; display: flex; align-items: center; height: 14px; border-left: 1px solid var(--rule); }
.track .bar { display: block; width: var(--w); height: 14px; background: var(--c); border-radius: 0 4px 4px 0; }
.track.split { display: grid; grid-template-columns: 1fr 1fr; border-left: 0; }
.track .half { display: flex; height: 14px; }
.track .half.neg { justify-content: flex-end; border-right: 1px solid var(--rule); }
.track .half.neg .bar { border-radius: 4px 0 0 4px; }
.track.dots { height: 22px; border-left: 0; }
.track.dots::before { content: ""; position: absolute; inset: 50% 0 auto; border-top: 1px solid var(--rule); }
.track.dots .tick { position: absolute; left: var(--x); top: 4px; height: 14px; border-left: 2px solid var(--muted); transform: translateX(-1px); }
.track.dots .dot { position: absolute; left: var(--x); top: 50%; width: 12px; height: 12px; margin: -6px 0 0 -6px; background: var(--s1); border-radius: 50%; box-shadow: 0 0 0 2px var(--surface); }
.axis { position: relative; height: 18px; font-size: 0.72rem; color: var(--muted); font-variant-numeric: tabular-nums; }
.axis span { position: absolute; left: var(--x); transform: translateX(-50%); }
.axis span:first-child { transform: none; }
.axis span:last-child { transform: translateX(-100%); }
.chart-caption { font-size: 0.78rem; color: var(--muted); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 0.8rem; color: var(--ink-2); }
.key { display: inline-flex; align-items: center; gap: 7px; }
.swatch { display: inline-block; flex: none; background: var(--c); }
.swatch.bar { width: 14px; height: 10px; border-radius: 0 3px 3px 0; }
.swatch.dot { width: 10px; height: 10px; border-radius: 50%; }
.swatch.tick { width: 2px; height: 12px; }
@container (max-width: 34rem) {
  .bar-row, .dot-row, .axis-row { grid-template-columns: minmax(0, 1fr) 6.5rem; }
  .bar-label { grid-column: 1 / -1; }
  .axis-row > span:first-child { display: none; }
}
.twin { min-width: 0; }
.twin summary { width: fit-content; font-size: 0.84rem; color: var(--accent); cursor: pointer; }
.twin summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 4px; }
.table-wrap:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.table-wrap { max-width: 100%; margin-top: 8px; overflow-x: auto; background: var(--surface); border: 1px solid var(--hair); border-radius: 8px; }
table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
th, td { padding: 7px 10px; text-align: left; white-space: nowrap; border-bottom: 1px solid var(--hair); }
th { font-weight: 600; color: var(--ink-2); }
tr:last-child td { border-bottom: 0; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.cards { display: grid; gap: 28px; min-width: 0; }
.card { display: grid; gap: 18px; min-width: 0; padding: 22px; background: var(--surface); border: 1px solid var(--hair); border-radius: 12px; }
.card .chart { padding: 0; background: none; border: 0; }
.card-head { display: grid; gap: 4px; min-width: 0; overflow-wrap: anywhere; }
.card-head h3 span { margin-left: 6px; font: 500 0.95rem/1 var(--font-body); color: var(--ink-2); }
.card-head p { font-size: 0.9rem; color: var(--ink-2); }
.ladder { position: relative; display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 2px; padding-bottom: 12px; }
.rung { padding: 5px 2px; font-size: 0.72rem; text-align: center; color: var(--ink-2); background: var(--hair); }
.rung:first-child { border-radius: 6px 0 0 6px; }
.rung:nth-child(5) { border-radius: 0 6px 6px 0; }
.rung.reached { color: var(--ink); background: color-mix(in srgb, var(--s1) 22%, var(--surface)); }
.rung.here { font-weight: 600; color: var(--on-accent); background: var(--accent); }
.expect { position: absolute; left: var(--x); bottom: 0; width: 0; height: 0; transform: translateX(-50%); border: 6px solid transparent; border-top: 0; border-bottom: 8px solid var(--ink); }
.ladder-note { font-size: 0.82rem; color: var(--ink-2); }
.narrative, .ai-summary { display: grid; gap: 10px; min-width: 0; overflow-wrap: anywhere; }
.ai-summary { padding-top: 16px; border-top: 1px solid var(--hair); }
.narrative ul, .ai-summary ul { margin: 0; padding-left: 1.2em; }
.ai-missing { font-size: 0.84rem; color: var(--ink-2); }
.badge { display: inline-flex; align-items: center; gap: 7px; width: fit-content; font-size: 0.78rem; color: var(--ink-2); }
.badge .icon { display: inline-grid; place-items: center; width: 17px; height: 17px; font-size: 0.68rem; font-weight: 700; color: var(--surface); background: var(--muted); border-radius: 50%; }
.badge.good .icon { color: var(--on-good); background: var(--good); }
.badge.warn .icon { color: var(--on-warn); background: var(--warn); }
mark.unverified { padding-inline: 2px; color: inherit; background: var(--mark); border-radius: 3px; }
.method ul { display: grid; gap: 8px; max-width: 65ch; margin: 0; padding-left: 1.2em; }
footer { font-size: 0.8rem; color: var(--muted); }
.tip { position: fixed; z-index: 10; max-width: 19rem; padding: 8px 11px; font-size: 0.8rem; line-height: 1.45; white-space: pre-line; color: var(--bg); background: var(--ink); border-radius: 7px; pointer-events: none; box-shadow: 0 6px 18px rgb(0 0 0 / 0.2); }
.tip::first-line { font-weight: 700; }
@media (max-width: 560px) { .final-list li { grid-template-columns: 1fr; } }
@media (forced-colors: active) { .track .bar, .track.dots .dot, .track.dots .tick, .swatch, .rung { forced-color-adjust: none; } }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
"""

JS = """
(() => {
  const tip = document.getElementById("tip");
  if (!tip) return;
  const place = (x, y) => {
    const w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = Math.max(8, Math.min(x + 14, window.innerWidth - w - 8)) + "px";
    tip.style.top = (y - h - 14 < 8 ? y + 18 : y - h - 14) + "px";
  };
  const show = (el, x, y) => {
    tip.textContent = el.getAttribute("data-tip");
    tip.hidden = false;
    place(x, y);
  };
  const owner = (e) => (e.target instanceof Element ? e.target.closest("[data-tip]") : null);
  document.addEventListener("pointermove", (e) => {
    const el = owner(e);
    if (el) show(el, e.clientX, e.clientY);
    else tip.hidden = true;
  });
  document.documentElement.addEventListener("pointerleave", () => { tip.hidden = true; });
  document.addEventListener("focusin", (e) => {
    const el = owner(e);
    if (!el) return;
    const r = el.getBoundingClientRect();
    show(el, r.left + Math.min(r.width / 2, 160), r.top);
  });
  document.addEventListener("focusout", () => { tip.hidden = true; });
  window.addEventListener("scroll", () => { tip.hidden = true; }, { passive: true });
})();
"""
```

- [ ] **Step 4: Write `src/ucl/report.py`**

```python
"""Render the report: an artifact-ready page fragment plus a standalone wrapper (spec §8)."""
from __future__ import annotations

import math
import re
from datetime import date
from html import escape

import pandas as pd

from . import charts, config
from .analyst import Analysis, Narrative
from .assets import CSS, JS
from .facts import DIRECTION, SIGNIFICANCE, beats, result_text, stat_label
from .features import display_value
from .model import ModelResults, feature_list

TITLE = "What Makes a Champions League Finalist?"
FONTS_URL = (
    "https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,100..900"
    "&family=Public+Sans:ital,wght@0,400..700;1,400..700&display=swap"
)
GROUP_NAMES = {"pedigree": "Pedigree", "results": "Results", "style": "Style"}
SET_NAMES = {
    "pedigree": "Pedigree only",
    "results": "Results only",
    "style": "Style only",
    "all": "Everything",
    "all minus pedigree": "Everything but pedigree",
    "all minus results": "Everything but results",
    "all minus style": "Everything but style",
}


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _span(bounds, fmt: str = "{:.2f}") -> str:
    """A 95% interval as 'a to b' (no dash, so a negative bound reads clearly)."""
    lo, hi = bounds
    return f"{fmt.format(lo)} to {fmt.format(hi)}"


def _flag_pattern(token: str) -> str:
    return rf"(?<![\w.]){re.escape(token)}(?!\.?\d)"


def _inline(text: str, flagged: list[str]) -> str:
    out = escape(text)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?![*\w])", r"<em>\1</em>", out)
    for token in flagged:
        out = re.sub(_flag_pattern(token),
                     lambda m: f'<mark class="unverified" title="Not found in the data">{m.group(0)}</mark>', out)
    return out


def md_to_html(text: str, flagged=()) -> str:
    """The markdown subset the analyst writes: headings, paragraphs, bullet lists, bold, italic."""
    flagged = list(flagged)
    blocks: list[str] = []
    paragraph: list[str] = []
    items: list[str] = []

    def flush() -> None:
        if paragraph:
            blocks.append(f"<p>{_inline(' '.join(paragraph), flagged)}</p>")
            paragraph.clear()
        if items:
            blocks.append("<ul>" + "".join(f"<li>{_inline(i, flagged)}</li>" for i in items) + "</ul>")
            items.clear()

    for raw in (text or "").splitlines():
        line = raw.strip()
        heading = re.match(r"^#{1,6}\s*(.+?)\s*#*$", line)
        bullet = re.match(r"^(?:[-*•]|\d+[.)])\s+(.+)$", line)
        if not line:
            flush()
        elif heading:
            flush()
            blocks.append(f"<h4>{_inline(heading.group(1), flagged)}</h4>")
        elif bullet:
            if paragraph:
                flush()
            items.append(bullet.group(1))
        else:
            if items:
                flush()
            paragraph.append(line)
    flush()
    return "\n".join(blocks)


def split_sections(text: str) -> dict[str, str]:
    """Markdown body under each heading, keyed by the lower-cased heading text."""
    sections: dict[str, list[str]] = {}
    current = None
    for line in (text or "").splitlines():
        heading = re.match(r"^\s*#{1,6}\s*(.+?)\s*[:.]?\s*#*$", line)
        if heading:
            current = heading.group(1).strip().lower()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def badge(narrative: Narrative | None) -> str:
    if narrative is None or narrative.status != "ok":
        return '<span class="badge"><span class="icon" aria-hidden="true">–</span>AI write-up unavailable</span>'
    n = len(narrative.unsupported)
    if n == 0:
        return ('<span class="badge good"><span class="icon" aria-hidden="true">✓</span>'
                "All figures found in the data</span>")
    noun = "figure" if n == 1 else "figures"
    return (f'<span class="badge warn"><span class="icon" aria-hidden="true">!</span>'
            f"{n} {noun} not found in the data</span>")


def _story(narrative: Narrative | None) -> str:
    if narrative is None:
        return ""
    if narrative.status != "ok" or not narrative.text:
        return badge(None)
    return badge(narrative) + md_to_html(narrative.text, narrative.unsupported)


def _half(synthesis: Narrative | None, text: str | None) -> Narrative | None:
    """One half of the synthesis, carrying only the flags that occur in that half."""
    if synthesis is None or synthesis.status != "ok":
        return synthesis
    if not text:
        return Narrative(synthesis.key, "unavailable", reason="section missing")
    flagged = [t for t in synthesis.unsupported if re.search(_flag_pattern(t), text)]
    return Narrative(synthesis.key, "ok", text, flagged, synthesis.calls)


def _summary_block(narrative: Narrative | None) -> str:
    story = _story(narrative)
    return f'<div class="ai-summary"><p class="eyebrow">AI summary</p>{story}</div>' if story else ""


def _masthead(ai_ran: bool) -> str:
    first = config.season_label(config.SEASONS[0])
    last = config.season_label(config.SEASONS[-1])
    ai = " A local AI model then writes up what it found." if ai_ran else ""
    return (
        '<header class="masthead">'
        f'<p class="eyebrow">UEFA Champions League · {first} to {last}</p>'
        f"<h1>{escape(TITLE)}</h1>"
        '<p class="dek">The ten finalists of 2022–2026, measured against every knockout team since '
        f"{first}. A model learns from group and league-phase play and is tested on seasons it never saw.{ai}</p>"
        "</header>"
    )


def _ai_notice(analysis: Analysis) -> str:
    if analysis.status == "skipped":
        return ('<p class="notice">The AI write-ups were skipped for this run. Run <code>uv run ucl analyze</code> '
                "and then <code>uv run ucl report</code> to add them.</p>")
    if analysis.status == "unavailable":
        why = f" ({escape(analysis.reason)})" if getattr(analysis, "reason", None) else ""
        return (f'<p class="notice">LM Studio wasn’t reachable{why}, so the AI write-ups are missing. Open LM Studio, '
                "then run <code>uv run ucl analyze</code> and <code>uv run ucl report</code>.</p>")
    return ""


def _finals(finals: pd.DataFrame) -> str:
    items = []
    recent = finals.loc[finals["season"].isin(config.TARGET_SEASONS)].sort_values("season", ascending=False)
    for f in recent.itertuples():
        winner = config.DISPLAY_NAMES.get(f.winner_id, f.winner)
        runner_up = config.DISPLAY_NAMES.get(f.runner_up_id, f.runner_up)
        pens = (f'<span class="pens">{int(f.winner_pens)}–{int(f.runner_up_pens)} on penalties</span>'
                if pd.notna(f.winner_pens) else "")
        items.append(
            f'<li><span class="season">{config.season_label(int(f.season))}</span>'
            f'<span class="tie"><strong>{escape(winner)}</strong>'
            f'<span class="score">{int(f.winner_goals)}–{int(f.runner_up_goals)}</span>'
            f"{escape(runner_up)}{pens}</span>"
            f'<span class="city">{escape(str(f.city))}</span></li>'
        )
    return (
        '<section aria-labelledby="finals-h"><h2 id="finals-h">The 10 finalists</h2>'
        "<p>Five finals, winner first on each line. Every side is one team in one season, so Real Madrid "
        "2021-22 and 2023-24 count as two finalists.</p>"
        f'<ol class="final-list">{"".join(items)}</ol></section>'
    )


def _picks(targets: pd.DataFrame, m: dict) -> str:
    extent = max(0.5, math.ceil(float(targets["p_final"].max()) * 10) / 10)
    rows, table_rows = [], []
    for t in targets.itertuples():
        season = config.season_label(int(t.season))
        result = "Winner" if int(t.ko_stage) == 4 else "Runner-up"
        rank = f"#{int(t.rank_in_season)} of {int(t.ko_size)}"
        rows.append({
            "label": t.team_display, "sub": f"{season} · {result} · expected {t.exp_stage:.2f}",
            "p": float(t.p_final), "base": float(t.base_rate), "value_text": f"{t.p_final:.0%} · {rank}",
            "tip": (f"{t.p_final:.1%} chance of reaching the final\n{t.team_display} {season}: {rank} knockout "
                    f"teams. Base rate {t.base_rate:.1%}. Expected stage {t.exp_stage:.2f} of 4."),
        })
        table_rows.append([t.team_display, season, result, f"{t.p_final:.1%}", rank,
                           f"{t.base_rate:.1%}", f"{t.exp_stage:.2f}"])
    picked = int((targets["rank_in_season"] <= 2).sum())
    legend = charts.legend([("Model’s probability of reaching the final", "var(--s1)", "dot"),
                            ("Base rate for a random knockout team", "var(--muted)", "tick")])
    return (
        '<section aria-labelledby="picks-h"><h2 id="picks-h">Would the model have picked them?</h2>'
        f"<p>Each finalist was scored by a model trained on the other {m['n_seasons'] - 1} seasons, using only "
        "how the team played in the group or league phase and its pre-season club coefficient. Training "
        "includes later seasons, so this checks whether the pattern holds across eras. Two teams reach each "
        f"final, so a pick means ranking first or second: {picked} of the 10 finalists did.</p>"
        f'<div class="chart">{legend}{charts.dot_chart(rows, extent=extent)}</div>'
        + charts.table_twin("Show as a table", ["Finalist", "Season", "Result", "P(final)", "Rank", "Base rate",
                                                "Expected stage"], table_rows, numeric_from=3)
        + "</section>"
    )


def _ablation(ablation: pd.DataFrame) -> str:
    def chart(key: str, baseline: float, title: str, caption: str) -> str:
        rows = [{
            "label": SET_NAMES.get(r.feature_set, r.feature_set), "value": float(getattr(r, key)),
            "color": "var(--s1)",
            "tip": (f"{getattr(r, key):.2f} (95% interval {_span((getattr(r, key + '_lo'), getattr(r, key + '_hi')))})"
                    f"\n{SET_NAMES.get(r.feature_set, r.feature_set)}, {int(r.n_features)} stats"),
        } for r in ablation.itertuples()]
        return (f'<div class="chart"><h4>{escape(title)}</h4>'
                f"{charts.bar_chart(rows, baseline=baseline, fmt='{:.2f}', caption=caption)}</div>")

    table_rows = [[SET_NAMES.get(r.feature_set, r.feature_set), int(r.n_features),
                   f"{r.spearman:.2f}", _span((r.spearman_lo, r.spearman_hi)),
                   f"{r.auc:.2f}", _span((r.auc_lo, r.auc_hi))] for r in ablation.itertuples()]
    return (
        "<h3>Pedigree, results or style?</h3>"
        "<p>The same models retrained on subsets of the stats. Dropping a group shows what it adds that the "
        "others don’t already carry. The 95% intervals overlap heavily, so treat the differences between "
        "groups as tentative.</p>"
        '<div class="pair">'
        + chart("spearman", 0.0, "Ranking teams within a season",
                "Spearman correlation between predicted and actual stage. Bars start at 0, a random ranking.")
        + chart("auc", 0.5, "Spotting the finalists", "AUC for reaching the final. Bars start at 0.5, a coin flip.")
        + "</div>"
        + charts.table_twin("Show as a table, with 95% intervals",
                            ["Stats used", "Count", "Spearman", "95% interval", "AUC", "95% interval"], table_rows)
    )


def _drivers(results: ModelResults, summary: Narrative | None, m: dict) -> str:
    rows, table_rows = [], []
    for r in results.drivers.itertuples():
        label, direction = stat_label(r.feature), DIRECTION[int(r.direction)]
        tag = r.label if isinstance(r.label, str) else ""
        rows.append({
            "label": label, "value": float(r.importance), "color": charts.GROUP_COLORS[r.group],
            "tag": f"{tag} · {direction}" if tag else "",
            "tip": (f"{r.importance:.3f} knockout stages on average\n{label} · {GROUP_NAMES[r.group]} · {direction}; "
                    f"on its own ρ = {r.marginal_rho:+.2f}"),
        })
        table_rows.append([int(r.rank), label, GROUP_NAMES[r.group], f"{r.importance:.3f}", direction,
                           f"{r.marginal_rho:+.2f}", tag, int(r.sign_agree_folds)])
    needed = math.ceil(config.ROBUST_SHARE * m["n_seasons"])
    legend = charts.legend([(GROUP_NAMES[g], color, "bar") for g, color in charts.GROUP_COLORS.items()])
    return (
        '<section aria-labelledby="drivers-h"><h2 id="drivers-h">What drives deep runs</h2>'
        "<p>Bar length is how far each stat moved the model’s expected stage on average, in knockout rounds "
        "(mean absolute SHAP value, measured only on seasons the model didn’t train on). The top "
        f"{config.TOP_DRIVERS} carry a label. <strong>Robust</strong>: a simpler logistic model agrees on the "
        f"direction in at least {needed} of {m['n_seasons']} seasons, and the stat points the same way on its own. "
        "<strong>Conditional</strong>: the direction holds only with the other stats held fixed; on its own the "
        "stat points the other way or barely at all. <strong>Model-dependent</strong>: the simpler model "
        "disagrees too often.</p>"
        f'<div class="chart">{legend}{charts.bar_chart(rows, fmt="{:.3f}")}</div>'
        + charts.table_twin("Show as a table", ["Rank", "Stat", "Group", "Importance", "Direction",
                                                "On its own (ρ)", "Label", "Seasons agreeing"],
                            table_rows, numeric_from=3)
        + _ablation(results.ablation)
        + _summary_block(summary)
        + "</section>"
    )


def _card(row: pd.Series, pred: pd.Series, shap_row: pd.Series, final: pd.Series,
          narrative: Narrative | None, ai_ran: bool, features: list[str]) -> str:
    season = config.season_label(int(row["season"]))
    contributions = {f: float(shap_row[f"shap_{f}"]) for f in features}
    top = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:7]
    top.sort(key=lambda kv: kv[1], reverse=True)
    bars = [{
        "label": stat_label(f), "value": c, "color": "var(--s1)" if c >= 0 else "var(--neg)",
        "tip": (f"{c:+.2f} knockout stages\n{stat_label(f)}: {display_value(row, f)}, beat "
                f"{beats(row, f, 'season')}% of that season's teams"),
    } for f, c in top]
    legend = charts.legend([("Pushed the prediction up", "var(--s1)", "bar"),
                            ("Pushed it down", "var(--neg)", "bar")])
    stats = [[stat_label(f), display_value(row, f), beats(row, f, "season"), beats(row, f, "all"),
              f"{contributions[f]:+.2f}"] for f in features]
    expected = float(pred["exp_stage"])
    note = (f"▲ marks the model’s expected stage, {expected:.2f} of 4. It gave a {float(pred['p_final']):.0%} "
            f"chance of reaching the final, #{int(pred['rank_in_season'])} of {int(pred['ko_size'])} knockout teams.")
    if ai_ran:
        story = _story(narrative) or badge(None)
    else:
        story = '<p class="ai-missing">No AI write-up for this run.</p>'
    return (
        f'<article class="card" id="team-{int(row["season"])}-{escape(str(row["team_id"]))}">'
        f'<div class="card-head"><h3>{escape(str(row["team_display"]))}<span>{season}</span></h3>'
        f"<p>{escape(result_text(row['team_id'], final))}</p></div>"
        f'{charts.stage_ladder(int(pred["ko_stage"]), expected)}<p class="ladder-note">{escape(note)}</p>'
        f'<div class="narrative">{story}</div>'
        f'<div class="chart"><h4>What moved the prediction</h4>{legend}{charts.bar_chart(bars, fmt="{:+.2f}")}</div>'
        + charts.table_twin(f"All {len(features)} stats",
                            ["Stat", "Value", "Teams beaten that season (%)", "Teams beaten since 2011-12 (%)",
                             "Moved the prediction"], stats)
        + "</article>"
    )


def _cards(team_seasons, finals, results: ModelResults, analysis: Analysis, targets: pd.DataFrame) -> str:
    features = feature_list(results)
    rows = team_seasons.set_index(["season", "team_id"], drop=False)
    preds = results.predictions.set_index(["season", "team_id"])
    shap_rows = results.shap.set_index(["season", "team_id"])
    finals_by_season = finals.set_index("season")
    ai_ran = analysis.status == "ok"
    cards = []
    for t in targets.itertuples():
        key = (t.season, t.team_id)
        cards.append(_card(rows.loc[key], preds.loc[key], shap_rows.loc[key], finals_by_season.loc[t.season],
                           analysis.narratives.get(f"{int(t.season)}-{t.team_id}"), ai_ran, features))
    ai = ", with the AI scouting report" if ai_ran else ""
    return ('<section aria-labelledby="teams-h"><h2 id="teams-h">The finalists, one by one</h2>'
            "<p>Where each side finished, what the model expected, and the stats that moved its prediction "
            f"most{ai}. “Teams beaten” is the share of teams a side did better than, so higher is always better, "
            "including on the stats where a lower number is better.</p>"
            f'<div class="cards">{"".join(cards)}</div></section>')


def _winners(results: ModelResults, summary: Narrative | None, n_finals: int) -> str:
    fc = results.finals_compare.sort_values("mean_diff", ascending=False, kind="stable")
    rows, table_rows = [], []
    for r in fc.itertuples():
        label = stat_label(r.feature)
        rows.append({
            "label": label, "value": float(r.mean_diff), "color": "var(--s1)",
            "note": f"{int(r.higher)}–{int(r.lower)}–{int(r.tied)} · p {r.p_holm:.2f}",
            "tip": (f"{r.mean_diff:+.2f} standard deviations on average\n{label}: the winner was higher in "
                    f"{int(r.higher)} of {n_finals} finals, lower in {int(r.lower)}. Holm-adjusted p {r.p_holm:.2f}."),
        })
        table_rows.append([label, f"{r.mean_diff:+.2f}", int(r.higher), int(r.lower), int(r.tied),
                           f"{r.p:.3f}", f"{r.p_holm:.3f}",
                           *[f"{getattr(r, f'diff_{s}'):+.2f}" for s in config.TARGET_SEASONS]])
    significant = int((fc["p_holm"] < SIGNIFICANCE).sum())
    verdict = (f"No difference survives a Holm correction at p < {SIGNIFICANCE}." if significant == 0
               else f"{significant} of {len(fc)} differences survive a Holm correction at p < {SIGNIFICANCE}.")
    return (
        '<section aria-labelledby="winners-h"><h2 id="winners-h">Winners vs runners-up</h2>'
        f"<p>For each of the {n_finals} finals since 2011-12: the winner’s group or league-phase stats minus the "
        f"runner-up’s, in within-season standard deviations. Under each value: the finals in which the winner was "
        f"higher–lower–tied, and the Holm-adjusted p across all {len(fc)} stats. {verdict} With so few finals, "
        "read these as hints at most.</p>"
        f'<div class="chart">{charts.bar_chart(rows, fmt="{:+.2f}")}</div>'
        + charts.table_twin("Show as a table, including the five recent finals",
                            ["Stat", "Mean difference", "Winner higher", "Lower", "Tied", "p", "Holm p",
                             *[config.season_label(s) for s in config.TARGET_SEASONS]], table_rows)
        + _summary_block(summary)
        + "</section>"
    )


def _method(team_seasons: pd.DataFrame, finals: pd.DataFrame, results: ModelResults, analysis: Analysis,
            generated: date) -> str:
    m = results.metrics
    ci = m.get("ci", {})
    first = config.season_label(config.SEASONS[0])
    last = config.season_label(config.SEASONS[-1])
    counts = finals["winner_id"].value_counts()
    top_id, top_wins = counts.idxmax(), int(counts.max())
    top_name = config.DISPLAY_NAMES.get(top_id, finals.loc[finals["winner_id"] == top_id, "winner"].iloc[0])
    skill_lo, skill_hi = ci.get("brier_skill", (math.nan, math.nan))
    skill_verdict = ("includes zero, so that edge is inconclusive" if skill_lo <= 0 <= skill_hi
                     else "lies entirely above zero")
    method = [
        f"Data: UEFA’s public match, team-statistics and club-coefficient feeds for {m['n_seasons']} seasons "
        f"({first} to {last}). That covers {len(team_seasons)} group or league-phase team-seasons, of which "
        f"{m['n_knockout']} reached the knockouts.",
        f"Inputs: {len(feature_list(results))} per-game stats from group and league-phase matches only, "
        "standardised within each season, plus the club’s five-year UEFA coefficient from before the season. "
        "Save rate was dropped because the 2011-12 feed lacks saves for too many matches.",
        f"Validation: each season is predicted by models trained on the other {m['n_seasons'] - 1}. The ranking "
        f"correlation averages {m['spearman_mean']:.2f} (95% interval {_span(ci.get('spearman_mean', (0, 0)))}), "
        f"and the AUC for reaching the final is {m['auc']:.2f} ({_span(ci.get('auc', (0, 0)))}). "
        f"{m['finalists_in_top4']:.0%} of finalists were in their season’s predicted top four "
        f"({_span(ci.get('finalists_in_top4', (0, 0)), '{:.0%}')}), against "
        f"{m['finalists_in_top4_chance']:.0%} for a random ranking. Against always guessing the base rate, the "
        f"probabilities have a Brier skill of {m['brier_skill']:.2f}; its interval ({_span((skill_lo, skill_hi))}) "
        f"{skill_verdict}. Intervals come from resampling seasons.",
        "Probabilities of reaching the final are scaled within each season to add up to two, because "
        "exactly two teams get there.",
    ]
    if analysis.status == "ok":
        method.append(
            f"AI write-ups: {analysis.model} running locally in LM Studio, given only the numbers behind this "
            "page. Every figure it writes is checked against those numbers. The check confirms that the figure "
            "appears in the data, not that it is attached to the right stat, and the wording itself is not checked."
        )
    caveats = [
        "Correlation is not causation, and group or league-phase stats depend on the opponents drawn.",
        "Knockout football is high-variance. The 2026 final was decided on penalties.",
        f"{top_name} won {top_wins} of the {len(finals)} finals, so one club’s profile weighs heavily on what "
        "winning looks like.",
        "In 2024-25 and 2025-26 a top-eight league finish skips the play-off, which builds in an advantage "
        "for results.",
        "UEFA publishes no expected-goals data for these seasons, and the shot counts leave out blocked shots.",
        "UEFA’s feed reported possession in seconds and distance in metres for some 2014-16 matches. These were "
        "converted using the feed’s own figures, and partial tracking and placeholder zeros were ignored.",
        "Some stat definitions changed between seasons, so comparisons with all teams since 2011-12 mix them. "
        "The model itself only compares teams within a season.",
        f"{m['n_seasons']} seasons contain only {m['n_finalists']} finalists, so the probabilities are rough.",
        "Models are trained on seasons after the one being scored as well as before it.",
        "UEFA’s APIs are undocumented. Their values are used as published.",
        "The AI text is limited to the numbers and checked against them, but its interpretations are not "
        "causal evidence.",
    ]
    items = "".join(f"<li>{escape(x)}</li>" for x in method)
    risks = "".join(f"<li>{escape(x)}</li>" for x in caveats)
    source = f" and a local model ({escape(analysis.model)})" if analysis.status == "ok" else ""
    return (
        '<section class="method" aria-labelledby="method-h"><h2 id="method-h">Method and caveats</h2>'
        f"<ul>{items}</ul><h3>Caveats</h3><ul>{risks}</ul></section>"
        f"<footer>Generated {generated.isoformat()} from UEFA data{source}.</footer>"
    )


def render(team_seasons: pd.DataFrame, finals: pd.DataFrame, results: ModelResults, analysis: Analysis,
           generated: date | None = None) -> str:
    """The artifact-ready page fragment: <title> first, then styles, content and the tooltip script."""
    m = results.metrics
    preds = results.predictions
    targets = preds.loc[preds["is_target"].astype(bool)].sort_values(["season", "ko_stage"], ascending=[False, False])
    synthesis = analysis.narratives.get("synthesis")
    parts = split_sections(synthesis.text) if synthesis and synthesis.status == "ok" else {}
    body = "".join([
        _masthead(analysis.status == "ok"),
        _ai_notice(analysis),
        _finals(finals),
        _picks(targets, m),
        _drivers(results, _half(synthesis, parts.get("why the best teams win")), m),
        _cards(team_seasons, finals, results, analysis, targets),
        _winners(results, _half(synthesis, parts.get("winners vs runners-up")), len(finals)),
        _method(team_seasons, finals, results, analysis, generated or date.today()),
    ])
    return (
        f"<title>{escape(TITLE)}</title>\n"
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        f'<link rel="stylesheet" href="{escape(FONTS_URL)}">\n'
        f"<style>{CSS}</style>\n"
        f'<main class="page">{body}</main>\n'
        '<div id="tip" class="tip" role="tooltip" hidden></div>\n'
        f"<script>{JS}</script>\n"
    )


def standalone(fragment: str) -> str:
    """Wrap the fragment as a full document for opening out/report.html directly."""
    head, _, body = fragment.partition('<main class="page">')
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        f'{head}</head>\n<body>\n<main class="page">{body}</body>\n</html>\n'
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_report.py -q`
Expected: `14 passed`

- [ ] **Step 6: Tick Chunk 6 in `tasks/todo.md` (`- [ ]` → `- [x]`) and commit**

```bash
git add src/ucl/assets.py src/ucl/report.py tests/test_report.py tasks/todo.md
git commit -m "feat: themed, accessible HTML report with AI write-ups and grounding badges"
```

---

## Chunk 7: End-to-end run, verification, publish

### Task 17: `report` and `all`; end-to-end run; visual verification

**Files:**
- Modify: `src/ucl/cli.py`, `tests/test_cli.py`
- Create: `.claude/launch.json`

- [ ] **Step 1: Update the CLI order test (failing)**

```python
def test_commands_run_in_pipeline_order():
    assert list(cli.COMMANDS) == ["fetch", "build", "model", "analyze", "report"]
```

Run: `uv run pytest tests/test_cli.py -q`
Expected: FAIL

- [ ] **Step 2: Add `cmd_report` (above `COMMANDS`) and register it**

```python
def cmd_report(args: argparse.Namespace) -> int:
    from . import analyst, dataset, model, report

    ds = dataset.load()
    page = report.render(ds.team_seasons, ds.finals, model.load(), analyst.load())
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUT_DIR / "report_page.html").write_text(page)  # artifact-ready fragment
    (config.OUT_DIR / "report.html").write_text(report.standalone(page))
    print(f"wrote {config.OUT_DIR / 'report.html'} and report_page.html")
    return 0
```

```python
COMMANDS = {"fetch": cmd_fetch, "build": cmd_build, "model": cmd_model, "analyze": cmd_analyze,
            "report": cmd_report}
```

- [ ] **Step 3: Run the whole suite**

Run: `uv run pytest -q`
Expected: all pass, 0 failures.

- [ ] **Step 4: End-to-end without the LLM, then with it**

Run: `uv run ucl all --no-ai`
Expected:
- Every stage succeeds. fetch is served from cache and model takes about a minute.
- `out/analysis.json` has `"status": "skipped"`.
- `out/report.html` shows the "AI write-ups were skipped" notice.

Run: `ls out data/processed`
Expected:
- `out`: `ablation.csv analysis.json drivers.csv finals_compare.csv metrics.json predictions.csv report.html report_page.html shap.csv`
- `data/processed`: `dataset.json finals.csv matches.csv missing_stats.csv team_match_stats.csv team_seasons.csv`

Run (LM Studio must be running): `uv run ucl analyze && uv run ucl report`
Expected: the analyst replays from `data/llm_cache`, and the report carries 10 cards plus 2 AI summary blocks with badges.

- [ ] **Step 5: Visual check in the browser pane (done by the controller, not an implementer subagent)**

Create `.claude/launch.json`:

```json
{
  "version": "0.0.1",
  "configurations": [
    {
      "name": "report",
      "runtimeExecutable": "python3",
      "runtimeArgs": ["-m", "http.server", "8765", "--directory", "out"],
      "port": 8765
    }
  ]
}
```

Start it with `preview_start` (name `report`) and navigate to `http://localhost:8765/report.html`. Then:
1. **Desktop, light theme (`resize_window` with `colorScheme: "light"`):** read the page top to bottom. Screenshot every section: masthead and finals, picks chart, drivers and ablation, two team cards, winners vs runners-up, method. Hover one bar and confirm the tooltip shows the value first.
2. **Dark theme:** use `colorScheme: "dark"` and repeat the section screenshots.
3. **Phone:** use the `mobile` preset with `colorScheme: "light"`. Open every table twin with `javascript_tool`, `document.querySelectorAll('details').forEach(d => d.open = true)`, then check `document.documentElement.scrollWidth <= window.innerWidth`. It must be `true`. Screenshot the picks chart and one card.

Fix anything broken: clipped or overlapping labels, unreadable text in either theme, or horizontal scroll. Then do this once:
- Re-run `uv run pytest tests/test_report.py tests/test_charts.py -q` and `uv run ucl report`.
- Reload, and re-measure step 3.
- Reset the window to the `desktop` preset. The pane re-syncs its colour scheme to the app theme when it is reopened.

- [ ] **Step 6: Final data and badge checks**

Run: `uv run python scripts/check_standings.py`
Expected: 10 lines starting with `OK `, exit 0.

Run: `grep -o 'All figures found in the data\|[0-9]* figures* not found in the data\|AI write-up unavailable' out/report_page.html | sort | uniq -c`
Expected: 12 badges in total (10 cards + 2 synthesis halves). Check every non-green badge against that narrative's `unsupported` list in `out/analysis.json`.

- [ ] **Step 7: Commit** (Chunk 7 is ticked in Task 18, after publishing)

```bash
git add -A
git commit -m "feat: report stage and one-command pipeline; end-to-end outputs"
```

### Task 18: README, task review, publish

**Files:**
- Create: `README.md`
- Modify: `tasks/todo.md`

- [ ] **Step 1: Write `README.md`**

````markdown
# UCL Finalists

Why do the best Champions League teams win? This project gathers 15 seasons of UEFA data and trains
leave-one-season-out models of how far knockout teams go. A local LLM (Qwen 3.5 in LM Studio) then
writes scouting reports on the 10 finalists of 2022–2026.

## Run it

```bash
uv sync
uv run ucl all            # fetch → build → model → analyze → report
uv run ucl all --no-ai    # everything except the LLM write-ups
open out/report.html
```

Each stage also runs alone: `uv run ucl fetch | build | model | analyze | report`. Use
`--llm-model KEY` to pick another LM Studio model.

## Stages and outputs

| Stage | Writes |
|-|-|
| fetch | `data/raw/`: UEFA matches, per-match team stats and club coefficients (cached, resumable) |
| build | `data/processed/`: 488 group/league-phase team-seasons with features, labels and validation |
| model | `out/predictions.csv`, `shap.csv`, `drivers.csv`, `ablation.csv`, `finals_compare.csv`, `metrics.json` |
| analyze | `out/analysis.json`: scouting reports, with every number checked against the data |
| report | `out/report.html` (open locally) and `out/report_page.html` (artifact-ready fragment) |

## Method

- Training uses only teams that reached the knockouts. The target is how far they went: out before the
  quarter-finals, quarter-final, semi-final, runner-up or winner.
- Inputs are group/league-phase stats only (16 features in three groups: pedigree, results and style),
  z-scored within each season.
- Gradient boosting (with exact TreeSHAP) and logistic regression are each validated leave-one-season-out.
  A driver counts as robust when both models agree on its direction. P(final) is scaled within each season
  to sum to two.
- The local LLM sees only a fact sheet of these numbers. Its output is checked for structure and for numbers
  that are not in the facts.

Design spec: `docs/specs/2026-10-01-ucl-finalists-design.md`
Implementation plan: `docs/plans/2026-10-01-ucl-finalists.md`

## Caveats

These are associations, not causes. Knockouts are noisy: the data holds 15 finals and 30 finalists. There is
no xG. UEFA's APIs are undocumented. The report lists the full set.
````

- [ ] **Step 2: Fill in `tasks/todo.md`**

Tick Chunk 7. Under `## Review`, record:
- the real metrics (Spearman, AUC, Brier vs base rate, finalists in top 4)
- the robust top drivers
- how many finalists ranked 1st or 2nd in their season
- any dropped features or imputations
- the AI narratives' badge counts
- anything left undone

- [ ] **Step 3: Final test run and commit**

Run: `uv run pytest -q`
Expected: all pass.

```bash
git add README.md tasks/todo.md
git commit -m "docs: README and task review"
```

- [ ] **Step 4: Publish (the controller does this, not the pipeline)**

Publish `out/report_page.html` with the Artifact tool. The first publish needs `icon: "chart"` and the description "Fifteen seasons of UEFA data, leave-one-season-out models and local-AI scouting reports on the 10 Champions League finalists of 2022–2026."

Expected: a private claude.ai artifact URL. Share it with the user, together with the local path `out/report.html`.
