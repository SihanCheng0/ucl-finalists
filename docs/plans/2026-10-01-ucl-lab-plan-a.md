# UCL Lab, Plan A (pipeline and history): Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local web app, started with `uv run ucl web`. It runs the existing pipeline live and lets the user explore and compare every Champions League team-season from 2011-12 to 2025-26.

**Architecture:**
- **Shared stages:** the CLI's stage bodies move into `src/ucl/stages.py`. The CLI and a threaded `PipelineRunner` both use them. The runner reports through an `EventBus`, which streams server-sent events.
- **Backend:** a FastAPI app serves JSON from an immutable `DataStore` snapshot of the pipeline's outputs, and also serves the built single-page app.
- **Data shaping:** pure `queries` functions.
- **Frontend:** Vite, React and TypeScript, with three screens: Pipeline, Explore and Compare.

**Tech stack:**
- **Python:** 3.13 with uv; pandas; FastAPI 0.142 on Starlette 1.7; uvicorn 0.54; pytest, with `httpx2` for FastAPI's `TestClient`.
- **Frontend:** Node 24, React 19.3, Vite 8, TypeScript 7, Vitest 5. All of these were probed on this machine before the plan was written:
  - **Vite 8:** `defineConfig` from `vite` rejects a `test` key, so don't add one. Vitest's defaults are fine.
  - **Starlette 1.7:** it warns when `TestClient` runs on `httpx`; it wants `httpx2`.

**Spec:** `docs/specs/2026-10-01-ucl-lab-dashboard-design.md`, revision 3. This plan is §10's Plan A. The live season, players and the Squad screen are Plan B. In Plan A, every live-related response field takes its "no live data" value (`live: false`, `live_available: false`, `stale: false`, `fetched_at: null`), so Plan B only adds data.

**Project:** `~/Projects/ucl-finalists`, git `main`, clean, 290 tests passing.
- Run Python through `uv run`, and the frontend with `npm` inside `web/`.
- End every commit message with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. The `git commit` lines below show only the subject.

**House style:**
- Lines up to 120 characters, `from __future__ import annotations`, and plain-English docstrings.
- Comments are sparse and say *why*.
- Tests never touch the network, LM Studio, `data/` or `out/`: use fakes, `monkeypatch` and `tmp_path`.

---

## File map

| Path | Status | Responsibility |
|-|-|-|
| `src/ucl/stages.py` | new | The stage functions, shared by the CLI and the web runner. They return results and never print |
| `src/ucl/cli.py` | modified | `cmd_*` call `stages.*` and keep their output byte-identical. Dispatches `ucl web`. Help epilog |
| `src/ucl/uefa.py` | modified | `on_request` hook (cache or network, once per resource) |
| `src/ucl/dataset.py` | modified | `fetch_all(progress=)`, called once per season |
| `src/ucl/model.py` | modified | `run(progress=)`: reports main LOSO folds, then ablation sets |
| `src/ucl/config.py` | modified | `STAT_SECTIONS`, `TEAM_SEARCH_ALIASES`, `WEB_PORT`, `WEB_DIR` |
| `src/ucl/web/__init__.py` | new | Package marker |
| `src/ucl/web/jsonsafe.py` | new | NaN and NaT become `null`, numpy values become Python |
| `src/ucl/web/store.py` | new | `Snapshot` and `DataStore`: reload from disk, atomic swap, stale-narrative keys |
| `src/ucl/web/events.py` | new | `Event`, `Subscription`, `EventBus`: boot-scoped ids, replay buffer, SSE frames |
| `src/ucl/web/pipeline.py` | new | `PipelineRunner`: one run at a time on a daemon thread; statuses, counters, events |
| `src/ucl/web/services.py` | new | The real stage registry, prerequisite checks, and `Services` wiring |
| `src/ucl/web/queries.py` | new | Pure shaping functions: meta, search, profile, compare, summary |
| `src/ucl/web/app.py` | new | `create_app`: routes, error format, SSE, static SPA |
| `src/ucl/web/server.py` | new | The `ucl web` process: dist check, bind to 127.0.0.1, open the browser, shutdown |
| `tests/test_cli_characterization.py` and `tests/golden/cli/` | new | The CLI's recorded output |
| `tests/test_stages.py`, `tests/test_web_*.py` | new | Unit and API tests |
| `tests/test_uefa.py`, `tests/test_dataset.py`, `tests/test_model.py` | modified | Tests for the new hooks, appended |
| `pyproject.toml`, `uv.lock` | modified | fastapi and uvicorn; httpx2 for tests |
| `.gitignore` | modified | `web/node_modules/` and `web/dist/` |
| `web/` | new | The SPA. `src/lib` holds pure helpers tested with Vitest; components and screens live under `src/` |

---

## Chunk 1: Safety net, hooks and shared stages

### Task 1: Record the CLI's current output (characterization tests)

These tests pin today's exact CLI output before Task 5 moves the stage bodies into `stages.py`. They pass immediately by design. Their job is to fail if the refactor changes a single character.

**Files:**
- Create: `tests/test_cli_characterization.py`
- Create: `tests/golden/cli/*.txt` (written by the first run, with `UPDATE_GOLDEN=1`)

- [ ] **Step 1: Write the characterization tests**

```python
"""The CLI's exact output, recorded before stages.py took over the stage bodies (spec §4.1).

Run once with UPDATE_GOLDEN=1 to record tests/golden/cli/*.txt from the current code. Every later run must
reproduce them byte for byte. model_ok and all_ok hold real scikit-learn/shap numbers from the synthetic `built`
fixture, so a dependency upgrade can change them: re-record those two deliberately when that happens."""
import dataclasses
import os
from pathlib import Path

import pandas as pd
import pytest

from ucl import analyst, cli, config, dataset, labels, model, report, uefa
from ucl.analyst import Analysis, Narrative

GOLDEN = Path(__file__).parent / "golden" / "cli"


def check(name: str, code: int, out: str, err: str) -> None:
    text = f"exit {code}\n--- stdout\n{out}--- stderr\n{err}"
    path = GOLDEN / f"{name}.txt"
    if os.environ.get("UPDATE_GOLDEN"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    assert path.exists(), f"no golden file {path.name}: run once with UPDATE_GOLDEN=1"
    assert text == path.read_text()


def run(capsys, *argv):
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


class FetchClient:
    """What fetch_all asks of UefaClient: 4 matches a season (3 in the first phase), coefficients and stats."""

    def __init__(self, fail=()):
        self.fail = set(fail)

    def matches(self, season):
        return [{"match_id": f"{season}-{i}", "depth": 0 if i < 3 else 5} for i in range(4)]

    def coefficients(self, season):
        return []

    def team_match_stats_many(self, match_ids):
        ids = list(match_ids)
        stats = {m: (None if m.endswith("-2") else [{}]) for m in ids if m not in self.fail}
        return stats, {m: "RuntimeError: giving up after 5 attempts" for m in ids if m in self.fail}


def patch_fetch(monkeypatch, fail=()):
    monkeypatch.setattr(config, "SEASONS", [2012, 2013])
    monkeypatch.setattr(labels, "match_rows", lambda raw, season: pd.DataFrame(raw, columns=["match_id", "depth"]))
    monkeypatch.setattr(uefa, "UefaClient", lambda: FetchClient(fail))


def patch_build(monkeypatch, ds, error=None, on_validate=False):
    calls = []

    def build(client, seasons):
        calls.append("build")
        if error and not on_validate:
            raise dataset.ValidationError(error)
        return ds

    def validate(built):
        calls.append("validate")
        if error and on_validate:
            raise dataset.ValidationError(error)

    monkeypatch.setattr(uefa, "UefaClient", lambda: FetchClient())
    monkeypatch.setattr(dataset, "build", build)
    monkeypatch.setattr(dataset, "validate", validate)
    monkeypatch.setattr(dataset, "save", lambda built: calls.append("save"))
    return calls


def with_real_notes(ds):
    """A copy with the notes a real build prints, leaving the shared fixtures untouched."""
    return dataclasses.replace(ds, notes={"dropped_features": ["save_pct"], "imputed_values": 0, "coef_imputed": 11,
                                          "team_matches_without_stats": 4})


def patch_model(monkeypatch, built):
    ds, results = built
    monkeypatch.setattr(dataset, "load", lambda: ds)
    monkeypatch.setattr(model, "run", lambda *args: results)
    monkeypatch.setattr(model, "save", lambda r: None)
    monkeypatch.setattr(model, "load", lambda: results)


def patch_analyze(monkeypatch, analysis):
    calls = []

    def run_llm(*args, **kwargs):
        calls.append({k: kwargs[k] for k in ("llm_model", "enabled")})
        log = kwargs.get("log", print)  # analyst.run's own default, which the CLI relies on today
        for key, n in analysis.narratives.items():
            log(f"  {key}: {n.status}, {n.calls} call(s), {len(n.unsupported)} unsupported")
        return analysis

    monkeypatch.setattr(analyst, "run", run_llm)
    monkeypatch.setattr(analyst, "save", lambda a: calls.append("save"))
    monkeypatch.setattr(analyst, "load", lambda: analysis)
    return calls


def patch_report(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(report, "render", lambda *args: "<p>page</p>")
    monkeypatch.setattr(report, "standalone", lambda page: f"<html>{page}</html>")


def mixed_analysis():
    return Analysis("ok", "m/key", {
        "2026-1": Narrative("2026-1", "ok", text="t", unsupported=["7.5"], calls=2),
        "2026-2": Narrative("2026-2", "unavailable", calls=3, reason="empty answer"),
    })


def test_fetch_ok(monkeypatch, capsys):
    patch_fetch(monkeypatch)
    check("fetch_ok", *run(capsys, "fetch"))


def test_fetch_some_failed(monkeypatch, capsys):
    patch_fetch(monkeypatch, fail={"2012-0"})
    check("fetch_some_failed", *run(capsys, "fetch"))


def test_fetch_stops_when_a_whole_season_fails(monkeypatch, capsys):
    patch_fetch(monkeypatch, fail={"2012-0", "2012-1", "2012-2"})
    check("fetch_stops", *run(capsys, "fetch"))


def test_build_ok(monkeypatch, capsys, synthetic_ds):
    calls = patch_build(monkeypatch, with_real_notes(synthetic_ds))
    check("build_ok", *run(capsys, "build"))
    assert calls == ["build", "validate", "save"]


def test_build_fails_validation_while_building(monkeypatch, capsys, synthetic_ds):
    calls = patch_build(monkeypatch, synthetic_ds, error="2016 Roma: distance_km_pg = 3.2 is outside 90-140")
    check("build_invalid", *run(capsys, "build"))
    assert "save" not in calls


def test_build_fails_validation_after_building(monkeypatch, capsys, synthetic_ds):
    calls = patch_build(monkeypatch, synthetic_ds, error="finals: 2026 expected 52747", on_validate=True)
    check("build_invalid_on_validate", *run(capsys, "build"))
    assert calls == ["build", "validate"]


def test_model_ok(monkeypatch, capsys, built):
    patch_model(monkeypatch, built)
    check("model_ok", *run(capsys, "model"))


def test_analyze_ok_with_a_flagged_and_an_unavailable_narrative(monkeypatch, capsys, built):
    patch_model(monkeypatch, built)
    calls = patch_analyze(monkeypatch, mixed_analysis())
    check("analyze_ok", *run(capsys, "analyze", "--llm-model", "m/key"))
    assert calls == [{"llm_model": "m/key", "enabled": True}, "save"]


def test_analyze_unavailable(monkeypatch, capsys, built):
    patch_model(monkeypatch, built)
    patch_analyze(monkeypatch, Analysis("unavailable", "m/key", reason="server did not start: lms not found"))
    check("analyze_unavailable", *run(capsys, "analyze", "--llm-model", "m/key"))


def test_analyze_skipped_with_no_ai(monkeypatch, capsys, built):
    patch_model(monkeypatch, built)
    calls = patch_analyze(monkeypatch, Analysis("skipped", "m/key"))
    check("analyze_skipped", *run(capsys, "analyze", "--no-ai", "--llm-model", "m/key"))
    assert calls == [{"llm_model": "m/key", "enabled": False}, "save"]


def test_report_ok(monkeypatch, capsys, tmp_path, built):
    patch_model(monkeypatch, built)
    patch_analyze(monkeypatch, mixed_analysis())
    patch_report(monkeypatch, tmp_path)
    code, out, err = run(capsys, "report")
    check("report_ok", code, out.replace(str(tmp_path), "<TMP>"), err)
    assert (tmp_path / "out" / "report_page.html").read_text() == "<p>page</p>"
    assert (tmp_path / "out" / "report.html").read_text() == "<html><p>page</p></html>"


def test_all_ok(monkeypatch, capsys, tmp_path, built):
    patch_fetch(monkeypatch)
    patch_build(monkeypatch, with_real_notes(built[0]))
    patch_model(monkeypatch, built)
    patch_analyze(monkeypatch, mixed_analysis())
    patch_report(monkeypatch, tmp_path)
    code, out, err = run(capsys, "all", "--llm-model", "m/key")
    check("all_ok", code, out.replace(str(tmp_path), "<TMP>"), err)


def test_all_stops_at_the_first_failing_stage(monkeypatch, capsys, built):
    patch_fetch(monkeypatch)
    patch_build(monkeypatch, built[0], error="2016 Roma: distance_km_pg = 3.2 is outside 90-140")
    later = []
    monkeypatch.setattr(model, "run", lambda *args: later.append("model"))
    check("all_stops_at_build", *run(capsys, "all"))
    assert later == []


def test_build_lets_a_failed_stats_request_raise(monkeypatch, synthetic_ds):
    patch_build(monkeypatch, synthetic_ds)

    def failing(client, seasons):
        raise RuntimeError("3 match-stat requests failed (e.g. 2012-0: down); run `uv run ucl fetch` again first")

    monkeypatch.setattr(dataset, "build", failing)
    with pytest.raises(RuntimeError, match="run `uv run ucl fetch` again first"):
        cli.main(["build"])
```

- [ ] **Step 2: Record the golden files from the current code**

Run: `UPDATE_GOLDEN=1 uv run pytest tests/test_cli_characterization.py -q`
Expected: `14 passed`, and `tests/golden/cli/` holds 13 `.txt` files.

- [ ] **Step 3: Check that the recordings are sensible, then replay them**

Run: `cat tests/golden/cli/fetch_some_failed.txt tests/golden/cli/analyze_ok.txt`
Expected:
- `fetch_some_failed.txt` holds `exit 1`, both season lines, `  e.g. match 2012-0: RuntimeError: giving up after 5 attempts`, and on stderr `1 match-stat requests failed; run \`uv run ucl fetch\` again to resume.`
- `analyze_ok.txt` starts with `  2026-1: ok, 2 call(s), 1 unsupported` and `  2026-2: unavailable, 3 call(s), 0 unsupported`, then `analysis ok with m/key: 2 narratives (1 unavailable), 1 figure(s) not found in the data`.

Run: `uv run pytest tests/test_cli_characterization.py -q`
Expected: `14 passed` (no `UPDATE_GOLDEN`).

- [ ] **Step 4: Commit**

```bash
git add tests/test_cli_characterization.py tests/golden/cli
git commit -m "test: record the CLI's exact output before the stages refactor"
```

### Task 2: Request hook on the UEFA client

**Files:**
- Modify: `src/ucl/uefa.py`, the `UefaClient.__init__` and `_cached` methods
- Test: `tests/test_uefa.py`, appended

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_uefa.py`:

```python
def hooked_client(tmp_path, responses, hook):
    fetch = FakeFetch(responses)
    return UefaClient(cache_dir=tmp_path, fetch=fetch, sleep=lambda _: None, on_request=hook)


def test_on_request_reports_each_resource_once_with_its_source(tmp_path):
    seen = []
    client = hooked_client(tmp_path, {config.MATCHES_URL.format(season=2026): [[{"id": "1"}]]},
                           lambda *args: seen.append(args))
    client.matches(2026)
    client.matches(2026)
    assert seen == [("matches", "matches/2026", "network"), ("matches", "matches/2026", "cache")]


def test_on_request_counts_a_missing_resource_as_fetched(tmp_path):
    seen = []
    client = hooked_client(tmp_path, {config.MATCH_STATS_URL.format(match_id="9"): [http_error(404)]},
                           lambda *args: seen.append(args))
    assert client.team_match_stats("9") is None
    assert client.team_match_stats("9") is None
    assert seen == [("stats", "stats/9", "network"), ("stats", "stats/9", "cache")]


def test_a_hook_that_raises_changes_nothing(tmp_path):
    def broken(*args):
        raise ValueError("bug in the hook")

    client = hooked_client(tmp_path, {config.MATCHES_URL.format(season=2026): [[{"id": "1"}]]}, broken)
    assert client.matches(2026) == [{"id": "1"}]
    assert client.matches(2026) == [{"id": "1"}]


def test_a_retried_request_is_reported_once(tmp_path):
    seen = []
    url = config.MATCHES_URL.format(season=2026)
    client = hooked_client(tmp_path, {url: [http_error(503), [{"id": "1"}]]}, lambda *args: seen.append(args))
    assert client.matches(2026) == [{"id": "1"}]
    assert seen == [("matches", "matches/2026", "network")]


def test_a_failed_request_is_not_reported(tmp_path):
    seen = []
    client = hooked_client(tmp_path, {config.MATCHES_URL.format(season=2026): [http_error(403)]},
                           lambda *args: seen.append(args))
    with pytest.raises(PermanentHTTPError):
        client.matches(2026)
    assert seen == []
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `uv run pytest tests/test_uefa.py -q -k "on_request or hook or failed_request_is_not"`
Expected: FAIL with `TypeError: UefaClient.__init__() got an unexpected keyword argument 'on_request'`.

- [ ] **Step 3: Implement the hook**

In `src/ucl/uefa.py`, add the parameter to `UefaClient.__init__` and store it:

```python
    def __init__(
        self,
        cache_dir: Path = config.RAW_DIR,
        max_workers: int = 4,
        fetch: Fetch = http_get_json,
        sleep: Callable[[float], None] = time.sleep,
        on_request: Callable[[str, str, str], None] | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.max_workers = max_workers
        self._fetch = fetch
        self._sleep = sleep
        self._on_request = on_request
```

In `_cached`, set `source` in each branch and report it just before returning. The body becomes:

```python
    def _cached(self, key: str, url: str, missing_ok: bool = False, check: Check | None = None) -> Any:
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"corrupt cache file {path}: delete it and run `uv run ucl fetch` again") from exc
            source = "cache"
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
            source = "network"
        self._notify(key, source)
        return None if data == MISSING_MARKER else data

    def _notify(self, key: str, source: str) -> None:
        """Tell the on_request hook about one resolved resource (once, however many attempts it took). It runs on
        team_match_stats_many's worker threads, so the hook must be thread-safe; a broken hook never breaks a fetch."""
        if self._on_request is None:
            return
        try:
            self._on_request(key.split("/", 1)[0], key, source)
        except Exception:  # noqa: BLE001 - the hook is a progress display, not part of the data path
            pass
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_uefa.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/ucl/uefa.py tests/test_uefa.py
git commit -m "feat: on_request hook reports each UEFA resource as cache or network"
```

### Task 3: Season progress from `fetch_all`

**Files:**
- Modify: `src/ucl/dataset.py`, the `fetch_all` function
- Test: `tests/test_dataset.py`, appended (it already has `FakeClient`)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dataset.py`:

```python
class DownClient(FakeClient):
    """Every first-phase stats request fails."""

    def team_match_stats_many(self, match_ids):
        return {}, {m: "RuntimeError: down" for m in match_ids}


def test_fetch_all_reports_progress_once_per_season():
    events, lines = [], []
    failed = dataset.fetch_all(FakeClient(possession=55), [2012, 2013], log=lines.append, progress=events.append)
    assert failed == {}
    assert events == [{"season": s, "matches": 2, "phase_matches": 1, "missing": 0, "failed": 0}
                      for s in (2012, 2013)]
    assert lines == [f"{s}: 2 matches, 1 phase matches, stats missing 0, failed 0" for s in (2012, 2013)]


def test_fetch_all_reports_a_season_that_stops_the_fetch():
    events, lines = [], []
    failed = dataset.fetch_all(DownClient(possession=55), [2012, 2013], log=lines.append, progress=events.append)
    assert failed == {"m1": "RuntimeError: down"}
    assert events == [{"season": 2012, "matches": 2, "phase_matches": 1, "missing": 0, "failed": 1}]
    assert lines == ["2012: 2 matches, 1 phase matches, stats missing 0, failed 1",
                     "  e.g. match m1: RuntimeError: down",
                     "  every request failed for this season; stopping. Fix the cause above, then run "
                     "`uv run ucl fetch` again."]
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `uv run pytest tests/test_dataset.py -q -k "fetch_all"`
Expected: FAIL with `TypeError: fetch_all() got an unexpected keyword argument 'progress'`.

- [ ] **Step 3: Add the `progress` parameter**

In `src/ucl/dataset.py`, replace `fetch_all` with:

```python
def fetch_all(client: UefaClient, seasons: list[int] = config.SEASONS,
              log: Callable[[str], None] = print,
              progress: Callable[[dict], None] | None = None) -> dict[str, str]:
    """Fill the raw cache. Returns {match id: reason} for stats that failed after retries. `progress` gets one
    {season, matches, phase_matches, missing, failed} per season, including a season that stops the fetch."""
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
        if progress is not None:
            progress({"season": season, "matches": len(matches), "phase_matches": len(phase_ids),
                      "missing": missing, "failed": len(failed)})
        failed_all.update(failed)
        if phase_ids and len(failed) == len(phase_ids):
            log("  every request failed for this season; stopping. Fix the cause above, then run `uv run ucl fetch` again.")
            break
    return failed_all
```

- [ ] **Step 4: Run the tests, including the characterization tests**

Run: `uv run pytest tests/test_dataset.py tests/test_cli_characterization.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/ucl/dataset.py tests/test_dataset.py
git commit -m "feat: fetch_all reports progress once per season"
```

### Task 4: Fold and ablation progress from `model.run`

**Files:**
- Modify: `src/ucl/model.py`, the `loso`, `ablation` and `run` functions
- Test: `tests/test_model.py`, appended

- [ ] **Step 1: Write the failing test**

Append to `tests/test_model.py`, which already imports `make_synthetic_dataset` and `model`:

```python
def test_run_reports_each_main_fold_then_each_ablation_set():
    ds = make_synthetic_dataset(seasons=(2021, 2022, 2023))
    events = []
    model.run(ds.team_seasons, ds.finals, ds.features, progress=events.append)
    folds = [e for e in events if "fold" in e]
    sets = [e for e in events if "ablation_set" in e]
    assert folds == [{"fold": k, "of": 3} for k in (1, 2, 3)]
    assert sets == [{"ablation_set": name, "k": k, "of": 7}
                    for k, name in enumerate(model.feature_sets(ds.features), 1)]
    assert events == folds + sets  # the loso calls inside ablation never report as folds
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/test_model.py -q -k reports_each_main_fold`
Expected: FAIL with `TypeError: run() got an unexpected keyword argument 'progress'`.

- [ ] **Step 3: Thread the progress callbacks through**

In `src/ucl/model.py`, add `from typing import Callable` to the imports. Then make these changes.

`loso` gains `on_fold`. Change its signature and loop header, and add the call at the end of the loop body (after the `if with_shap:` block):

```python
def loso(ko: pd.DataFrame, features: list[str], with_shap: bool = True,
         on_fold: Callable[[int, int], None] | None = None) -> LosoOutput:
    """Each season predicted by models trained on the complete rows of the other seasons. `on_fold(k, n)` is
    called after each of the n folds."""
    cols = [f"z_{f}" for f in features]
    predictions, shap_frames, coefs = [], [], []
    seasons = sorted(ko["season"].unique())
    for k, season in enumerate(seasons, 1):
        # ... the existing body of the loop, unchanged ...
        if on_fold is not None:
            on_fold(k, len(seasons))
```

`ablation` gains `progress`:

```python
def ablation(ko: pd.DataFrame, features: list[str], progress: Callable[[dict], None] | None = None) -> pd.DataFrame:
    rows = []
    sets = feature_sets(features)
    for k, (name, fs) in enumerate(sets.items(), 1):
        pred = loso(ko, fs, with_shap=False).predictions
        metrics = evaluate(pred)
        ci = bootstrap_ci(pred)  # as for the headline, so the "all" row's interval is exactly the headline's
        rows.append({"feature_set": name, "n_features": len(fs),
                     "spearman": metrics["spearman_mean"], "spearman_lo": ci["spearman_mean"][0],
                     "spearman_hi": ci["spearman_mean"][1],
                     "auc": metrics["auc"], "auc_lo": ci["auc"][0], "auc_hi": ci["auc"][1]})
        if progress is not None:
            progress({"ablation_set": name, "k": k, "of": len(sets)})
    return pd.DataFrame(rows)
```

`run` gains `progress`:

```python
def run(team_seasons: pd.DataFrame, finals: pd.DataFrame, features: list[str],
        progress: Callable[[dict], None] | None = None) -> ModelResults:
    """`progress` gets {"fold": k, "of": n} for each main LOSO fold, then {"ablation_set", "k", "of"} per set."""
    ko = knockout_population(team_seasons)
    on_fold = None if progress is None else (lambda k, n: progress({"fold": k, "of": n}))
    main = loso(ko, features, on_fold=on_fold)
    display = team_seasons[["season", "team_id", "team_display", "is_target", "complete"]]
    return ModelResults(
        predictions=main.predictions.merge(display, on=["season", "team_id"], how="left"),
        shap=main.shap,
        drivers=drivers(ko, main.shap, main.fold_coefs, features),
        ablation=ablation(ko, features, progress=progress),
        metrics={**evaluate(main.predictions), "ci": bootstrap_ci(main.predictions)},
        finals_compare=finals_compare(team_seasons, finals, features),
    )
```

- [ ] **Step 4: Run the model tests and the characterization tests**

Run: `uv run pytest tests/test_model.py tests/test_cli_characterization.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/ucl/model.py tests/test_model.py
git commit -m "feat: model.run reports LOSO folds and ablation sets"
```

### Task 5: Shared stage functions, and the CLI on top of them

**Files:**
- Create: `src/ucl/stages.py`
- Modify: `src/ucl/cli.py`, the five `cmd_*` functions
- Test: `tests/test_stages.py`. The characterization tests from Task 1 must still pass.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stages.py`:

```python
from types import SimpleNamespace

import pytest

from ucl import analyst, config, dataset, model, report, stages
from ucl.analyst import Analysis, Narrative

DS = SimpleNamespace(team_seasons="ts", finals="finals", features=["x"])


def test_fetch_passes_progress_only_when_given(monkeypatch):
    calls = []
    monkeypatch.setattr(dataset, "fetch_all", lambda client, seasons, **kwargs: calls.append(kwargs) or {"9": "down"})
    assert stages.fetch("client").failed == {"9": "down"}
    stages.fetch("client", progress=print, log=print)
    assert calls == [{"log": print}, {"log": print, "progress": print}]


def test_build_validates_before_saving(monkeypatch):
    saved = []
    monkeypatch.setattr(dataset, "build", lambda client, seasons: "ds")
    monkeypatch.setattr(dataset, "save", saved.append)

    def invalid(ds):
        raise dataset.ValidationError("2016 Roma: distance_km_pg = 3.2 is outside 90-140")

    monkeypatch.setattr(dataset, "validate", invalid)
    with pytest.raises(dataset.ValidationError):
        stages.build("client")
    assert saved == []
    monkeypatch.setattr(dataset, "validate", lambda ds: None)
    assert stages.build("client") == "ds" and saved == ["ds"]


def test_model_calls_run_positionally_and_passes_progress_only_when_given(monkeypatch):
    calls, saved = [], []
    monkeypatch.setattr(model, "run", lambda *args, **kwargs: calls.append((args, kwargs)) or "results")
    monkeypatch.setattr(model, "save", saved.append)
    assert stages.model(DS) == "results"
    stages.model(DS, progress=print)
    assert calls == [(("ts", "finals", ["x"]), {}), (("ts", "finals", ["x"]), {"progress": print})]
    assert saved == ["results", "results"]


def analysis_with(narratives, facts):
    return Analysis("ok", "m", dict(narratives), dict(facts))


def test_analyze_without_merge_saves_every_status_as_the_cli_always_has(monkeypatch):
    for status in ("ok", "unavailable", "skipped"):
        saved = []
        monkeypatch.setattr(analyst, "run", lambda *a, **k: Analysis(status, "m"))
        monkeypatch.setattr(analyst, "save", saved.append)
        result = stages.analyze(DS, "results", "m", True)
        assert saved == [result.analysis] and result.unavailable == []


def test_analyze_with_merge_never_saves_a_run_the_llm_could_not_do(monkeypatch):
    saved = []
    monkeypatch.setattr(analyst, "run", lambda *a, **k: Analysis("unavailable", "m", reason="server down"))
    monkeypatch.setattr(analyst, "save", saved.append)
    result = stages.analyze(DS, "results", "m", True, merge=True)
    assert saved == [] and result.analysis.reason == "server down"


def test_analyze_with_merge_keeps_old_text_and_its_facts_for_narratives_it_could_not_write(monkeypatch):
    old = analysis_with({"a": Narrative("a", "ok", text="old a"), "b": Narrative("b", "ok", text="old b")},
                        {"a": {"x": 1}, "b": {"x": 2}})
    new = analysis_with({"a": Narrative("a", "unavailable", reason="empty answer"),
                         "b": Narrative("b", "ok", text="new b"),
                         "c": Narrative("c", "unavailable", reason="empty answer")},
                        {"a": {"x": 10}, "b": {"x": 20}, "c": {"x": 30}})
    saved = []
    monkeypatch.setattr(analyst, "run", lambda *a, **k: new)
    monkeypatch.setattr(analyst, "load", lambda: old)
    monkeypatch.setattr(analyst, "save", saved.append)
    result = stages.analyze(DS, "results", "m", True, merge=True)
    assert result.unavailable == ["a", "c"]
    merged = saved[0]
    assert (merged.narratives["a"].text, merged.facts["a"]) == ("old a", {"x": 1})  # flagged stale later
    assert (merged.narratives["b"].text, merged.facts["b"]) == ("new b", {"x": 20})
    assert merged.narratives["c"].status == "unavailable"  # nothing older to keep


def test_analyze_with_merge_saves_the_new_run_when_the_old_file_is_unreadable(monkeypatch):
    new = analysis_with({"a": Narrative("a", "unavailable", reason="empty answer")}, {"a": {"x": 10}})
    saved = []

    def unreadable():
        raise ValueError("Expecting value: line 1 column 1 (char 0)")

    monkeypatch.setattr(analyst, "run", lambda *a, **k: new)
    monkeypatch.setattr(analyst, "load", unreadable)
    monkeypatch.setattr(analyst, "save", saved.append)
    assert stages.analyze(DS, "results", "m", True, merge=True).unavailable == ["a"]
    assert saved == [new]


def test_analyze_passes_log_through(monkeypatch):
    seen = []
    monkeypatch.setattr(analyst, "run", lambda *a, **k: seen.append(k) or Analysis("skipped", "m"))
    monkeypatch.setattr(analyst, "save", lambda a: None)
    stages.analyze(DS, "results", "m/key", False, log=print)
    assert seen == [{"llm_model": "m/key", "enabled": False, "log": print}]


def test_report_writes_both_pages_into_the_out_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(report, "render", lambda *args: "<p>page</p>")
    monkeypatch.setattr(report, "standalone", lambda page: f"<html>{page}</html>")
    path = stages.report(DS, "results", "analysis")
    assert path == tmp_path / "out" / "report.html"
    assert path.read_text() == "<html><p>page</p></html>"
    assert (tmp_path / "out" / "report_page.html").read_text() == "<p>page</p>"
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `uv run pytest tests/test_stages.py -q`
Expected: FAIL with `ImportError: cannot import name 'stages' from 'ucl'`.

Two return types differ from spec §4.2 on purpose. `build` returns the `Dataset` itself, which already carries the build notes, instead of a separate `BuildResult`. `analyze` returns `AnalyzeResult(analysis, unavailable)`, so the web runner can name the narratives it could not write. Chunk 3 relies on both.

- [ ] **Step 3: Write `src/ucl/stages.py`**

```python
"""The pipeline's stages as plain functions, shared by the CLI and the web runner (spec §4.2).

Each stage returns its result and never prints; messages go through `log`. Module functions are called with the
positional signatures the CLI has always used, and new keyword arguments only when given, so tests that patch
those functions keep working. Modules are imported inside each stage, as in cli.py, so `ucl fetch` doesn't load
scikit-learn."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

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
    """Build, validate and save the dataset. Raises dataset.ValidationError before anything is saved; the
    returned Dataset carries the build notes."""
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
```

- [ ] **Step 4: Run the stage tests**

Run: `uv run pytest tests/test_stages.py -q`
Expected: `9 passed`.

- [ ] **Step 5: Move the CLI onto the stages**

In `src/ucl/cli.py`, replace each of the five command functions in place with its version below. `_with_ci` (between `cmd_build` and `cmd_model`), `COMMANDS` and `main` stay as they are. Each function keeps its local imports, so `monkeypatch.setattr(uefa, "UefaClient", …)` and the other patches still apply.

```python
def cmd_fetch(args: argparse.Namespace) -> int:
    from . import stages
    from .uefa import UefaClient

    failed = stages.fetch(UefaClient()).failed
    if failed:
        print(f"{len(failed)} match-stat requests failed; run `uv run ucl fetch` again to resume.", file=sys.stderr)
        return 1
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    from . import dataset, stages
    from .uefa import UefaClient

    try:
        ds = stages.build(UefaClient())
    except dataset.ValidationError as exc:
        print(f"dataset validation failed:\n{exc}", file=sys.stderr)
        return 2
    ts, notes = ds.team_seasons, ds.notes
    print(f"built {len(ts)} team-seasons ({int(ts['in_ko'].sum())} knockout), {len(ds.features)} features")
    if notes["dropped_features"]:
        print("dropped (coverage < 90%):", ", ".join(notes["dropped_features"]))
    print(f"imputed values: {notes['imputed_values']}, coefficients imputed: {notes['coef_imputed']}, "
          f"team-matches without stats: {notes['team_matches_without_stats']}")
    return 0


def cmd_model(args: argparse.Namespace) -> int:
    from . import dataset, stages

    results = stages.model(dataset.load())
    m = results.metrics
    print(f"Spearman {_with_ci(m, 'spearman_mean', '.2f')} | AUC {_with_ci(m, 'auc', '.2f')} | "
          f"Brier skill {_with_ci(m, 'brier_skill', '.2f')} "
          f"(Brier {m['brier']:.3f}, base rate {m['brier_base_rate']:.3f})")
    print(f"finalists in top 4: {_with_ci(m, 'finalists_in_top4', '.0%')}, chance {m['finalists_in_top4_chance']:.0%} "
          "(brackets: 95% intervals from resampling seasons)")
    print(results.drivers.head(config.TOP_DRIVERS)[
        ["feature", "importance", "direction", "marginal_rho", "sign_agree_folds", "label"]
    ].to_string(index=False, float_format="{:.3f}".format))
    print(results.ablation.to_string(index=False, float_format="{:.2f}".format))
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from . import dataset, model, stages

    analysis = stages.analyze(dataset.load(), model.load(), args.llm_model, not args.no_ai).analysis
    flagged = sum(len(n.unsupported) for n in analysis.narratives.values())
    unavailable = sum(n.status != "ok" for n in analysis.narratives.values())
    print(f"analysis {analysis.status} with {args.llm_model}: {len(analysis.narratives)} narratives "
          f"({unavailable} unavailable), {flagged} figure(s) not found in the data")
    if analysis.status == "unavailable":
        print(f"LM Studio not ready: {analysis.reason or 'unknown reason'}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from . import analyst, dataset, model, stages

    path = stages.report(dataset.load(), model.load(), analyst.load())
    print(f"wrote {path} and report_page.html")
    return 0
```

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -q`
Expected: `321 passed`, including all 14 in `tests/test_cli_characterization.py` without `UPDATE_GOLDEN`. That proves the CLI output didn't change.

- [ ] **Step 7: Commit**

```bash
git add src/ucl/stages.py src/ucl/cli.py tests/test_stages.py
git commit -m "refactor: stage bodies move into stages.py, shared by the CLI and the web runner"
```

---

## Chunk 2: Web core (JSON safety, store, event bus)

### Task 6: Web dependencies and JSON-safe values

**Files:**
- Modify: `pyproject.toml` and `uv.lock` (through `uv add`)
- Create: `src/ucl/web/__init__.py`, `src/ucl/web/jsonsafe.py`
- Test: `tests/test_web_jsonsafe.py`

- [ ] **Step 1: Add the dependencies**

Run: `uv add "fastapi>=0.142" "uvicorn>=0.54" && uv add --dev "httpx2>=2.13"`
Expected: `pyproject.toml` lists `fastapi` and `uvicorn` under `dependencies` and `httpx2` under the `dev` group, and `uv.lock` is updated.

Run: `uv run python -W error -c "from fastapi.testclient import TestClient; import uvicorn; print('ok')"`
Expected: `ok`. A `StarletteDeprecationWarning` here means `httpx2` is missing from the environment.

- [ ] **Step 2: Write the failing tests**

Create `src/ucl/web/__init__.py` containing only `"""UCL Lab: the local dashboard (spec 2026-10-01-ucl-lab-dashboard-design.md)."""`.

Create `tests/test_web_jsonsafe.py`:

```python
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ucl.web.jsonsafe import to_jsonable


def test_missing_and_infinite_values_become_null():
    assert to_jsonable([float("nan"), np.nan, pd.NA, pd.NaT, None, float("inf")]) == [None] * 6


def test_numpy_scalars_and_arrays_become_python():
    out = to_jsonable({"i": np.int64(3), "f": np.float32(0.5), "b": np.bool_(True), "a": np.array([1, 2])})
    assert out == {"i": 3, "f": 0.5, "b": True, "a": [1, 2]}
    assert (type(out["i"]), type(out["f"]), type(out["b"])) == (int, float, bool)


def test_nested_containers_string_keys_and_timestamps():
    when = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
    out = to_jsonable({7889: ({"x": np.nan},), "t": pd.Timestamp(when), "d": when})
    assert out == {"7889": [{"x": None}], "t": "2026-10-01T12:00:00+00:00", "d": "2026-10-01T12:00:00+00:00"}


def test_plain_values_pass_through():
    value = {"s": "Inter", "n": 2, "x": 0.25, "ok": False}
    assert to_jsonable(value) == value
```

Run: `uv run pytest tests/test_web_jsonsafe.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ucl.web.jsonsafe'`.

- [ ] **Step 3: Implement**

Create `src/ucl/web/jsonsafe.py`:

```python
"""Values from pandas and numpy made safe for JSON: NaN, NaT and pd.NA become null, numpy scalars become Python."""
from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd


def to_jsonable(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:  # before the datetime check: NaT is a datetime
        return None
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, (bool, np.bool_)):  # before int: bool is an int
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_web_jsonsafe.py -q`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/ucl/web/__init__.py src/ucl/web/jsonsafe.py tests/test_web_jsonsafe.py
git commit -m "feat: web dependencies and JSON-safe values"
```

### Task 7: DataStore, an immutable snapshot of the outputs

**Files:**
- Create: `src/ucl/web/store.py`
- Test: `tests/test_web_store.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_web_store.py`:

```python
import pytest

from ucl import analyst, dataset, model
from ucl.analyst import Analysis, Narrative
from ucl.facts import build_facts
from ucl.web.store import DataStore


@pytest.fixture
def outputs(tmp_path, built):
    ds, results = built
    processed, out = tmp_path / "processed", tmp_path / "out"
    dataset.save(ds, processed)
    model.save(results, out)
    return processed, out


def save_analysis(processed, out, tamper=False):
    """An analysis written from the saved outputs, as `ucl analyze` writes it; returns one team key."""
    ds, results = dataset.load(processed), model.load(out)
    facts = build_facts(ds.team_seasons, ds.finals, results)
    key = next(k for k in facts if k != "synthesis")
    if tamper:
        facts[key] = {"Club": "numbers from an earlier run"}
    narratives = {k: Narrative(k, "ok", text="## Heading\nText.") for k in facts}
    analyst.save(Analysis("ok", "m", narratives, facts), out / "analysis.json")
    return key


def test_a_complete_set_of_outputs_is_ready_with_timestamps(outputs):
    processed, out = outputs
    save_analysis(processed, out)
    snap = DataStore(processed, out).snapshot
    assert snap.ready and snap.errors == {}
    assert snap.analysis.status == "ok"
    assert snap.built_at and snap.modelled_at and snap.analysed_at
    assert snap.stale_keys == frozenset()


def test_missing_outputs_are_not_ready_and_say_which_stage_to_run(tmp_path):
    snap = DataStore(tmp_path / "processed", tmp_path / "out").snapshot
    assert not snap.ready
    assert snap.errors == {"dataset": "processed data missing: run build", "results": "model outputs missing: run model"}
    assert snap.analysis.status == "skipped"  # no analysis.json is normal, not an error
    assert snap.built_at is None and snap.stale_keys == frozenset()


def test_a_corrupt_file_means_not_ready_rather_than_a_crash(outputs):
    processed, out = outputs
    (out / "metrics.json").write_text("{not json")
    snap = DataStore(processed, out).snapshot
    assert not snap.ready and snap.results is None
    assert snap.errors["results"].startswith("model outputs could not be read: JSONDecodeError")


def test_reload_swaps_in_a_new_snapshot_and_leaves_the_old_one_alone(outputs):
    processed, out = outputs
    store = DataStore(processed, out)
    before = store.snapshot
    save_analysis(processed, out)
    after = store.reload("analysis")
    assert store.snapshot is after and after is not before
    assert before.analysis.status == "skipped" and after.analysis.status == "ok"
    assert after.dataset is before.dataset  # parts not reloaded are carried over, not re-read


def test_a_fixed_file_clears_its_error_on_reload(outputs):
    processed, out = outputs
    good = (out / "metrics.json").read_text()
    (out / "metrics.json").write_text("{not json")
    store = DataStore(processed, out)
    (out / "metrics.json").write_text(good)
    assert store.reload("results").errors == {} and store.snapshot.ready


def test_narratives_written_for_other_numbers_are_flagged_stale(outputs):
    processed, out = outputs
    key = save_analysis(processed, out, tamper=True)
    assert DataStore(processed, out).snapshot.stale_keys == frozenset({key})
```

Run: `uv run pytest tests/test_web_store.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ucl.web.store'`.

- [ ] **Step 2: Implement**

Create `src/ucl/web/store.py`:

```python
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
MISSING = {"dataset": "processed data missing: run build", "results": "model outputs missing: run model",
           "analysis": "analysis missing: run analyze"}


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
                    errors.pop(part, None)
                except Exception as exc:  # noqa: BLE001 - a missing or corrupt output means "not ready", not a crash
                    changes[part] = None
                    errors[part] = MISSING[part] if isinstance(exc, FileNotFoundError) else (
                        f"{NAMES[part]} could not be read: {type(exc).__name__}: {exc}")
                changes[stamp] = _mtime(marker)
            snapshot = replace(self.snapshot, **changes, errors=errors)
            self.snapshot = replace(snapshot, stale_keys=stale_narratives(snapshot))
            return self.snapshot


def _mtime(path: Path) -> datetime | None:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc) if path.exists() else None


def _normalised(value):
    return json.loads(json.dumps(to_jsonable(value), sort_keys=True, default=str))


def stale_narratives(snapshot: Snapshot) -> frozenset[str]:
    """Narrative keys whose stored facts differ from the facts built from today's data and model (spec §4.2).
    When the two can't even be lined up (a dataset newer than the model), every narrative is stale."""
    analysis = snapshot.analysis
    if not snapshot.ready or analysis is None or not analysis.narratives:
        return frozenset()
    from ..facts import build_facts

    try:
        current = _normalised(build_facts(snapshot.dataset.team_seasons, snapshot.dataset.finals,
                                          snapshot.results))
    except Exception:  # noqa: BLE001
        return frozenset(analysis.narratives)
    stored = _normalised(analysis.facts)
    return frozenset(key for key in analysis.narratives if stored.get(key) != current.get(key))
```

- [ ] **Step 3: Run the tests, then check against the real outputs**

Run: `uv run pytest tests/test_web_store.py -q`
Expected: `6 passed`.

Run: `uv run python -c "from ucl.web.store import DataStore; s = DataStore().snapshot; print(s.ready, s.errors, sorted(s.stale_keys))"`
Expected: `True {} []`. The real `analysis.json` was written from today's numbers, so nothing is stale. Anything else is a bug in `_normalised` or the comparison: fix it before going on.

- [ ] **Step 4: Commit**

```bash
git add src/ucl/web/store.py tests/test_web_store.py
git commit -m "feat: DataStore holds the outputs as one swappable snapshot"
```

### Task 8: EventBus, with boot-scoped ids and SSE frames

**Files:**
- Create: `src/ucl/web/events.py`
- Test: `tests/test_web_events.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_web_events.py`:

```python
import threading

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


def test_payloads_are_made_json_safe_and_types_are_checked():
    bus = EventBus(boot="b1")
    assert bus.publish("progress", {"counters": {"x": float("nan")}}).data == {"counters": {"x": None}}
    with pytest.raises(ValueError):
        bus.publish("error", {})  # EventSource reserves "error" for connection errors


def test_a_new_subscriber_replays_the_buffer_or_resumes_after_its_last_id():
    bus = EventBus(boot="b1", buffer=3)
    for i in range(5):
        bus.publish("log", {"text": str(i)})
    assert drain(bus.subscribe()) == ["b1-3", "b1-4", "b1-5"]  # the buffer keeps the last 3
    assert drain(bus.subscribe("b1-4")) == ["b1-5"]
    assert drain(bus.subscribe("a0-9")) == ["b1-3", "b1-4", "b1-5"]  # an id from an earlier server: replay all
    assert drain(bus.subscribe("garbage")) == ["b1-3", "b1-4", "b1-5"]


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
    assert list(stream) == []
    assert bus.subscribers == 0
    late = bus.stream(bus.subscribe(), heartbeat=5)
    assert [frame[:9] for frame in late] == ["id: b1-1\n"]
```

Run: `uv run pytest tests/test_web_events.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ucl.web.events'`.

- [ ] **Step 2: Implement**

Create `src/ucl/web/events.py`:

```python
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

    def publish(self, type: str, data: dict, run_id: str | None = None, stage: str | None = None) -> Event:
        if type not in TYPES:
            raise ValueError(f"unknown event type {type!r}")
        with self._lock:  # ids, the buffer and every queue advance together, so every client sees one order
            self._n += 1
            event = Event(f"{self.boot}-{self._n}", run_id, self._clock(), type, stage, to_jsonable(data))
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
    return int(n) if id_boot == boot and n.isdigit() else None
```

- [ ] **Step 3: Run the tests**

Run: `uv run pytest tests/test_web_events.py -q`
Expected: `9 passed`.

- [ ] **Step 4: Commit**

```bash
git add src/ucl/web/events.py tests/test_web_events.py
git commit -m "feat: EventBus with boot-scoped ids, replay and SSE frames"
```

---

## Chunk 3: Pipeline runner and the real stages

### Task 9: PipelineRunner, one run at a time

**Files:**
- Create: `src/ucl/web/pipeline.py`
- Test: `tests/test_web_pipeline.py`

The runner is generic. Stages are injected as `name -> callable(ctx) -> StageOutcome`, so these tests use fakes; Task 10 wires the real stages.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_web_pipeline.py`:

```python
import threading

import pytest

from ucl.web.events import EventBus
from ucl.web.pipeline import ORDER, BadRun, PipelineRunner, RunInProgress, StageOutcome


def events_of(bus, type=None):
    subscription = bus.subscribe()
    events = []
    while (event := subscription.get(0)) is not None:
        if type is None or event.type == type:
            events.append(event)
    bus.unsubscribe(subscription)
    return events


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
    registry = {name: recorder(calls, name) for name in ORDER}
    registry.update(overrides or {})
    bus = EventBus(boot="b")
    return PipelineRunner(registry, bus, **kwargs), bus, calls


def test_the_default_run_is_the_five_core_stages_in_order():
    runner, bus, calls = make()
    run_id = runner.start()
    assert runner.wait(5)
    assert run_id == "b-r1" and calls == ORDER
    assert set(statuses(runner).values()) == {"done"}
    assert [e.data["status"] for e in events_of(bus, "done")] == ["done"]


def test_requested_stages_run_in_canonical_order_once_each():
    runner, bus, calls = make()
    runner.start(["report", "fetch", "fetch"])
    assert runner.wait(5)
    assert calls == ["fetch", "report"]
    assert statuses(runner) == {"fetch": "done", "build": "idle", "model": "idle", "analyze": "idle",
                                "report": "done"}


def test_bad_requests_are_refused_before_anything_runs():
    def missing(names):
        return "model needs data/processed/dataset.json: run build first" if names == ["model"] else None

    runner, bus, calls = make(missing_inputs=missing)
    with pytest.raises(BadRun, match="unknown stage"):
        runner.start(["fetch", "train"])
    with pytest.raises(BadRun, match="run build first"):
        runner.start(["model"])
    with pytest.raises(BadRun, match="no stages"):
        runner.start([])
    assert calls == [] and runner.state()["running"] is False


def test_a_failed_stage_stops_the_run_and_lists_its_errors():
    failing = lambda ctx: StageOutcome("failed", "dataset validation failed", ["2016 Roma: a", "2017 Ajax: b"])
    runner, bus, calls = make({"build": failing})
    runner.start()
    assert runner.wait(5)
    assert calls == ["fetch"]
    assert [e.data for e in events_of(bus, "run_failed")] == [
        {"stage": "build", "errors": ["2016 Roma: a", "2017 Ajax: b"]}]
    assert events_of(bus, "done") == []
    assert statuses(runner)["build"] == "failed" and statuses(runner)["model"] == "idle"


def test_an_exception_fails_the_stage_with_its_message():
    runner, bus, calls = make({"model": recorder([], "model", error=ValueError("bad fold"))})
    runner.start()
    assert runner.wait(5)
    model = next(s for s in runner.state()["stages"] if s["name"] == "model")
    assert (model["status"], model["message"]) == ("failed", "ValueError: bad fold")
    assert [e.data for e in events_of(bus, "run_failed")] == [{"stage": "model", "errors": ["ValueError: bad fold"]}]
    assert not runner.state()["running"]


def test_a_warning_continues_and_marks_the_whole_run():
    runner, bus, calls = make({"analyze": lambda ctx: StageOutcome("warning", "LM Studio not ready")})
    runner.start()
    assert runner.wait(5)
    assert calls == ["fetch", "build", "model", "report"]
    done = events_of(bus, "done")[0].data
    assert done["status"] == "warning" and {"name": "analyze", "status": "warning"} in done["stages"]
    assert {"level": "warn", "text": "analyze: LM Studio not ready"} in [e.data for e in events_of(bus, "log")]


def test_skip_ai_marks_analyze_skipped_without_calling_it():
    runner, bus, calls = make()
    runner.start(skip_ai=True)
    assert runner.wait(5)
    assert "analyze" not in calls and statuses(runner)["analyze"] == "skipped"


def test_two_runs_cannot_overlap():
    gate = threading.Event()
    runner, bus, calls = make({"fetch": lambda ctx: gate.wait(5) and StageOutcome("done")})
    results = []

    def attempt():
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
    assert runner._thread.daemon  # a Ctrl-C never waits on a run
    gate.set()
    assert runner.wait(5)
    assert runner.start(["fetch"]) == "b-r2"  # free again once the run ended
    assert runner.wait(5)


def test_the_run_ends_cleanly_even_when_publishing_fails():
    class BrokenBus(EventBus):
        def publish(self, *args, **kwargs):
            raise RuntimeError("bus down")

    runner = PipelineRunner({name: (lambda ctx: StageOutcome("done")) for name in ORDER}, BrokenBus(boot="b"))
    runner.start()
    assert runner.wait(5)
    assert set(statuses(runner).values()) == {"done"}


def test_state_is_a_deep_copy():
    runner, bus, calls = make()
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

    runner, bus, calls = make({"fetch": fetch})
    runner.start(["fetch"])
    assert runner.wait(5)
    counters = runner.state()["counters"]
    assert counters["requests"] == {"network": 3, "cache": 0}
    assert counters["seasons"] == {"done": 1, "of": 15} and counters["validated"] is True
    assert counters["folds"] == {"done": 0, "of": 15} and counters["narratives"] == {"done": 0, "of": 11}
    runner.start(["report"])
    assert runner.wait(5)
    assert runner.state()["counters"]["requests"]["network"] == 0


def test_progress_is_coalesced_and_flushed_at_the_end_of_each_stage():
    now = [0.0]

    def fetch(ctx):
        for _ in range(3):
            ctx.bump("seasons")  # at t=0: the first publishes, the next two wait
        now[0] = 1.0
        ctx.bump("seasons")  # t=1: publishes at once, carrying all four
        ctx.bump("seasons")  # waits, then goes out when the stage ends
        return StageOutcome("done")

    runner, bus, calls = make({"fetch": fetch}, clock=lambda: now[0])
    runner.start(["fetch"])
    assert runner.wait(5)
    assert [e.data["counters"]["seasons"]["done"] for e in events_of(bus, "progress")] == [1, 4, 5]


def test_stage_events_carry_status_times_and_message():
    wall = iter([100.0, 101.5])
    runner, bus, calls = make(wall=lambda: next(wall))
    runner.start(["fetch"])
    assert runner.wait(5)
    assert [e.data for e in events_of(bus, "stage")] == [
        {"name": "fetch", "status": "running", "started": 100.0, "finished": None, "message": ""},
        {"name": "fetch", "status": "done", "started": 100.0, "finished": 101.5, "message": "fetch ok"},
    ]
    assert [e.data for e in events_of(bus, "log")] == [{"level": "info", "text": "fetch says hi"}]
    assert {e.run_id for e in events_of(bus)} == {"b-r1"}
```

Run: `uv run pytest tests/test_web_pipeline.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ucl.web.pipeline'`.

- [ ] **Step 2: Implement**

Create `src/ucl/web/pipeline.py`:

```python
"""One pipeline run at a time, on a daemon thread. Stage statuses, counters and log lines go out through the
event bus (spec §4.3). The stages are injected (services.py has the real ones), so this file knows nothing
about UEFA, models or LM Studio."""
from __future__ import annotations

import copy
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .. import config
from .events import EventBus

ORDER = ["fetch", "build", "model", "analyze", "report"]
STAGE_LABELS = {"fetch": "Fetch", "build": "Build", "model": "Model", "analyze": "Analyze", "report": "Report"}
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
    }


def _fresh_states() -> list[dict]:
    return [{"name": name, "status": "idle", "started": None, "finished": None, "message": ""} for name in ORDER]


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
                 clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time):
        self._stages, self.bus, self._missing_inputs = stages, bus, missing_inputs
        self._clock, self._wall = clock, wall  # clock paces progress events; wall stamps stage times
        self._lock = threading.Lock()
        self._running = False
        self._run_id: str | None = None
        self._runs = 0
        self._thread: threading.Thread | None = None
        self._states = _fresh_states()
        self._counters = fresh_counters()
        self._last_progress = float("-inf")
        self._dirty = False

    def start(self, stages: list[str] | None = None, skip_ai: bool = False) -> str:
        """Start a run of `stages` (default: all) in canonical order. Raises BadRun or RunInProgress."""
        requested = list(ORDER) if stages is None else list(stages)
        unknown = [name for name in requested if name not in ORDER]
        if unknown:
            raise BadRun(f"unknown stage(s): {', '.join(unknown)}")
        names = [name for name in ORDER if name in requested]
        if not names:
            raise BadRun("no stages to run")
        missing = self._missing_inputs(names)
        if missing:
            raise BadRun(missing)
        with self._lock:
            if self._running:
                raise RunInProgress(f"run {self._run_id} is still going")
            self._runs += 1
            run_id = f"{self.bus.boot}-r{self._runs}"
            self._running, self._run_id = True, run_id
            self._states, self._counters = _fresh_states(), fresh_counters()
            self._last_progress, self._dirty = float("-inf"), False
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

    def _run(self, run_id: str, names: list[str], skip_ai: bool) -> None:
        overall = "done"
        try:
            for name in names:
                if name == "analyze" and skip_ai:
                    self._set(run_id, name, status="skipped", message="AI write-ups skipped")
                    continue
                self._set(run_id, name, status="running", started=self._wall())
                try:
                    outcome = self._stages[name](StageContext(self, run_id, name))
                except Exception as exc:  # noqa: BLE001 - any crash fails the stage and ends the run
                    outcome = StageOutcome("failed", f"{type(exc).__name__}: {exc}")
                self._flush(run_id)
                self._set(run_id, name, status=outcome.status, finished=self._wall(), message=outcome.message)
                if outcome.status == "failed":
                    self._emit("log", {"level": "error", "text": f"{name} failed: {outcome.message}"}, run_id, name)
                    self._emit("run_failed", {"stage": name, "errors": outcome.errors or [outcome.message]},
                               run_id, name)
                    return
                if outcome.status == "warning":
                    overall = "warning"
                    self._emit("log", {"level": "warn", "text": f"{name}: {outcome.message}"}, run_id, name)
            stages = [{"name": s["name"], "status": s["status"]} for s in self.state()["stages"]]
            self._emit("done", {"status": overall, "stages": stages}, run_id)
        finally:
            with self._lock:
                self._running = False

    def _set(self, run_id: str, name: str, **changes) -> None:
        with self._lock:
            state = next(s for s in self._states if s["name"] == name)
            state.update(changes)
            self._emit("stage", dict(state), run_id, name)  # under the lock, so events keep the states' order

    def _update(self, change: Callable[[dict], None]) -> None:
        """Apply a counter change. Publish progress now if the last one went out long enough ago; else later."""
        with self._lock:
            change(self._counters)
            now = self._clock()
            if now - self._last_progress >= PROGRESS_EVERY_S:
                self._last_progress, self._dirty = now, False
                self._emit("progress", {"counters": copy.deepcopy(self._counters)}, self._run_id)
            else:
                self._dirty = True

    def _flush(self, run_id: str) -> None:
        """Publish a counter change still waiting for its slot (at the end of each stage)."""
        with self._lock:
            if self._dirty:
                self._last_progress, self._dirty = self._clock(), False
                self._emit("progress", {"counters": copy.deepcopy(self._counters)}, run_id)

    def _emit(self, type: str, data: dict, run_id: str | None, stage: str | None = None) -> None:
        """Publish; a bus problem must never change a stage's result."""
        try:
            self.bus.publish(type, data, run_id=run_id, stage=stage)
        except Exception:  # noqa: BLE001
            pass
```

- [ ] **Step 3: Run the tests**

Run: `uv run pytest tests/test_web_pipeline.py -q`
Expected: `13 passed`.

- [ ] **Step 4: Run the whole suite and commit**

Run: `uv run pytest -q`
Expected: all pass.

```bash
git add src/ucl/web/pipeline.py tests/test_web_pipeline.py
git commit -m "feat: PipelineRunner runs one pipeline at a time with statuses, counters and events"
```

### Task 10: The real stages, prerequisites and the `Services` bundle

**Files:**
- Create: `src/ucl/web/services.py`
- Test: `tests/test_web_services.py`

Each real stage calls `stages.py`, translates its progress into runner counters, and reloads the matching `DataStore` part. Its tests patch the `stages.*` functions and the loaders, so they never touch the network, LM Studio, `data/` or `out/`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_web_services.py`:

```python
from types import SimpleNamespace

import pytest

from ucl import analyst, dataset, model, stages
from ucl.analyst import Analysis
from ucl.web import services
from ucl.web.events import EventBus
from ucl.web.pipeline import BadRun, PipelineRunner


class FakeStore:
    def __init__(self):
        self.reloads = []

    def reload(self, *parts):
        self.reloads.append(parts)


def run_stage(name, **kwargs):
    """Run one real stage through a runner; returns its state, the counters, the bus and the store."""
    store, bus = FakeStore(), EventBus(boot="b")
    runner = PipelineRunner(services.stage_registry(store, **kwargs), bus)
    runner.start([name])
    assert runner.wait(10)
    state = runner.state()
    return next(s for s in state["stages"] if s["name"] == name), state["counters"], bus, store


def events_of(bus, type):
    subscription = bus.subscribe()
    events = []
    while (event := subscription.get(0)) is not None:
        if event.type == type:
            events.append(event.data)
    return events


@pytest.fixture
def loaders(monkeypatch):
    monkeypatch.setattr(dataset, "load", lambda: "ds")
    monkeypatch.setattr(model, "load", lambda: "results")
    monkeypatch.setattr(analyst, "load", lambda: "analysis")


def client_with_hook(hook):
    return SimpleNamespace(hook=hook)


def test_fetch_counts_requests_and_seasons(monkeypatch):
    def fetch(client, progress=None, log=print):
        client.hook("matches", "matches/2012", "network")
        client.hook("stats", "stats/1", "cache")
        progress({"season": 2012, "matches": 4, "phase_matches": 3, "missing": 0, "failed": 0})
        log("2012: 4 matches, 3 phase matches, stats missing 0, failed 0")
        return stages.FetchResult({})

    monkeypatch.setattr(stages, "fetch", fetch)
    stage, counters, bus, _ = run_stage("fetch", client_factory=client_with_hook)
    assert stage["status"] == "done"
    assert counters["requests"] == {"network": 1, "cache": 1} and counters["seasons"]["done"] == 1
    assert {"level": "info", "text": "2012: 4 matches, 3 phase matches, stats missing 0, failed 0"} in (
        events_of(bus, "log"))


def test_fetch_with_failed_stats_fails_and_lists_them(monkeypatch):
    monkeypatch.setattr(stages, "fetch", lambda client, progress=None, log=print: stages.FetchResult({"9": "down"}))
    stage, _, bus, _ = run_stage("fetch", client_factory=client_with_hook)
    assert stage["status"] == "failed"
    assert stage["message"] == "1 match-stat requests failed; run fetch again to resume"
    assert events_of(bus, "run_failed") == [{"stage": "fetch", "errors": ["9: down"]}]


def test_build_marks_the_dataset_validated_and_reloads_it(monkeypatch):
    monkeypatch.setattr(stages, "build", lambda client: SimpleNamespace(team_seasons=[1, 2, 3], features=["a", "b"]))
    stage, counters, _, store = run_stage("build", client_factory=client_with_hook)
    assert (stage["status"], stage["message"]) == ("done", "3 team-seasons, 2 features")
    assert counters["validated"] is True and store.reloads == [("dataset",)]


def test_a_build_that_fails_validation_lists_every_error_and_reloads_nothing(monkeypatch):
    def invalid(client):
        raise dataset.ValidationError("2016 Roma: distance_km_pg = 3.2 is outside 90-140\n2017 Ajax: possession")

    monkeypatch.setattr(stages, "build", invalid)
    stage, counters, bus, store = run_stage("build", client_factory=client_with_hook)
    assert stage["status"] == "failed" and counters["validated"] is False and store.reloads == []
    assert events_of(bus, "run_failed") == [{"stage": "build", "errors": [
        "2016 Roma: distance_km_pg = 3.2 is outside 90-140", "2017 Ajax: possession"]}]


def test_model_moves_the_fold_and_ablation_counters_and_reloads_results(monkeypatch, loaders):
    def fit(ds, progress=None):
        progress({"fold": 3, "of": 15})
        progress({"ablation_set": "pedigree", "k": 1, "of": 7})
        return SimpleNamespace(metrics={"spearman_mean": 0.4512, "auc": 0.7349})

    monkeypatch.setattr(stages, "model", fit)
    stage, counters, _, store = run_stage("model")
    assert counters["folds"] == {"done": 3, "of": 15} and counters["ablation"] == {"done": 1, "of": 7}
    assert (stage["status"], stage["message"]) == ("done", "Spearman 0.45, AUC 0.73")
    assert store.reloads == [("results",)]


def fake_analyze(result):
    def analyze(ds, results, llm_model, enabled, log=print, merge=False):
        assert (ds, results, enabled, merge) == ("ds", "results", True, True)
        log("  2026-52280: ok, 2 call(s), 0 unsupported")
        log("  synthesis: unavailable, 3 call(s), 0 unsupported")
        return result
    return analyze


def test_analyze_counts_narratives_and_warns_about_any_it_could_not_write(monkeypatch, loaders):
    monkeypatch.setattr(stages, "analyze", fake_analyze(stages.AnalyzeResult(Analysis("ok", "m"), ["synthesis"])))
    stage, counters, bus, store = run_stage("analyze")
    assert counters["narratives"] == {"done": 2, "of": 11}
    assert stage["status"] == "warning"
    assert stage["message"] == "1 write-up could not be written this time (earlier text kept where there was one): synthesis"
    logs = [e["text"] for e in events_of(bus, "log")]
    assert logs[0] == "Checking LM Studio; loading the model can take a few minutes"
    assert "2026-52280: ok, 2 call(s), 0 unsupported" in logs  # stripped of the CLI's indent
    assert store.reloads == [("analysis",)]


def test_analyze_warns_when_lm_studio_is_not_ready(monkeypatch, loaders):
    unavailable = Analysis("unavailable", "m", reason="server did not start: lms not found")
    monkeypatch.setattr(stages, "analyze", fake_analyze(stages.AnalyzeResult(unavailable)))
    stage, _, _, _ = run_stage("analyze")
    assert stage["status"] == "warning"
    assert stage["message"] == "LM Studio not ready (server did not start: lms not found); earlier write-ups kept"


def test_analyze_uses_the_chosen_llm_model(monkeypatch, loaders):
    seen = []

    def analyze(ds, results, llm_model, enabled, log=print, merge=False):
        seen.append(llm_model)
        return stages.AnalyzeResult(Analysis("ok", llm_model))

    monkeypatch.setattr(stages, "analyze", analyze)
    stage, _, _, _ = run_stage("analyze", llm_model="other/model")
    assert seen == ["other/model"] and (stage["status"], stage["message"]) == ("done", "0 write-ups")


def test_report_writes_the_pages(monkeypatch, loaders, tmp_path):
    monkeypatch.setattr(stages, "report", lambda ds, results, analysis: tmp_path / "report.html")
    stage, _, _, _ = run_stage("report")
    assert (stage["status"], stage["message"]) == ("done", "wrote report.html and report_page.html")


def test_missing_inputs_name_the_stage_to_run_first(tmp_path):
    processed, out = tmp_path / "processed", tmp_path / "out"
    assert services.missing_inputs(["model"], processed, out) == (
        f"model needs {processed / 'dataset.json'}: run build first")
    assert services.missing_inputs(["build", "model"], processed, out) is None  # the run makes it
    assert services.missing_inputs(["fetch"], processed, out) is None
    processed.mkdir()
    (processed / "dataset.json").write_text("{}")
    assert services.missing_inputs(["report"], processed, out) == (
        f"report needs {out / 'metrics.json'}: run model first")
    assert services.missing_inputs(["model", "report"], processed, out) is None


def test_build_services_shares_one_bus_and_checks_inputs(tmp_path):
    svc = services.build_services(tmp_path / "processed", tmp_path / "out")
    assert not svc.store.snapshot.ready and svc.runner.bus is svc.bus
    with pytest.raises(BadRun, match="run build first"):
        svc.runner.start(["model"])
```

Run: `uv run pytest tests/test_web_services.py -q`
Expected: FAIL with `ImportError: cannot import name 'services' from 'ucl.web'`.

- [ ] **Step 2: Implement**

Create `src/ucl/web/services.py`:

```python
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
    }
    for name in names:
        for producer, path in needs.get(name, []):
            if producer not in names and not path.exists():
                return f"{name} needs {_shown(path)}: run {producer} first"
    return None


def stage_registry(store, llm_model: str = config.LLM_MODEL,
                   client_factory: Callable[[Callable], object] | None = None) -> dict[str, Stage]:
    """name -> stage for the runner. `store` needs only `reload(*parts)`; `client_factory(hook)` makes the UEFA
    client, so tests can pass a fake."""
    from .. import analyst, dataset, model

    def make_client(ctx: StageContext):
        def hook(kind: str, key: str, source: str) -> None:
            ctx.bump("requests", source)

        if client_factory is not None:
            return client_factory(hook)
        from ..uefa import UefaClient

        return UefaClient(on_request=hook)

    def fetch(ctx: StageContext) -> StageOutcome:
        result = stages.fetch(make_client(ctx), progress=lambda p: ctx.bump("seasons"), log=ctx.log)
        if result.failed:
            errors = [f"{match_id}: {reason}" for match_id, reason in result.failed.items()]
            return StageOutcome("failed", f"{len(errors)} match-stat requests failed; run fetch again to resume",
                                errors[:MAX_ERRORS])
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
    store = DataStore(processed_dir, out_dir)
    bus = EventBus()
    runner = PipelineRunner(stage_registry(store, llm_model), bus,
                            missing_inputs=lambda names: missing_inputs(names, processed_dir, out_dir))
    return Services(store, bus, runner)
```

`stages.*` and the loaders write and read the default `config` paths. `build_services` takes directories only for the store and the prerequisite check, so tests can point those at `tmp_path`. The real app always uses the defaults.

- [ ] **Step 3: Run the tests**

Run: `uv run pytest tests/test_web_services.py -q`
Expected: `11 passed`.

- [ ] **Step 4: Run the whole suite and commit**

Run: `uv run pytest -q`
Expected: all pass.

```bash
git add src/ucl/web/services.py tests/test_web_services.py
git commit -m "feat: real pipeline stages wired into the web runner"
```

---

## Chunk 4: Queries (the API's data shapes)

### Task 11: Dashboard config, feature metadata, meta and search

**Files:**
- Modify: `src/ucl/config.py` (append a block)
- Create: `src/ucl/web/queries.py`
- Test: `tests/test_web_queries.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_web_queries.py`:

```python
import json

import pandas as pd
import pytest

from ucl import analyst, config, dataset, facts, model
from ucl.analyst import Analysis, Narrative
from ucl.facts import build_facts
from ucl.web import queries
from ucl.web.store import DataStore


def test_every_feature_has_exactly_one_section():
    placed = [f for features in config.STAT_SECTIONS.values() for f in features]
    assert sorted(placed) == sorted(config.FEATURES) and len(placed) == len(set(placed))


def test_feature_meta_says_how_to_show_a_stat():
    assert queries.feature_meta("possession_pct") == {
        "feature": "possession_pct", "label": "Possession (%)", "section": "Control", "group": "style",
        "decimals": 1, "percent": True, "lower_is_better": False}
    assert queries.feature_meta("fouls_pg")["lower_is_better"] is True
    assert queries.feature_meta("conversion")["percent"] is True and queries.feature_meta("coef_log")["percent"] is False


def teams(*rows):
    return pd.DataFrame([{"season": s, "team_id": t, "team": n, "team_display": d, "stage_label": stage}
                         for s, t, n, d, stage in rows])


TS = teams(
    (2026, "52747", "Paris", "Paris Saint-Germain", "Winner"),
    (2025, "52747", "Paris", "Paris Saint-Germain", "Winner"),
    (2026, "50124", "Atleti", "Atleti", "Round of 16"),
    (2024, "50051", "Real Madrid", "Real Madrid", "Winner"),
    (2026, "999", "Surreal FC", "Surreal FC", "League phase"),
    (2024, "52758", "B. Dortmund", "Borussia Dortmund", "Runner-up"),
)


def ids(results):
    return [r["team_id"] for r in results]


def test_short_queries_find_nothing():
    assert queries.search(TS, "") == [] and queries.search(TS, " p ") == []


def test_an_alias_finds_a_team_with_its_seasons_newest_first():
    assert queries.search(TS, "PSG") == [{"team_id": "52747", "name": "Paris Saint-Germain", "seasons": [
        {"season": 2026, "label": "2025-26", "stage_label": "Winner"},
        {"season": 2025, "label": "2024-25", "stage_label": "Winner"}]}]


def test_matching_ignores_case_accents_and_punctuation():
    assert ids(queries.search(TS, "atlético")) == ["50124"]  # through the alias "atletico madrid"
    assert ids(queries.search(TS, "b dortmund")) == ["52758"]  # UEFA's "B. Dortmund"
    assert ids(queries.search(TS, "DORTMUND")) == ["52758"]  # inside "Borussia Dortmund"


def test_prefix_matches_rank_before_matches_inside_a_name():
    assert ids(queries.search(TS, "real")) == ["50051", "999"]  # Real Madrid first, though Surreal FC is newer
    assert ids(queries.search(TS, "real", limit=1)) == ["50051"]


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory, built):
    """The synthetic outputs on disk, with narratives that exercise every badge."""
    ds, results = built
    root = tmp_path_factory.mktemp("outputs")
    processed, out = root / "processed", root / "out"
    dataset.save(ds, processed)
    model.save(results, out)
    loaded = dataset.load(processed)
    sheets = build_facts(loaded.team_seasons, loaded.finals, model.load(out))
    narratives = {k: Narrative(k, "ok", text="## How they got there\nThey beat **everyone**.") for k in sheets}
    narratives["2026-52747"] = Narrative("2026-52747", "ok", text="## How they got there\nA 99.9 figure.",
                                         unsupported=["99.9"])
    narratives["2025-52747"] = Narrative("2025-52747", "unavailable", reason="empty answer")
    sheets["2024-50051"] = {"Club": "numbers from an earlier run"}
    analyst.save(Analysis("ok", "m", narratives, sheets), out / "analysis.json")
    return DataStore(processed, out).snapshot


def test_meta_lists_the_datasets_seasons_features_and_stages(snapshot):
    meta = queries.meta(snapshot)
    assert meta["ready"] is True and meta["problems"] == []
    assert [s["season"] for s in meta["seasons"]] == [2021, 2022, 2023, 2024, 2025, 2026]
    assert meta["seasons"][0] == {"season": 2021, "label": "2020-21", "live": False}
    assert [f["feature"] for f in meta["features"]] == snapshot.dataset.features
    assert [s["name"] for s in meta["stages"]] == ["fetch", "build", "model", "analyze", "report"]
    assert meta["stages"][0] == {"name": "fetch", "label": "Fetch", "optional": False}
    assert (meta["live_season"], meta["live_status"], meta["player_stats"]) == (None, "unavailable", [])
    assert meta["data"]["built_at"].startswith("20") and meta["data"]["modelled_at"].startswith("20")


def test_meta_before_anything_is_built(tmp_path):
    meta = queries.meta(DataStore(tmp_path / "processed", tmp_path / "out").snapshot)
    assert meta["ready"] is False and meta["features"] == []
    assert meta["problems"] == ["processed data missing: run build", "model outputs missing: run model"]
    assert [s["season"] for s in meta["seasons"]] == config.SEASONS
    json.dumps(meta, allow_nan=False)
```

Run: `uv run pytest tests/test_web_queries.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ucl.web.queries'`.

- [ ] **Step 2: Add the dashboard config**

Append to `src/ucl/config.py`:

```python
# --- UCL Lab, the local dashboard (spec 2026-10-01-ucl-lab-dashboard-design.md) ---
WEB_PORT = 8787
WEB_DIR = ROOT / "web"
# How Explore groups the stats. save_pct is listed so a dataset that keeps it still has a home for it.
STAT_SECTIONS = {
    "Results": ["points_pg", "goal_diff_pg"],
    "Attack": ["shots_pg", "shot_accuracy", "conversion", "attacks_pg"],
    "Defence": ["shots_against_pg", "on_target_against_pg", "save_pct"],
    "Control": ["possession_pct", "pass_accuracy", "passes_pg", "long_pass_share"],
    "Intensity": ["distance_km_pg", "fouls_pg"],
    "Pedigree": ["coef_log"],
}
# Names people type that UEFA's names don't contain, matched without case or accents.
TEAM_SEARCH_ALIASES = {
    "psg": "52747",
    "man city": "52919",
    "man united": "52682",
    "manchester united": "52682",
    "barca": "50080",
    "atletico madrid": "50124",
    "spurs": "1652",
    "bvb": "52758",
    "inter milan": "50138",
    "internazionale": "50138",
    "ac milan": "50058",
    "juve": "50139",
    "bayern munich": "50037",
    "gladbach": "52757",
}
```

- [ ] **Step 3: Write `queries.py` with meta and search**

Create `src/ucl/web/queries.py`:

```python
"""Pure functions that shape a snapshot into the API's JSON (spec §5). app.py adds HTTP; everything returned
here has been through to_jsonable."""
from __future__ import annotations

import unicodedata

import pandas as pd

from .. import config, facts
from ..features import display_value
from .jsonsafe import to_jsonable
from .pipeline import ORDER, STAGE_LABELS

SECTION_OF = {feature: section for section, features in config.STAT_SECTIONS.items() for feature in features}
SEARCH_LIMIT = 12
MIN_QUERY = 2
TIE_Z = 0.05  # a smaller gap in oriented z counts as level
HELPS = {1: "higher", -1: "lower"}


class NotFound(Exception):
    """An unknown team or season (HTTP 404)."""


def feature_meta(feature: str) -> dict:
    label, decimals, fraction = config.FEATURE_META[feature]
    return {"feature": feature, "label": label, "section": SECTION_OF[feature],
            "group": config.FEATURE_GROUP[feature], "decimals": decimals,
            "percent": fraction or feature == "possession_pct",
            "lower_is_better": feature in config.LOWER_IS_BETTER}


def meta(snapshot) -> dict:
    ds = snapshot.dataset
    seasons = sorted(int(s) for s in ds.team_seasons["season"].unique()) if ds is not None else list(config.SEASONS)
    return to_jsonable({
        "ready": snapshot.ready,
        "problems": list(snapshot.errors.values()),
        "seasons": [{"season": s, "label": config.season_label(s), "live": False} for s in seasons],
        "live_season": None,  # the live season arrives with Plan B
        "live_status": "unavailable",
        "features": [feature_meta(f) for f in ds.features] if ds is not None else [],
        "player_stats": [],  # Plan B
        "stages": [{"name": name, "label": STAGE_LABELS[name], "optional": False} for name in ORDER],
        "data": {"built_at": snapshot.built_at, "modelled_at": snapshot.modelled_at,
                 "analysed_at": snapshot.analysed_at},
    })


def _fold(text: str) -> str:
    """Lower case without accents or punctuation, so 'Atlético' matches 'atletico' and 'B. Dortmund' 'b dortmund'."""
    decomposed = unicodedata.normalize("NFKD", str(text).casefold())
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join("".join(c if c.isalnum() else " " for c in plain).split())


def search(team_seasons: pd.DataFrame, q: str, limit: int = SEARCH_LIMIT) -> list[dict]:
    """Teams whose display name, UEFA name or alias contains `q`; names that start with it come first, then the
    most recent season."""
    query = _fold(q)
    if len(query) < MIN_QUERY:
        return []
    aliases: dict[str, list[str]] = {}
    for alias, team_id in config.TEAM_SEARCH_ALIASES.items():
        aliases.setdefault(team_id, []).append(alias)
    hits = []
    for team_id, rows in team_seasons.groupby("team_id", sort=False):
        rows = rows.sort_values("season", ascending=False)
        names = {_fold(n) for n in (*rows["team_display"], *rows["team"], *aliases.get(team_id, []))}
        if any(n.startswith(query) for n in names):
            rank = 0
        elif any(query in n for n in names):
            rank = 1
        else:
            continue
        latest = rows.iloc[0]
        hits.append((rank, -int(latest["season"]), _fold(latest["team_display"]), {
            "team_id": team_id,
            "name": latest["team_display"],
            "seasons": [{"season": int(s), "label": config.season_label(int(s)), "stage_label": label}
                        for s, label in zip(rows["season"], rows["stage_label"])],
        }))
    hits.sort(key=lambda hit: hit[:3])
    return to_jsonable([hit[3] for hit in hits[:limit]])
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_web_queries.py -q`
Expected: `8 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/ucl/config.py src/ucl/web/queries.py tests/test_web_queries.py
git commit -m "feat: dashboard config, feature metadata, meta and team search"
```

### Task 12: Profile, compare and summary

**Files:**
- Modify: `src/ucl/web/queries.py` (append)
- Test: `tests/test_web_queries.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_web_queries.py`:

```python
def row_of(snapshot, team_id, season):
    ts = snapshot.dataset.team_seasons
    return ts[(ts["team_id"] == team_id) & (ts["season"] == season)].iloc[0]


def test_profile_header_result_and_seasons_played(snapshot):
    p = queries.profile(snapshot, "52280", 2026)
    assert (p["team_id"], p["name"], p["season"], p["label"]) == ("52280", "Team 52280", 2026, "2025-26")
    assert (p["ko_stage"], p["matches_played"], p["live"], p["live_available"]) == (3.0, 8, False, False)
    assert (p["stale"], p["fetched_at"]) == (False, None)
    assert p["result"] == "Runner-up: lost the final to Paris Saint-Germain 1-2"
    assert p["seasons"] == [{"season": 2026, "label": "2025-26"}]
    json.dumps(p, allow_nan=False)


def test_profile_stats_are_display_values_with_teams_beaten(snapshot):
    p = queries.profile(snapshot, "52280", 2026)
    row = row_of(snapshot, "52280", 2026)
    stats = {s["feature"]: s for s in p["features"]}
    assert list(stats) == snapshot.dataset.features
    assert stats["fouls_pg"]["value"] == round(row["fouls_pg"], 1) and stats["fouls_pg"]["section"] == "Intensity"
    assert stats["fouls_pg"]["beats_season"] == 100 - int(row["pct_season_fouls_pg"])  # lower is better: flipped
    assert stats["points_pg"]["beats_all"] == int(row["pct_all_points_pg"])
    assert stats["shot_accuracy"]["value"] == round(row["shot_accuracy"] * 100, 1)  # stored as a fraction
    assert stats["coef_log"]["value"] == round(row["coef"], 1)  # shown as coefficient points
    assert stats["points_pg"]["z"] == pytest.approx(row["z_points_pg"]) and stats["points_pg"]["provisional"] is False


def test_profile_trend_covers_every_season_the_team_played(snapshot):
    p = queries.profile(snapshot, "52747", 2026)
    assert [point["season"] for point in p["trend"]["points_pg"]] == [2025, 2026]
    assert p["trend"]["points_pg"][0]["live"] is False
    assert p["seasons"] == [{"season": 2025, "label": "2024-25"}, {"season": 2026, "label": "2025-26"}]


def test_profile_model_card_comes_from_the_predictions_and_shap(snapshot):
    card = queries.profile(snapshot, "52280", 2026)["model"]
    preds = snapshot.results.predictions
    pred = preds[(preds["season"] == 2026) & (preds["team_id"] == "52280")].iloc[0]
    assert (card["rank"], card["ko_size"]) == (int(pred["rank_in_season"]), 24)
    assert card["p_final"] == pytest.approx(pred["p_final"]) and card["base_rate"] == pytest.approx(2 / 24)
    assert card["nearest_stage"] in facts.STAGES
    assert 0 < len(card["top_up"]) <= 4 and len(card["top_down"]) <= 3
    assert all(c["contribution"] > 0 for c in card["top_up"]) and all(c["contribution"] < 0 for c in card["top_down"])


def test_a_team_outside_the_knockouts_has_no_model_card_or_narrative(snapshot):
    ts = snapshot.dataset.team_seasons
    out = ts[(ts["season"] == 2026) & ~ts["in_ko"].astype(bool)].iloc[0]
    p = queries.profile(snapshot, out["team_id"], 2026)
    assert p["model"] is None and p["narrative"] is None and p["ko_stage"] is None and p["result"] is None


def test_narrative_badges_cover_good_warn_unavailable_and_stale(snapshot):
    def badge(team_id, season):
        return queries.profile(snapshot, team_id, season)["narrative"]["badge"]

    assert badge("52280", 2026) == {"kind": "good", "label": "All figures found in the data"}
    assert badge("52747", 2026) == {"kind": "warn", "label": "1 figure not found in the data"}
    assert badge("52747", 2025) == {"kind": "unavailable", "label": "AI write-up unavailable"}
    assert badge("50051", 2024) == {"kind": "stale", "label": "Written for earlier numbers"}
    narrative = queries.profile(snapshot, "52280", 2026)["narrative"]
    assert "<strong>everyone</strong>" in narrative["html"] and narrative["text"].startswith("## How")
    assert queries.profile(snapshot, "52747", 2025)["narrative"]["html"] is None


def test_an_unknown_team_season_is_not_found(snapshot):
    with pytest.raises(queries.NotFound):
        queries.profile(snapshot, "52280", 2021)
    with pytest.raises(queries.NotFound):
        queries.profile(snapshot, "nope", 2026)


def test_parse_pick():
    assert queries.parse_pick("52280:2026") == ("52280", 2026)
    for bad in ("", "52280", "52280:", ":2026", "52280:20x6"):
        with pytest.raises(ValueError):
            queries.parse_pick(bad)


def test_compare_orients_z_so_that_positive_is_always_better(snapshot):
    out = queries.compare(snapshot, ("52280", 2026), ("52747", 2026))
    rows = {r["feature"]: r for r in out["rows"]}
    arsenal = row_of(snapshot, "52280", 2026)
    assert rows["fouls_pg"]["a_z"] == pytest.approx(-arsenal["z_fouls_pg"])  # fewer fouls is better
    assert rows["points_pg"]["a_z"] == pytest.approx(arsenal["z_points_pg"])
    assert rows["points_pg"]["a_value"] == round(arsenal["points_pg"], 2)
    assert out["a"] == {"team_id": "52280", "name": "Team 52280", "season": 2026, "label": "2025-26",
                        "stage_label": "synthetic"}
    for r in out["rows"]:
        assert r["ahead"] == queries.ahead(r["a_z"], r["b_z"])
    json.dumps(out, allow_nan=False)


def test_ahead_calls_a_small_gap_or_a_missing_value_a_tie():
    assert queries.ahead(0.30, 0.27) == "tie" and queries.ahead(None, 1.0) == "tie"
    assert queries.ahead(0.4, 0.2) == "a" and queries.ahead(-1.0, 0.5) == "b"


def test_compare_with_an_unknown_pick_is_not_found(snapshot):
    with pytest.raises(queries.NotFound):
        queries.compare(snapshot, ("52280", 2026), ("nope", 2026))


def test_summary_has_metrics_with_intervals_the_top_drivers_and_the_synthesis(snapshot):
    s = queries.summary(snapshot)
    m = snapshot.results.metrics
    assert s["metrics"]["spearman"]["value"] == pytest.approx(m["spearman_mean"])
    assert s["metrics"]["auc"]["ci"] == pytest.approx(m["ci"]["auc"])
    assert s["metrics"]["brier_skill"]["value"] == pytest.approx(m["brier_skill"])
    assert s["metrics"]["top4_share"]["chance"] == pytest.approx(m["finalists_in_top4_chance"])
    assert len(s["drivers"]) == 6
    assert {d["kind"] for d in s["drivers"]} <= {"robust", "conditional", "model-dependent"}
    assert all(d["helps"] in ("higher", "lower", None) for d in s["drivers"])
    assert s["drivers"][0]["importance"] >= s["drivers"][-1]["importance"]
    assert s["synthesis"]["badge"]["kind"] == "good"
    json.dumps(s, allow_nan=False)
```

Run: `uv run pytest tests/test_web_queries.py -q`
Expected: the 8 Task 11 tests pass; the new ones FAIL with `AttributeError: module 'ucl.web.queries' has no attribute 'profile'` (or `parse_pick`, `compare`, `ahead`, `summary`).

- [ ] **Step 2: Implement**

Append to `src/ucl/web/queries.py`:

```python
def _row(team_seasons: pd.DataFrame, team_id: str, season: int) -> pd.Series:
    hit = team_seasons[(team_seasons["team_id"] == team_id) & (team_seasons["season"] == season)]
    if hit.empty:
        raise NotFound(f"no team {team_id} in {config.season_label(season)}")
    return hit.iloc[0]


def _header(row: pd.Series) -> dict:
    season = int(row["season"])
    return {"team_id": row["team_id"], "name": row["team_display"], "season": season,
            "label": config.season_label(season), "stage_label": row["stage_label"]}


def _value(row: pd.Series, feature: str):
    raw = row["coef"] if feature == "coef_log" else row[feature]
    return None if pd.isna(raw) else display_value(row, feature)


def _beats(row: pd.Series, feature: str, scope: str) -> int | None:
    return None if pd.isna(row.get(f"pct_{scope}_{feature}")) else facts.beats(row, feature, scope)


def _stat(row: pd.Series, feature: str) -> dict:
    return {"feature": feature, "label": config.FEATURE_META[feature][0], "section": SECTION_OF[feature],
            "value": _value(row, feature), "z": row.get(f"z_{feature}"),
            "beats_season": _beats(row, feature, "season"), "beats_all": _beats(row, feature, "all"),
            "provisional": False}  # only live rows are provisional (Plan B)


def _point(row: pd.Series, feature: str) -> dict:
    season = int(row["season"])
    return {"season": season, "label": config.season_label(season), "value": _value(row, feature),
            "beats_season": _beats(row, feature, "season"), "live": False}


def _result(finals: pd.DataFrame, team_id: str, season: int) -> str | None:
    final = finals[finals["season"] == season]
    if final.empty or team_id not in (final.iloc[0]["winner_id"], final.iloc[0]["runner_up_id"]):
        return None
    return facts.result_text(team_id, final.iloc[0])


def _model_card(results, row: pd.Series, features: list[str]) -> dict | None:
    if results is None:
        return None
    season, team_id = int(row["season"]), row["team_id"]
    preds = results.predictions
    hit = preds[(preds["season"] == season) & (preds["team_id"] == team_id)]
    if hit.empty:
        return None  # not a knockout team, or model outputs older than the dataset
    pred = hit.iloc[0]
    shap = results.shap
    shap_row = shap[(shap["season"] == season) & (shap["team_id"] == team_id)]
    columns = [f for f in features if f"shap_{f}" in shap.columns]
    contributions = [] if shap_row.empty else sorted(
        ((f, float(shap_row.iloc[0][f"shap_{f}"])) for f in columns), key=lambda kv: kv[1], reverse=True)

    def entry(feature: str, contribution: float) -> dict:
        return {"feature": feature, "label": config.FEATURE_META[feature][0], "value": _value(row, feature),
                "contribution": round(contribution, 3)}

    exp_stage = float(pred["exp_stage"])
    return {"p_final": float(pred["p_final"]), "base_rate": float(pred["base_rate"]),
            "rank": int(pred["rank_in_season"]), "ko_size": int(pred["ko_size"]), "exp_stage": exp_stage,
            "nearest_stage": facts.nearest_stage(round(exp_stage, 2)),
            "top_up": [entry(f, c) for f, c in contributions if c > 0][:4],
            "top_down": [entry(f, c) for f, c in reversed(contributions) if c < 0][:3]}


def _narrative(analysis, key: str, stale_keys: frozenset[str]) -> dict | None:
    """The write-up for `key` with its badge; None when there is none (only the finalists and the synthesis)."""
    narrative = analysis.narratives.get(key) if analysis is not None else None
    if narrative is None:
        return None
    if narrative.status != "ok" or not narrative.text:
        return {"text": None, "html": None, "badge": {"kind": "unavailable", "label": "AI write-up unavailable"}}
    from ..report import md_to_html  # escapes the model's text, so the browser can render the HTML as is

    if key in stale_keys:
        badge = {"kind": "stale", "label": "Written for earlier numbers"}
    elif narrative.unsupported:
        n = len(narrative.unsupported)
        badge = {"kind": "warn", "label": f"{n} figure{'s' if n != 1 else ''} not found in the data"}
    else:
        badge = {"kind": "good", "label": "All figures found in the data"}
    return {"text": narrative.text, "html": md_to_html(narrative.text, narrative.unsupported), "badge": badge}


def profile(snapshot, team_id: str, season: int) -> dict:
    """One team-season: header, stats with "teams beaten", trends, model card and AI report (spec §5)."""
    ds = snapshot.dataset
    row = _row(ds.team_seasons, team_id, season)
    played = ds.team_seasons[ds.team_seasons["team_id"] == team_id].sort_values("season")
    return to_jsonable({
        **_header(row),
        "live": False, "live_available": False, "stale": False, "fetched_at": None,  # live data is Plan B
        "matches_played": int(row["n_matches"]),
        "ko_stage": row["ko_stage"],
        "result": _result(ds.finals, team_id, season),
        "seasons": [{"season": int(s), "label": config.season_label(int(s))} for s in played["season"]],
        "features": [_stat(row, f) for f in ds.features],
        "trend": {f: [_point(r, f) for _, r in played.iterrows()] for f in ds.features},
        "model": _model_card(snapshot.results, row, ds.features),
        "narrative": _narrative(snapshot.analysis, f"{season}-{team_id}", snapshot.stale_keys),
    })


def parse_pick(text: str) -> tuple[str, int]:
    """'52280:2026' -> ('52280', 2026). Raises ValueError for anything else."""
    team_id, sep, season = (text or "").partition(":")
    if not sep or not team_id or not season.isdigit():
        raise ValueError(f"expected team:season, like 52280:2026, not {text!r}")
    return team_id, int(season)


def _oriented(row: pd.Series, feature: str) -> float | None:
    z = row.get(f"z_{feature}")
    if z is None or pd.isna(z):
        return None
    return -float(z) if feature in config.LOWER_IS_BETTER else float(z)


def ahead(a_z: float | None, b_z: float | None) -> str:
    if a_z is None or b_z is None or abs(a_z - b_z) < TIE_Z:
        return "tie"
    return "a" if a_z > b_z else "b"


def compare(snapshot, a: tuple[str, int], b: tuple[str, int]) -> dict:
    """Two team-seasons stat by stat. a_z/b_z are oriented: flipped for lower-is-better stats, so positive is
    always better and the UI draws them as given (spec §5)."""
    ds = snapshot.dataset
    row_a, row_b = _row(ds.team_seasons, *a), _row(ds.team_seasons, *b)
    rows = []
    for feature in ds.features:
        a_z, b_z = _oriented(row_a, feature), _oriented(row_b, feature)
        rows.append({"feature": feature, "label": config.FEATURE_META[feature][0], "section": SECTION_OF[feature],
                     "a_value": _value(row_a, feature), "b_value": _value(row_b, feature), "a_z": a_z, "b_z": b_z,
                     "a_beats": _beats(row_a, feature, "season"), "b_beats": _beats(row_b, feature, "season"),
                     "ahead": ahead(a_z, b_z)})
    return to_jsonable({"a": _header(row_a), "b": _header(row_b), "rows": rows})


def summary(snapshot) -> dict:
    """The overview: headline metrics with 95% intervals, the top drivers and the AI summary."""
    metrics = snapshot.results.metrics

    def metric(key: str) -> dict:
        return {"value": metrics[key], "ci": list(metrics["ci"][key])}

    drivers = snapshot.results.drivers
    top = drivers[drivers["rank"] <= config.TOP_DRIVERS].sort_values("rank")
    return to_jsonable({
        "metrics": {"spearman": metric("spearman_mean"), "auc": metric("auc"), "brier_skill": metric("brier_skill"),
                    "top4_share": {**metric("finalists_in_top4"), "chance": metrics["finalists_in_top4_chance"]}},
        "drivers": [{"feature": d.feature, "label": d.label_text, "group": d.group, "importance": d.importance,
                     "kind": d.label or None, "helps": HELPS.get(int(d.direction))} for d in top.itertuples()],
        "synthesis": _narrative(snapshot.analysis, "synthesis", snapshot.stale_keys),
    })
```

- [ ] **Step 3: Run the tests, then try the real data**

Run: `uv run pytest tests/test_web_queries.py -q`
Expected: `20 passed`.

Run:
```bash
uv run python -c "
from ucl.web.store import DataStore; from ucl.web import queries
s = DataStore().snapshot
print([h['name'] for h in queries.search(s.dataset.team_seasons, 'man')][:3])
p = queries.profile(s, '52280', 2026)
print(p['result'], '|', p['model']['rank'], '|', p['narrative']['badge']['kind'])
print(queries.summary(s)['drivers'][0]['label'])"
```
Expected, line by line:
- A list that starts with the "Man…" clubs.
- `Runner-up: lost the final to Paris Saint-Germain on penalties (1-1, 3-4 on penalties) | 2 | good`
- `Attacks per game`

- [ ] **Step 4: Commit**

```bash
git add src/ucl/web/queries.py tests/test_web_queries.py
git commit -m "feat: profile, compare and summary queries"
```

---

## Chunk 5: HTTP API and the `ucl web` command

### Task 13: The FastAPI app

**Files:**
- Create: `src/ucl/web/app.py`
- Test: `tests/test_web_app.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_web_app.py`:

```python
import threading

import pytest
from fastapi.testclient import TestClient

from ucl import analyst, dataset, model
from ucl.analyst import Analysis, Narrative
from ucl.facts import build_facts
from ucl.web.app import create_app
from ucl.web.events import EventBus
from ucl.web.pipeline import ORDER, PipelineRunner, StageOutcome
from ucl.web.services import Services
from ucl.web.store import DataStore

NOT_READY = {"error": {"code": "not_ready",
                       "message": "Run the pipeline first: processed data or model outputs are missing."}}


@pytest.fixture(scope="module")
def outputs(tmp_path_factory, built):
    ds, results = built
    root = tmp_path_factory.mktemp("app")
    processed, out = root / "processed", root / "out"
    dataset.save(ds, processed)
    model.save(results, out)
    loaded = dataset.load(processed)
    sheets = build_facts(loaded.team_seasons, loaded.finals, model.load(out))
    narratives = {k: Narrative(k, "ok", text="## How they got there\nThey beat **everyone**.") for k in sheets}
    analyst.save(Analysis("ok", "m", narratives, sheets), out / "analysis.json")
    return processed, out


def make_services(processed, out, gate=None):
    """Real store and bus; fake stages that wait on `gate`, so a run can be held open."""
    gate = gate or threading.Event()

    def stage(ctx):
        gate.wait(5)
        return StageOutcome("done", "ok")

    def missing(names):
        return "model needs data/processed/dataset.json: run build first" if names == ["model"] else None

    bus = EventBus(boot="b")
    return Services(DataStore(processed, out), bus,
                    PipelineRunner({name: stage for name in ORDER}, bus, missing_inputs=missing))


@pytest.fixture
def api(outputs):
    gate = threading.Event()
    services = make_services(*outputs, gate=gate)
    with TestClient(create_app(services, heartbeat=0.05)) as client:
        yield client, services, gate
    gate.set()
    services.runner.wait(5)


def error(response):
    return response.status_code, response.json()["error"]["code"]


def test_meta_reports_the_loaded_data(api):
    client, _, _ = api
    meta = client.get("/api/meta").json()
    assert meta["ready"] is True and meta["seasons"][-1] == {"season": 2026, "label": "2025-26", "live": False}
    assert len(meta["features"]) == 16 and meta["data"]["modelled_at"]


def test_data_routes_answer_503_until_the_pipeline_has_run(tmp_path):
    services = make_services(tmp_path / "processed", tmp_path / "out")
    with TestClient(create_app(services)) as client:
        assert client.get("/api/meta").json()["ready"] is False
        for path in ("/api/summary", "/api/teams?q=team", "/api/teams/52280/seasons/2026",
                     "/api/compare?a=1:2026&b=2:2026"):
            response = client.get(path)
            assert (response.status_code, response.json()) == (503, NOT_READY)
        assert client.get("/api/pipeline/state").status_code == 200


def test_summary_and_search(api):
    client, _, _ = api
    assert client.get("/api/summary").json()["synthesis"]["badge"]["kind"] == "good"
    assert client.get("/api/teams", params={"q": "t"}).json() == []
    hits = client.get("/api/teams", params={"q": "team 5228"}).json()
    assert hits[0]["team_id"] == "52280" and hits[0]["seasons"][0]["season"] == 2026


def test_profile_and_its_errors(api):
    client, _, _ = api
    profile = client.get("/api/teams/52280/seasons/2026").json()
    assert profile["result"].startswith("Runner-up") and "<strong>everyone</strong>" in profile["narrative"]["html"]
    assert error(client.get("/api/teams/52280/seasons/2012")) == (404, "not_found")
    assert error(client.get("/api/teams/52280/seasons/abc")) == (422, "invalid_request")


def test_compare_and_its_errors(api):
    client, _, _ = api
    out = client.get("/api/compare", params={"a": "52280:2026", "b": "52747:2026"}).json()
    assert out["a"]["team_id"] == "52280" and {r["ahead"] for r in out["rows"]} <= {"a", "b", "tie"}
    assert error(client.get("/api/compare", params={"a": "52280", "b": "52747:2026"})) == (422, "invalid_request")
    assert error(client.get("/api/compare", params={"a": "52280:2026", "b": "nope:2026"})) == (404, "not_found")


def test_a_run_starts_once_and_bad_runs_are_refused(api):
    client, services, gate = api
    response = client.post("/api/pipeline/runs", json={"stages": ["fetch"]})
    assert (response.status_code, response.json()) == (202, {"run_id": "b-r1"})
    assert error(client.post("/api/pipeline/runs", json={})) == (409, "run_in_progress")
    state = client.get("/api/pipeline/state").json()
    assert (state["running"], state["run_id"], state["boot"]) == (True, "b-r1", "b")
    gate.set()
    assert services.runner.wait(5)
    assert error(client.post("/api/pipeline/runs", json={"stages": ["model"]})) == (422, "bad_run")
    assert error(client.post("/api/pipeline/runs", json={"stages": ["train"]})) == (422, "bad_run")
    assert error(client.post("/api/pipeline/runs", json={"skip_ai": "sometimes"})) == (422, "invalid_request")
    assert client.post("/api/pipeline/runs").status_code == 202  # no body at all: every stage


def test_events_stream_the_backlog_and_resume_after_the_last_event_id(api):
    client, services, _ = api
    services.bus.publish("log", {"level": "info", "text": "one"})
    services.bus.publish("log", {"level": "info", "text": "two"})
    services.bus.close()  # the stream then ends after the backlog, so each response completes
    response = client.get("/api/pipeline/events")
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert [line for line in response.text.splitlines() if line.startswith("id: ")] == ["id: b-1", "id: b-2"]
    resumed = client.get("/api/pipeline/events", headers={"Last-Event-ID": "b-1"})
    assert [line for line in resumed.text.splitlines() if line.startswith("id: ")] == ["id: b-2"]


def test_shutdown_closes_the_event_bus(outputs):
    services = make_services(*outputs)
    with TestClient(create_app(services)):
        assert not services.bus.closed
    assert services.bus.closed


def test_the_built_spa_is_served_at_the_root(outputs, tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>UCL Lab</title>")
    with TestClient(create_app(make_services(*outputs), dist_dir=dist)) as client:
        assert "UCL Lab" in client.get("/").text
        assert client.get("/api/meta").json()["ready"] is True  # API routes win over the static mount
```

Run: `uv run pytest tests/test_web_app.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ucl.web.app'`.

- [ ] **Step 2: Implement**

Create `src/ucl/web/app.py`:

```python
"""The HTTP layer (spec §5): JSON routes over the snapshot and the runner, server-sent events, and the built
SPA. Shaping lives in queries.py; this file maps requests to it and failures to {"error": {code, message}}."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import queries
from .events import HEARTBEAT_S
from .pipeline import BadRun, RunInProgress
from .services import Services

NOT_READY = "Run the pipeline first: processed data or model outputs are missing."


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


class RunBody(BaseModel):
    stages: list[str] | None = None
    skip_ai: bool = False


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


def create_app(services: Services, dist_dir: Path | None = None, heartbeat: float = HEARTBEAT_S) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        services.bus.close()  # ends open event streams, so shutdown doesn't wait on them

    app = FastAPI(title="UCL Lab", lifespan=lifespan)

    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError) -> JSONResponse:
        return _error(exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first.get("loc", ())[1:])
        return _error(422, "invalid_request", f"{where}: {first['msg']}" if where else first["msg"])

    def ready():
        snapshot = services.store.snapshot  # read once per request
        if not snapshot.ready:
            raise ApiError(503, "not_ready", NOT_READY)
        return snapshot

    @app.get("/api/meta")
    def meta():
        return queries.meta(services.store.snapshot)

    @app.get("/api/summary")
    def summary():
        return queries.summary(ready())

    @app.get("/api/teams")
    def teams(q: str = ""):
        return queries.search(ready().dataset.team_seasons, q)

    @app.get("/api/teams/{team_id}/seasons/{season}")
    def profile(team_id: str, season: int):
        snapshot = ready()
        try:
            return queries.profile(snapshot, team_id, season)
        except queries.NotFound as exc:
            raise ApiError(404, "not_found", str(exc)) from exc

    @app.get("/api/compare")
    def compare(a: str = "", b: str = ""):
        try:
            picks = queries.parse_pick(a), queries.parse_pick(b)
        except ValueError as exc:
            raise ApiError(422, "invalid_request", str(exc)) from exc
        snapshot = ready()
        try:
            return queries.compare(snapshot, *picks)
        except queries.NotFound as exc:
            raise ApiError(404, "not_found", str(exc)) from exc

    @app.post("/api/pipeline/runs", status_code=202)
    def start_run(body: RunBody | None = None):
        body = body or RunBody()
        try:
            return {"run_id": services.runner.start(body.stages, skip_ai=body.skip_ai)}
        except RunInProgress as exc:
            raise ApiError(409, "run_in_progress", str(exc)) from exc
        except BadRun as exc:
            raise ApiError(422, "bad_run", str(exc)) from exc

    @app.get("/api/pipeline/state")
    def pipeline_state():
        return services.runner.state()

    @app.get("/api/pipeline/events")
    def pipeline_events(request: Request, last_event_id: str | None = None):
        resume = request.headers.get("last-event-id") or last_event_id

        def frames():
            # subscribe once streaming starts, so a client that leaves before then never holds a queue
            yield from services.bus.stream(services.bus.subscribe(resume), heartbeat)

        return StreamingResponse(frames(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    if dist_dir is not None:
        app.mount("/", StaticFiles(directory=dist_dir, html=True), name="spa")
    return app
```

- [ ] **Step 3: Run the tests, then the API on the real data**

Run: `uv run pytest tests/test_web_app.py -q`
Expected: `9 passed`.

Run:
```bash
uv run python -c "
from fastapi.testclient import TestClient
from ucl.web.app import create_app
from ucl.web.services import build_services
with TestClient(create_app(build_services())) as c:
    print(c.get('/api/meta').json()['ready'], len(c.get('/api/teams', params={'q': 'arsenal'}).json()))
    print(c.get('/api/teams/52280/seasons/2026').json()['result'])
    print(c.get('/api/compare', params={'a': '52280:2026', 'b': '52747:2026'}).json()['rows'][0]['label'])"
```
Expected:
```
True 1
Runner-up: lost the final to Paris Saint-Germain on penalties (1-1, 3-4 on penalties)
Pre-season UEFA club coefficient (points)
```

- [ ] **Step 4: Commit**

```bash
git add src/ucl/web/app.py tests/test_web_app.py
git commit -m "feat: FastAPI app with JSON routes, run control and an SSE event stream"
```

### Task 14: `ucl web`, the process

**Files:**
- Create: `src/ucl/web/server.py`
- Modify: `src/ucl/cli.py`, the `main` function
- Test: `tests/test_web_server.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_web_server.py`:

```python
import threading
from types import SimpleNamespace

import pytest

from ucl import cli, config
from ucl.web import server


def test_web_is_dispatched_before_the_stage_parser(monkeypatch):
    seen = []
    monkeypatch.setattr(server, "main", lambda argv: seen.append(argv) or 0)
    assert cli.main(["web", "--no-open", "--port", "9000"]) == 0
    assert seen == [["--no-open", "--port", "9000"]]


def test_help_mentions_the_dashboard(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    assert "ucl web" in capsys.readouterr().out


def test_a_missing_build_prints_the_commands_and_exits_1(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(config, "WEB_DIR", tmp_path / "web")
    assert server.main(["--no-open"]) == 1
    err = capsys.readouterr().err
    assert "npm install && npm run build" in err and str(tmp_path / "web") in err


def test_the_server_listens_on_localhost_only_and_stops_quickly():
    settings = server.uvicorn_config(object(), 9000)
    assert (settings.host, settings.port, settings.timeout_graceful_shutdown) == ("127.0.0.1", 9000, 2)


def test_the_browser_opens_once_the_server_is_up(monkeypatch):
    opened = []
    monkeypatch.setattr(server.webbrowser, "open", opened.append)
    fake = SimpleNamespace(started=False, should_exit=False)
    threading.Timer(0.05, lambda: setattr(fake, "started", True)).start()
    assert server.open_when_up(fake, "http://127.0.0.1:8787/", poll=0.01, limit=2)
    assert opened == ["http://127.0.0.1:8787/"]


def test_the_browser_stays_shut_if_the_server_never_starts(monkeypatch):
    opened = []
    monkeypatch.setattr(server.webbrowser, "open", opened.append)
    assert not server.open_when_up(SimpleNamespace(started=False, should_exit=True), "http://x/", poll=0.01)
    assert opened == []
```

Run: `uv run pytest tests/test_web_server.py -q`
Expected: FAIL with `ImportError: cannot import name 'server' from 'ucl.web'`.

- [ ] **Step 2: Write `server.py`**

Create `src/ucl/web/server.py`:

```python
"""`ucl web`: serve UCL Lab on 127.0.0.1 and open it in the browser (spec §4, "Process")."""
from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser

from .. import config


def uvicorn_config(app, port: int):
    """Localhost only: the POST routes start jobs and LM Studio, and there is no authentication. A short
    graceful-shutdown window means Ctrl-C returns promptly, even mid-run."""
    import uvicorn

    return uvicorn.Config(app, host="127.0.0.1", port=port, timeout_graceful_shutdown=2, log_level="warning")


def open_when_up(server, url: str, poll: float = 0.1, limit: float = 30.0) -> bool:
    """Open `url` once uvicorn says it has started; give up if it stops first or after `limit` seconds."""
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        if server.started:
            webbrowser.open(url)
            return True
        if server.should_exit:
            return False
        time.sleep(poll)
    return False


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="ucl web", description="Start UCL Lab, the local dashboard.")
    parser.add_argument("--port", type=int, default=config.WEB_PORT)
    parser.add_argument("--no-open", action="store_true", help="don't open the browser")
    parser.add_argument("--llm-model", default=config.LLM_MODEL, help="LM Studio model key for the analyze stage")
    args = parser.parse_args(argv)
    dist = config.WEB_DIR / "dist"
    if not (dist / "index.html").exists():
        print(f"The dashboard hasn't been built yet. Build it once with:\n"
              f"  cd {config.WEB_DIR} && npm install && npm run build", file=sys.stderr)
        return 1
    import uvicorn

    from .app import create_app
    from .services import build_services

    server = uvicorn.Server(uvicorn_config(create_app(build_services(llm_model=args.llm_model), dist_dir=dist),
                                           args.port))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"UCL Lab: {url} (Ctrl-C to stop)")
    if not args.no_open:
        threading.Thread(target=open_when_up, args=(server, url), daemon=True).start()
    server.run()
    return 0
```

- [ ] **Step 3: Dispatch `web` from the CLI**

In `src/ucl/cli.py`, replace `main` with:

```python
def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["web"]:  # the dashboard has its own options, so it bypasses the stage parser
        from .web.server import main as web_main

        return web_main(argv[1:])
    parser = argparse.ArgumentParser(
        prog="ucl", description="Why the best Champions League teams win.",
        epilog="ucl web [--port N] [--no-open] starts UCL Lab, the local dashboard.")
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
```

- [ ] **Step 4: Run the tests, then the real command**

Run: `uv run pytest tests/test_web_server.py tests/test_cli.py tests/test_cli_characterization.py -q`
Expected: all pass.

Run: `uv run ucl web --no-open; echo "exit $?"`
Expected: the "hasn't been built yet" message with the `npm` commands, then `exit 1` (`web/dist` comes in Chunk 6).

- [ ] **Step 5: Run the whole suite and commit**

Run: `uv run pytest -q`
Expected: all pass.

```bash
git add src/ucl/web/server.py src/ucl/cli.py tests/test_web_server.py
git commit -m "feat: ucl web serves the dashboard on 127.0.0.1"
```
