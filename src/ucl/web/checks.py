"""Quick checks before a run: is LM Studio up with the model loaded, can UEFA be reached, and are the pipeline's
outputs, the live season, the player index and the dashboard build in place. Every check returns a Check and never
raises; the slow ones (LM Studio and the UEFA feeds) run in parallel with short timeouts."""
from __future__ import annotations

import json
import shutil
import time
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .. import config
from ..llm import LMStudio, http_post_json, parse_loaded
from ..uefa import USER_AGENT

OK, WARN, FAIL, INFO = "ok", "warn", "fail", "info"
GROUPS = ["Local AI", "UEFA feeds", "Pipeline outputs", "Live season and players", "Dashboard"]
NETWORK_TIMEOUT_S = 4.0
LMS_TIMEOUT_S = 10.0
PROBE_TEAM = "52280"  # any club with a cached squad works for the player-feed probe

Get = Callable[[str, float], bytes]
RunLms = Callable[..., "tuple[int | None, str]"]


@dataclass
class Check:
    id: str
    group: str
    label: str
    status: str  # "ok" | "warn" | "fail" | "info"
    detail: str
    fix: str = ""  # what to do about it, when it isn't ok; commands go in backticks


def http_get(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def ago(when: datetime | float | None, now: float | None = None) -> str:
    if when is None:
        return "never"
    seconds = (now if now is not None else time.time()) - (when.timestamp() if isinstance(when, datetime) else when)
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{round(seconds / 60)} min ago"
    if seconds < 86400:
        return f"{round(seconds / 3600)} h ago"
    days = round(seconds / 86400)
    return f"{days} day{'s' if days != 1 else ''} ago"


def _reason(exc: Exception) -> str:
    text = str(getattr(exc, "reason", "") or exc)
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


# --- Local AI ---

def lms_path() -> str | None:
    return str(config.LMS_BIN) if config.LMS_BIN.exists() else shutil.which("lms")


def lms_cli() -> Check:
    path = lms_path()
    if path:
        return Check("lms_cli", "Local AI", "LM Studio command line", OK, f"Found at {path.replace(str(Path.home()), '~')}")
    return Check("lms_cli", "Local AI", "LM Studio command line", FAIL, "The `lms` command isn't installed",
                 "Install LM Studio and open it once, then run `~/.lmstudio/bin/lms bootstrap`")


def lm_server(get: Get, cli_found: bool) -> Check:
    url = f"{config.LLM_BASE_URL}/models"
    try:
        models = json.loads(get(url, 2.0)).get("data", [])
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        if cli_found:
            return Check("lm_server", "Local AI", "LM Studio server", WARN,
                         f"Not answering at {config.LLM_BASE_URL} ({_reason(exc)})",
                         "The analyze stage starts it for you; to start it now, run `lms server start` or turn on "
                         "the server in LM Studio's Developer tab")
        return Check("lm_server", "Local AI", "LM Studio server", FAIL,
                     f"Not answering at {config.LLM_BASE_URL} ({_reason(exc)})",
                     "Open LM Studio and turn on the server in its Developer tab")
    return Check("lm_server", "Local AI", "LM Studio server", OK,
                 f"Answering at {config.LLM_BASE_URL} ({len(models)} model{'s' if len(models) != 1 else ''} listed)")


def lm_model_downloaded(run_lms: RunLms, model: str) -> Check:
    label = "Model downloaded"
    code, output = run_lms("ls", "--json", timeout=LMS_TIMEOUT_S)
    if code != 0:
        return Check("lm_model_downloaded", "Local AI", label, WARN, f"Couldn't list LM Studio's models ({output[:120]})",
                     "Check that LM Studio is installed and `lms ls` works")
    try:
        entries = {entry.get("modelKey"): entry for entry in json.loads(output) if isinstance(entry, dict)}
    except (TypeError, ValueError):
        entries = {}
    if model in entries:
        size = entries[model].get("sizeBytes")
        size_text = f" ({size / 1e9:.1f} GB)" if isinstance(size, (int, float)) else ""
        return Check("lm_model_downloaded", "Local AI", label, OK, f"{model} is downloaded{size_text}")
    return Check("lm_model_downloaded", "Local AI", label, FAIL, f"{model} isn't downloaded",
                 f"Download it with `lms get {model}`, or start `ucl web` with `--llm-model` set to one you have")


def lm_model_loaded(run_lms: RunLms, model: str) -> Check:
    label = "Model loaded"
    want = config.LLM_CONTEXT_LENGTH
    code, output = run_lms("ps", "--json", timeout=LMS_TIMEOUT_S)
    if code != 0:
        return Check("lm_model_loaded", "Local AI", label, WARN, f"Couldn't ask LM Studio what's loaded ({output[:120]})",
                     "The analyze stage loads the model when it needs it")
    context = parse_loaded(output, model)
    if context is None:
        return Check("lm_model_loaded", "Local AI", label, WARN, "Not loaded yet",
                     f"The analyze stage loads it with a {want:,}-token context (a minute or so). To load it now, use "
                     "Test the local model")
    if context < want:
        return Check("lm_model_loaded", "Local AI", label, WARN, f"Loaded with a {context:,}-token context",
                     f"The analyze stage reloads it with {want:,}, which the write-ups need")
    return Check("lm_model_loaded", "Local AI", label, OK, f"Loaded with a {context:,}-token context")


def try_model(model: str, load: bool = True, llm_factory: Callable[[str], LMStudio] = LMStudio,
               post: Callable[[str, dict, float], dict] = http_post_json,
               clock: Callable[[], float] = time.monotonic) -> dict:
    """Load the model if asked (as the analyze stage would) and send it one tiny prompt, bypassing the answer cache,
    so the reply proves LM Studio and the model really work."""
    llm = llm_factory(model)
    started = clock()
    if load and not llm.ensure_ready():
        return {"ok": False, "detail": f"LM Studio isn't ready: {llm.reason or 'unknown reason'}",
                "reply": None, "load_seconds": round(clock() - started, 1), "answer_seconds": None}
    loaded = clock()
    body = {"model": model, "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
            "max_tokens": 16, "temperature": 0, "reasoning_effort": "none"}
    try:
        reply = post(f"{config.LLM_BASE_URL}/chat/completions", body, 120)["choices"][0]["message"].get("content")
    except Exception as exc:  # noqa: BLE001 - reported to the browser
        return {"ok": False, "detail": f"The model didn't answer: {_reason(exc)}", "reply": None,
                "load_seconds": round(loaded - started, 1), "answer_seconds": None}
    answered = clock() - loaded
    return {"ok": True, "detail": f"{model} answered in {answered:.1f} s", "reply": (reply or "").strip()[:120],
            "load_seconds": round(loaded - started, 1), "answer_seconds": round(answered, 1)}


# --- UEFA feeds ---

def uefa_probes(raw_dir: Path) -> list[tuple[str, str, str | None]]:
    """(id, label, url) for each feed the pipeline uses, asking for as little as each allows."""
    cached_stats = next(iter(sorted((raw_dir / "stats").glob("*.json"))), None) if (raw_dir / "stats").exists() else None
    return [
        ("uefa_matches", "Matches (match.uefa.com)",
         config.MATCHES_URL.format(season=config.LIVE_SEASON).replace("limit=500", "limit=1")),
        ("uefa_stats", "Match stats (matchstats.uefa.com)",
         config.MATCH_STATS_URL.format(match_id=cached_stats.stem) if cached_stats else None),
        ("uefa_coefficients", "Club coefficients (comp.uefa.com)",
         config.COEF_URL.format(page=1, season=max(config.SEASONS)).replace(f"pagesize={config.COEF_PAGE_SIZE}",
                                                                              "pagesize=1")),
        ("uefa_players", "Player stats (compstats.uefa.com)",
         config.PLAYERS_URL.format(season=max(config.SEASONS), team_id=PROBE_TEAM, limit=1, offset=0)),
    ]


def uefa_feed(id: str, label: str, url: str | None, get: Get, clock: Callable[[], float] = time.monotonic) -> Check:
    if url is None:
        return Check(id, "UEFA feeds", label, INFO, "No cached match to ask about yet", "Run fetch once")
    started = clock()
    try:
        get(url, NETWORK_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return Check(id, "UEFA feeds", label, WARN, f"Couldn't reach it ({_reason(exc)})",
                     "Cached seasons keep working offline; fetching, the live season and new squads need it. "
                     "Check your connection")
    return Check(id, "UEFA feeds", label, OK, f"Answering ({(clock() - started) * 1000:.0f} ms)")


# --- Pipeline outputs ---

def raw_cache(raw_dir: Path) -> Check:
    seasons = [s for s in config.SEASONS if (raw_dir / "matches" / f"{s}.json").exists()]
    stats = len(list((raw_dir / "stats").glob("*.json"))) if (raw_dir / "stats").exists() else 0
    if len(seasons) == len(config.SEASONS):
        return Check("raw_cache", "Pipeline outputs", "Raw UEFA data", OK,
                     f"All {len(seasons)} seasons cached, {stats:,} match-stat files")
    return Check("raw_cache", "Pipeline outputs", "Raw UEFA data", WARN,
                 f"{len(seasons)} of {len(config.SEASONS)} seasons cached", "Run fetch (it needs UEFA)")


def dataset_check(snapshot) -> Check:
    ds = snapshot.dataset
    if ds is None:
        return Check("dataset", "Pipeline outputs", "Dataset", FAIL,
                     snapshot.errors.get("dataset", "processed data missing"), "Run build")
    return Check("dataset", "Pipeline outputs", "Dataset", OK,
                 f"{len(ds.team_seasons)} team-seasons, {len(ds.features)} features, built {ago(snapshot.built_at)}")


def model_check(snapshot) -> Check:
    results = snapshot.results
    if results is None:
        return Check("model", "Pipeline outputs", "Model", FAIL, snapshot.errors.get("results", "model outputs missing"),
                     "Run model")
    m = results.metrics
    detail = f"Trained {ago(snapshot.modelled_at)}: Spearman {m['spearman_mean']:.2f}, AUC {m['auc']:.2f}"
    if snapshot.built_at and snapshot.modelled_at and snapshot.built_at > snapshot.modelled_at:
        return Check("model", "Pipeline outputs", "Model", WARN, f"{detail}, but the dataset was rebuilt since",
                     "Run model so the predictions match the data")
    return Check("model", "Pipeline outputs", "Model", OK, detail)


def write_ups_check(snapshot) -> Check:
    label = "AI write-ups"
    analysis = snapshot.analysis
    narratives = analysis.narratives if analysis is not None else {}
    if not narratives:
        reason = f" ({analysis.reason})" if analysis is not None and analysis.reason else ""
        return Check("write_ups", "Pipeline outputs", label, WARN, f"None written yet{reason}",
                     "Run analyze with LM Studio ready")
    written = sum(n.status == "ok" for n in narratives.values())
    flagged = sum(len(n.unsupported) for n in narratives.values())
    stale = len(snapshot.stale_keys)
    if analysis.status == "unavailable":
        return Check("write_ups", "Pipeline outputs", label, WARN,
                     f"The last analyze couldn't reach LM Studio ({analysis.reason}); earlier write-ups are kept",
                     "Run analyze again once the Local AI checks pass")
    if stale:
        return Check("write_ups", "Pipeline outputs", label, WARN,
                     f"{stale} of {len(narratives)} were written for earlier numbers", "Run analyze")
    if written < len(narratives):
        return Check("write_ups", "Pipeline outputs", label, WARN,
                     f"{len(narratives) - written} of {len(narratives)} couldn't be written", "Run analyze again")
    figures = "every figure found in the data" if flagged == 0 else f"{flagged} figures not found in the data"
    return Check("write_ups", "Pipeline outputs", label, OK,
                 f"{written} of {len(narratives)} written {ago(snapshot.analysed_at)}, {figures}")


def report_check(snapshot, out_dir: Path) -> Check:
    path = out_dir / "report.html"
    if not path.exists():
        return Check("report", "Pipeline outputs", "Report page", WARN, "Not written yet", "Run report")
    written = path.stat().st_mtime
    newer = [when.timestamp() for when in (snapshot.modelled_at, snapshot.analysed_at) if when is not None]
    if newer and max(newer) > written + 1:
        return Check("report", "Pipeline outputs", "Report page", WARN,
                     f"Written {ago(written)}, before the latest model or write-ups", "Run report")
    return Check("report", "Pipeline outputs", "Report page", OK, f"Written {ago(written)} (out/report.html)")


def llm_cache_check(cache_dir: Path) -> Check:
    saved = len(list(cache_dir.glob("*.json"))) if cache_dir.exists() else 0
    return Check("llm_cache", "Pipeline outputs", "Saved AI answers", INFO,
                 f"{saved:,} saved: if no fact has changed, analyze replays them without LM Studio")


# --- Live season and players ---

def live_check(state) -> Check:
    label = f"Live season ({config.season_label(config.LIVE_SEASON)})"
    if state is None:
        return Check("live", "Live season and players", label, INFO, "Not part of this app")
    if state.status == "loading":
        return Check("live", "Live season and players", label, INFO, "Loading from UEFA…")
    if state.snapshot is None:
        return Check("live", "Live season and players", label, WARN, f"Unavailable ({state.message or 'no data'})",
                     "Check the UEFA feeds, then refresh the live season")
    snapshot = state.snapshot
    detail = (f"{len(snapshot.rows)} teams, {snapshot.finished_matches} finished matches, "
              f"fetched {ago(snapshot.fetched_at)}")
    if state.status == "stale":
        return Check("live", "Live season and players", label, WARN, f"{detail}; UEFA couldn't be reached since",
                     "Refresh the live season once UEFA answers")
    return Check("live", "Live season and players", label, OK, detail)


def player_index_check(players) -> Check:
    label = "Player index"
    if players is None:
        return Check("player_index", "Live season and players", label, INFO, "Not part of this app")
    status = players.index_status()
    done, of = status["squads"]["done"], status["squads"]["of"]
    if status["running"]:
        return Check("player_index", "Live season and players", label, INFO, f"Building: {done} of {of} squads")
    if status["complete"]:
        return Check("player_index", "Live season and players", label, OK,
                     f"All {of} squads cached, so histories cover every club")
    return Check("player_index", "Live season and players", label, WARN,
                 f"{done} of {of} squads cached, so player histories only cover those",
                 "Build the player index (about a minute and a half)")


# --- Dashboard ---

def frontend_check(web_dir: Path) -> Check:
    index = web_dir / "dist" / "index.html"
    if not index.exists():
        return Check("frontend", "Dashboard", "Dashboard build", FAIL, "web/dist is missing",
                     "Run `cd web && npm install && npm run build`")
    sources = [p for p in (web_dir / "src").rglob("*") if p.is_file()] + [web_dir / "index.html"]
    newest = max((p.stat().st_mtime for p in sources if p.exists()), default=0)
    built = index.stat().st_mtime
    if newest > built + 1:
        return Check("frontend", "Dashboard", "Dashboard build", WARN, "The frontend changed after the last build",
                     "Run `cd web && npm run build`, then reload this page")
    return Check("frontend", "Dashboard", "Dashboard build", OK, f"Built {ago(built)}")


def run_checks(services, llm_model: str = config.LLM_MODEL, get: Get = http_get, run_lms: RunLms | None = None,
               raw_dir: Path = config.RAW_DIR, out_dir: Path = config.OUT_DIR,
               cache_dir: Path = config.LLM_CACHE_DIR, web_dir: Path = config.WEB_DIR) -> dict:
    """Every check, grouped and in a stable order, with a summary count per status."""
    run_lms = run_lms or LMStudio(model=llm_model)._run_lms
    cli = lms_cli()
    slow = [lambda: lm_server(get, cli.status == OK), lambda: lm_model_downloaded(run_lms, llm_model),
            lambda: lm_model_loaded(run_lms, llm_model)]
    slow += [lambda probe=probe: uefa_feed(*probe, get) for probe in uefa_probes(raw_dir)]
    with ThreadPoolExecutor(max_workers=len(slow)) as pool:
        slow_results = list(pool.map(lambda check: check(), slow))
    snapshot = services.store.snapshot
    live_state = services.live.current() if services.live is not None else None
    checks = [cli, *slow_results, raw_cache(raw_dir), dataset_check(snapshot), model_check(snapshot),
              write_ups_check(snapshot), report_check(snapshot, out_dir), llm_cache_check(cache_dir),
              live_check(live_state), player_index_check(services.players), frontend_check(web_dir)]
    checks.sort(key=lambda check: GROUPS.index(check.group))  # stable, so each group keeps its own order
    summary = {status: sum(check.status == status for check in checks) for status in (OK, WARN, FAIL, INFO)}
    return {"checked_at": datetime.now(timezone.utc).isoformat(), "model": llm_model, "summary": summary,
            "checks": [asdict(check) for check in checks]}
