"""The website: what every GET of the dashboard returns, saved as static JSON next to the website build of the
frontend (web/dist-site), in Vercel's Build Output layout. `ucl publish --run live,players` first runs those stages
with the dashboard's own runner and keeps the run's events, so the site's Pipeline screen replays the run.

The browser does the rest (web/src/staticApi.ts): it searches teams.json, builds a comparison from two profiles, and
plays head-to-heads from matchups.json."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from fastapi.encoders import jsonable_encoder
from fastapi.routing import APIRoute

from .. import config
from . import checks, queries
from .app import ApiError, create_app
from .forecasts import Unavailable
from .jsonsafe import to_jsonable
from .pipeline import BadRun
from .players import with_freshness
from .services import Services, build_services

SITE_BUILD = config.WEB_DIR / "dist-site"
OUTPUT_DIR = config.ROOT / ".vercel" / "output"
# Hashed assets never change; everything else keeps Vercel's default (revalidate on every visit).
VERCEL_CONFIG = {"version": 3, "routes": [
    {"src": "^/assets/(.*)$", "headers": {"cache-control": "public, max-age=31536000, immutable"}, "continue": True},
    {"handle": "filesystem"},
]}
Log = Callable[[str], None]


class PublishError(Exception):
    """The site can't be written: no frontend build, or no processed data to show."""


def player_shard(player_id: str) -> str:
    """Which file of data/players/ holds a player: the last two characters of the id (staticApi.ts agrees)."""
    return str(player_id)[-2:].rjust(2, "0")


def get_routes(app) -> dict[str, Callable]:
    """The app's GET handlers by path, so that each file is exactly what its route would send."""
    return {route.path: route.endpoint for route in app.routes
            if isinstance(route, APIRoute) and "GET" in route.methods}


def run_stages(services: Services, stages: list[str], log: Log) -> dict:
    """Run `stages` with the dashboard's runner and wait, echoing the log. Returns what the Pipeline screen replays:
    the runner's final state and every event of the run."""
    subscription = services.bus.subscribe()  # before the run starts, so that no event is missed
    try:
        services.runner.start(stages)
    except BadRun as exc:
        services.bus.unsubscribe(subscription)
        raise PublishError(str(exc)) from exc
    events = []

    def keep(event) -> None:
        events.append(event.to_dict())
        if event.type == "log":
            log(f"  {event.stage or 'run'}: {event.data['text']}")
        elif event.type == "stage" and event.data["status"] not in ("idle", "running"):
            log(f"{event.data['name']}: {event.data['status']}"
                + (f" ({event.data['message']})" if event.data.get("message") else ""))

    while services.runner.state()["running"]:
        event = subscription.get(timeout=0.5)
        if event is not None:
            keep(event)
    services.runner.wait()
    while (event := subscription.get(timeout=0)) is not None:  # whatever the run sent as it ended
        keep(event)
    services.bus.unsubscribe(subscription)
    return {"state": services.runner.state(), "events": events}


class Writer:
    def __init__(self, data_dir: Path):
        self.data_dir, self.files = data_dir, 0

    def save(self, relative: str, payload) -> None:
        path = self.data_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(jsonable_encoder(payload), ensure_ascii=False, separators=(",", ":"),
                                   allow_nan=False), encoding="utf-8")
        self.files += 1

    def save_route(self, relative: str, handler: Callable, **params) -> bool:
        """A page-wide view: when the route would answer with an error, the error goes in the file instead, with
        its status, and the site shows it the way the dashboard shows the API's."""
        try:
            self.save(relative, handler(**params))
            return True
        except ApiError as exc:
            self.save(relative, {"error": {"code": exc.code, "message": exc.message}, "status": exc.status})
            return False


def export(services: Services, data_dir: Path, record: dict, log: Log, web_build: str = "dist-site") -> dict:
    """Write data_dir and return counts: profiles, squads, players and the files written."""
    snapshot = services.store.snapshot
    if not snapshot.ready:
        raise PublishError("processed data or model outputs are missing: run the pipeline first")
    get = get_routes(create_app(services))
    live_state = services.live.current() if services.live is not None else None
    every = services.team_seasons(snapshot, live_state)
    out = Writer(data_dir)

    for name, path in (("meta", "/api/meta"), ("summary", "/api/summary"), ("forecast", "/api/forecast")):
        if not out.save_route(f"{name}.json", get[path]):
            log(f"{name}: the route answered with an error, which the site will show")
    out.save("teams.json", queries.search_index(every))
    try:
        if services.forecasts is None:
            raise Unavailable("Forecasts aren't available in this app")
        out.save("matchups.json", services.forecasts.matchups(snapshot, live_state))
    except Unavailable as exc:
        out.save("matchups.json", {"error": {"code": "forecast_unavailable", "message": str(exc)}, "status": 503})

    pairs = sorted({(str(t), int(s)) for t, s in zip(every["team_id"], every["season"])})
    profiles = sum(out.save_route(f"profiles/{team_id}/{season}.json", get["/api/teams/{team_id}/seasons/{season}"],
                                  team_id=team_id, season=season) for team_id, season in pairs)
    squads, players = 0, 0
    if services.players is not None:
        known = set(pairs)
        for season, team_id in sorted(services.players.cached()):
            if (team_id, season) not in known:
                continue
            try:
                out.save(f"squads/{team_id}/{season}.json",
                         get["/api/squads/{team_id}/{season}"](team_id=team_id, season=season))
                squads += 1
            except ApiError as exc:
                log(f"squad {team_id} {config.season_label(season)}: {exc.message}")
        shards: dict[str, dict] = defaultdict(dict)
        for player_id, history in services.players.histories().items():
            shards[player_shard(player_id)][player_id] = to_jsonable(with_freshness(services.players, history))
            players += 1
        for shard, histories in sorted(shards.items()):
            out.save(f"players/{shard}.json", histories)

    report = checks.run_checks(services, services.llm_model, web_build=web_build, public=True)
    out.save("checks.json", report)
    out.save("pipeline.json", {"published_at": datetime.now(timezone.utc).isoformat(), **record})
    return {"profiles": profiles, "squads": squads, "players": players, "files": out.files,
            "checks": report["summary"]}


def publish(out_dir: Path = OUTPUT_DIR, stages: list[str] | None = None, llm_model: str = config.LLM_MODEL,
            site_build: Path = SITE_BUILD, log: Log = print) -> dict:
    """Run `stages` (if any), then write the site to out_dir: config.json and static/ (the frontend and data/).
    `live` in the result says whether the live season made it in: without it the site would lose the current
    season, so the nightly job doesn't deploy it."""
    if not (site_build / "index.html").exists():
        shown = site_build.relative_to(config.ROOT) if site_build.is_relative_to(config.ROOT) else site_build
        raise PublishError(f"{shown} is missing: run `cd web && npm run build:site`")
    services = build_services(llm_model=llm_model, start_live=False, background=False)
    try:
        record = run_stages(services, stages, log) if stages else {"state": services.runner.state(), "events": []}
        if services.live is not None and services.live.current().snapshot is None:
            services.live.refresh()  # a run without the live stage still shows the season (cached copies first)
        static = out_dir / "static"
        if static.exists():
            shutil.rmtree(static)
        shutil.copytree(site_build, static)
        counts = export(services, static / "data", record, log, web_build=site_build.name)
        (out_dir / "config.json").write_text(json.dumps(VERCEL_CONFIG, indent=2) + "\n")
        live = services.live.current() if services.live is not None else None
    finally:
        services.bus.close()
    return {**counts, "live": live is not None and live.snapshot is not None,
            "live_status": live.status if live is not None else "unavailable"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ucl publish",
        description="Write the website: every page's data as static JSON beside web/dist-site, for Vercel.")
    parser.add_argument("--run", default="", metavar="STAGES",
                        help="stages to run first, comma separated (the nightly job runs live,players)")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR, help="output folder (default .vercel/output)")
    parser.add_argument("--llm-model", default=config.LLM_MODEL, help="model id for the analyze stage")
    args = parser.parse_args(argv)
    stages = [name.strip() for name in args.run.split(",") if name.strip()]
    try:
        result = publish(args.out, stages, llm_model=args.llm_model)
    except PublishError as exc:
        print(f"publish failed: {exc}", file=sys.stderr)
        return 1
    summary = result["checks"]
    print(f"wrote {result['files']:,} files to {args.out}: {result['profiles']} profiles, {result['squads']} squads, "
          f"{result['players']:,} players; checks {summary['ok']} ok, {summary['warn']} warn, {summary['fail']} fail; "
          f"live season {result['live_status']}")
    if not result["live"]:
        print("the live season is missing, so this site shouldn't replace the deployed one", file=sys.stderr)
        return 2
    return 0
