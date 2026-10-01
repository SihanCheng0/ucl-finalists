"""The HTTP layer (spec §5): JSON routes over the snapshot, the live season, squads and the runner, server-sent
events, and the built SPA. Shaping lives in queries.py; this file maps requests to it and every failure to
{"error": {code, message}}."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import config
from . import queries
from .events import HEARTBEAT_S
from .jsonsafe import to_jsonable
from .pipeline import BadRun, RunInProgress
from .services import Services

NOT_READY = "Run the pipeline first: processed data or model outputs are missing."
HTTP_CODES = {404: "not_found", 405: "method_not_allowed"}


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


class SpaFiles(StaticFiles):
    """The built SPA. Pages go out with Cache-Control: no-cache, so a rebuild shows at the next load; the hashed
    assets they point to can be cached."""

    def file_response(self, full_path, stat_result, scope, status_code=200):
        response = super().file_response(full_path, stat_result, scope, status_code)
        if str(full_path).endswith(".html"):
            response.headers["Cache-Control"] = "no-cache"
        return response


class RunBody(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a typo like "stage" must not start every stage

    stages: list[str] | None = None
    skip_ai: bool = False
    refresh_live: bool = False


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


def create_app(services: Services, dist_dir: Path | None = None, heartbeat: float = HEARTBEAT_S) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        services.bus.close()  # a backstop: `ucl web` closes the bus before uvicorn waits for connections

    app = FastAPI(title="UCL Lab", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError) -> JSONResponse:
        return _error(exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0]
        where = ".".join(part for part in first.get("loc", ())[1:] if isinstance(part, str))
        return _error(422, "invalid_request", f"{where}: {first['msg']}" if where else first["msg"])

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error(exc.status_code, HTTP_CODES.get(exc.status_code, "http_error"), str(exc.detail))

    @app.exception_handler(Exception)
    async def crash(request: Request, exc: Exception) -> JSONResponse:
        return _error(500, "internal", f"{type(exc).__name__}: {exc}")

    def ready():
        snapshot = services.store.snapshot  # read once per request
        if not snapshot.ready:
            raise ApiError(503, "not_ready", NOT_READY)
        return snapshot

    def live_state(revalidate: bool = True):
        """The live season as it is now; a request may start a background refresh, never wait for one."""
        if services.live is None:
            return None
        if revalidate:
            services.live.refresh_in_background()
        return services.live.current()

    def live_fields(state) -> dict:
        if state is None:
            return {"available": False, "stale": False, "fetched_at": None}
        return {"available": state.available, "stale": state.status == "stale", "fetched_at": state.fetched_at}

    @app.get("/api/meta")
    def meta():
        from .players import STAT_META

        return queries.meta(services.store.snapshot, services.runner.stage_names, live_state(),
                            STAT_META if services.players is not None else [])

    @app.get("/api/summary")
    def summary():
        return queries.summary(ready())

    @app.get("/api/teams")
    def teams(q: str = ""):
        snapshot = ready()
        return queries.search(services.team_seasons(snapshot, live_state(revalidate=False)), q)

    @app.get("/api/teams/{team_id}/seasons/{season}")
    def profile(team_id: str, season: int):
        snapshot, state = ready(), live_state(revalidate=season == config.LIVE_SEASON)
        try:
            return queries.profile(snapshot, team_id, season, services.team_seasons(snapshot, state),
                                   live_fields(state))
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
            return queries.compare(snapshot, *picks, team_seasons=services.team_seasons(snapshot, live_state()))
        except queries.NotFound as exc:
            raise ApiError(404, "not_found", str(exc)) from exc

    @app.get("/api/squads/{team_id}/{season}")
    def squad(team_id: str, season: int):
        from .players import SquadUnavailable

        snapshot = ready()
        if services.players is None:
            raise ApiError(503, "players_unavailable", "Squads aren't available in this app")
        every = services.team_seasons(snapshot, live_state(revalidate=season == config.LIVE_SEASON))
        hit = every[(every["team_id"] == team_id) & (every["season"] == season)]
        if hit.empty:
            raise ApiError(404, "not_found", f"no team {team_id} in {config.season_label(season)}")
        try:
            squad = services.players.squad(team_id, season)
        except SquadUnavailable as exc:
            raise ApiError(503, "uefa_unreachable", str(exc)) from exc
        row = hit.iloc[0]
        return to_jsonable({"team_id": team_id, "name": row["team_display"], "season": season,
                            "label": config.season_label(season), "live": season == config.LIVE_SEASON, **squad})

    @app.get("/api/players/{player_id}")
    def player(player_id: str):
        if services.players is None:
            raise ApiError(503, "players_unavailable", "Player histories aren't available in this app")
        history = services.players.history(player_id)
        if history is None:
            raise ApiError(404, "not_found", f"no player {player_id} in the cached squads")
        live_entry = next((h for h in history["history"] if h["live"]), None)
        state = live_state(revalidate=False)
        return to_jsonable({**history, "stale": bool(live_entry and state and state.status == "stale"),
                            "fetched_at": state.fetched_at if live_entry and state else None})

    @app.post("/api/pipeline/runs", status_code=202)
    def start_run(body: RunBody | None = None):
        body = body or RunBody()
        stages = body.stages
        if body.refresh_live:
            stages = [*(stages if stages is not None else services.runner.default_stages), "live"]
        try:
            return {"run_id": services.runner.start(stages, skip_ai=body.skip_ai)}
        except RunInProgress as exc:
            raise ApiError(409, "run_in_progress", str(exc)) from exc
        except BadRun as exc:
            raise ApiError(422, "bad_run", str(exc)) from exc

    @app.get("/api/pipeline/state")
    def pipeline_state():
        return services.runner.state()

    @app.get("/api/pipeline/events")
    def pipeline_events(request: Request):
        resume = request.headers.get("last-event-id")

        def frames():
            # subscribe once streaming starts, so a client that leaves before then never holds a queue
            yield from services.bus.stream(services.bus.subscribe(resume), heartbeat)

        return StreamingResponse(frames(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    if dist_dir is not None:
        app.mount("/", SpaFiles(directory=dist_dir, html=True), name="spa")
    return app
