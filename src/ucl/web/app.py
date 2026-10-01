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
    refresh_live: bool = False


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


def create_app(services: Services, dist_dir: Path | None = None, heartbeat: float = HEARTBEAT_S) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        services.bus.close()  # ends open event streams, so shutdown doesn't wait on them

    app = FastAPI(title="UCL Lab", lifespan=lifespan)
    app.state.services = services

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
        return queries.meta(services.store.snapshot, services.runner.stage_names)

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
