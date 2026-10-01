"""`ucl web`: serve UCL Lab on 127.0.0.1 and open it in the browser (spec §4, "Process")."""
from __future__ import annotations

import argparse
import os
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


def make_server(settings, bus):
    """A uvicorn server that ends the event streams before it waits for open connections, so a Ctrl-C with a
    browser tab open returns at once (the lifespan shutdown only runs after that wait)."""
    import uvicorn

    class Server(uvicorn.Server):
        async def shutdown(self, sockets=None):
            bus.close()
            await super().shutdown(sockets)

    return Server(settings)


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
    from .app import create_app
    from .services import build_services

    services = build_services(llm_model=args.llm_model)
    server = make_server(uvicorn_config(create_app(services, dist_dir=dist), args.port), services.bus)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"UCL Lab: {url} (Ctrl-C to stop)")
    if not args.no_open:
        threading.Thread(target=open_when_up, args=(server, url), daemon=True).start()
    try:
        server.run()
    except KeyboardInterrupt:  # uvicorn re-raises the signal after its graceful shutdown
        pass
    # A pipeline run, a live refresh or a squad refresh may still be working through a pool of UEFA requests,
    # and Python would wait for every queued one before exiting. The server is already down, so leave now.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
