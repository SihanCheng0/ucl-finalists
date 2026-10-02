"""Retake the README screenshots in docs/images from a running UCL Lab (`make web`), in light mode.

    uv run --with websocket-client python scripts/screenshots.py [out_dir]

Drives headless Chrome through its DevTools protocol, so each shot waits until its screen has data. Set CHROME to
the browser's path if it isn't in the default macOS location.
"""
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket  # websocket-client

CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
LAB = "http://localhost:8787"
PORT = 9333
SHOTS = [  # name, hash route, a selector that appears once the screen has its data
    ("overview", "#/explore", ".metric"),
    ("explore", "#/explore?t=52280&s=2026", ".side .card"),  # Arsenal 2025-26
    ("compare", "#/compare?a=52280:2026&b=52747:2026", ".mirror-row"),  # against PSG, the final
    ("squad", "#/squad?t=52280&s=2026&stat=goals&p=250106939", ".player-panel table"),  # Saka selected
    ("pipeline", "#/pipeline", ".checks-head p"),
]


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as profile:
        chrome = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}",
                                   f"--user-data-dir={profile}", "--hide-scrollbars", "--window-size=1360,900",
                                   "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            capture(out)
        finally:
            chrome.terminate()
            chrome.wait()


def capture(out: Path) -> None:
    for _ in range(50):
        try:
            targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list"))
            page = next(t for t in targets if t["type"] == "page")
            break
        except Exception:  # noqa: BLE001 - Chrome is still starting
            time.sleep(0.2)
    else:
        sys.exit("Chrome didn't start; set CHROME to its path")
    ws = websocket.create_connection(page["webSocketDebuggerUrl"], suppress_origin=True)
    ids = iter(range(1, 10_000))

    def call(method: str, **params) -> dict:
        msg_id = next(ids)
        ws.send(json.dumps({"id": msg_id, "method": method, "params": params}))
        while True:
            reply = json.loads(ws.recv())
            if reply.get("id") == msg_id:
                return reply.get("result", {})

    call("Emulation.setDeviceMetricsOverride", width=1360, height=900, deviceScaleFactor=1, mobile=False)
    call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "light"}])
    for name, route, ready in SHOTS:
        call("Page.navigate", url=f"{LAB}/{route}")
        deadline, found = time.time() + 20, False
        while not found and time.time() < deadline:
            found = call("Runtime.evaluate", expression=f"!!document.querySelector({json.dumps(ready)})",
                         returnByValue=True)["result"].get("value")
            time.sleep(0.25)
        time.sleep(1.2 if name != "pipeline" else 2.5)  # let animations and the checks settle
        path = out / f"{name}.png"
        path.write_bytes(base64.b64decode(call("Page.captureScreenshot", format="png")["data"]))
        print(f"{name}: {'ok' if found else 'timed out waiting for ' + ready}, {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "docs/images"))
