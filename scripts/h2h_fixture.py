"""Write web/src/lib/h2h.fixture.json: head-to-heads worked out by forecast.head_to_head, which the browser's port
(web/src/lib/h2h.ts) is tested against. Run it again after changing the head-to-head math:

    uv run python scripts/h2h_fixture.py
"""
from __future__ import annotations

import json

from ucl import config, forecast
from ucl.forecast import Goals, Models

PATH = config.WEB_DIR / "src" / "lib" / "h2h.fixture.json"
# goals models like the fitted ones (the deeper the round, the less a rating gap is worth)
MODELS = Models(Goals(0.28, 0.95), Goals(0.25, 0.8), Goals(0.15, 0.7))
GAPS = (-420.5, -150.25, -30.0, 0.0, 12.75, 90.0, 260.5, 610.0)  # 0.0 has equally likely scores (1-0 and 0-1)
B = 1650.0


def build() -> dict:
    cases = [{"a": B + gap, "b": B, "venue": venue,
              "expected": forecast.head_to_head(B + gap, B, MODELS, forecast.PARAMS.home, venue)}
             for gap in GAPS for venue in ("neutral", "a", "b")]
    models = {"league": vars(MODELS.league), "early": vars(MODELS.early), "late": vars(MODELS.late)}
    return {"home": forecast.PARAMS.home, "max_goals": forecast.MAX_GOALS, "models": models, "cases": cases}


if __name__ == "__main__":
    PATH.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {PATH.relative_to(config.ROOT)}")
