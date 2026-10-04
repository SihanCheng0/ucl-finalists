"""The browser plays head-to-heads with a port of forecast.head_to_head (web/src/lib/h2h.ts), tested against
web/src/lib/h2h.fixture.json. This keeps that fixture true to the Python: after changing the math, rerun
`uv run python scripts/h2h_fixture.py` and make the port agree again."""
import json

import pytest

from ucl import config, forecast
from ucl.forecast import Goals, Models

FIXTURE = config.WEB_DIR / "src" / "lib" / "h2h.fixture.json"


def close(actual, expected, path="result"):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), path
        for key in expected:
            close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected)):
            close(a, e, f"{path}[{i}]")
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-15), path
    else:
        assert actual == expected, path


def test_the_fixture_matches_forecast_head_to_head():
    fixture = json.loads(FIXTURE.read_text())
    models = Models(*(Goals(**fixture["models"][stage]) for stage in ("league", "early", "late")))
    assert (fixture["home"], fixture["max_goals"]) == (forecast.PARAMS.home, forecast.MAX_GOALS)
    for case in fixture["cases"]:
        actual = forecast.head_to_head(case["a"], case["b"], models, fixture["home"], case["venue"])
        close(json.loads(json.dumps(actual)), case["expected"], f"{case['a']} v {case['b']} ({case['venue']})")


def test_equally_likely_scores_come_in_the_grids_reading_order():
    models = Models(Goals(0.28, 0.95), Goals(0.25, 0.8), Goals(0.15, 0.7))
    scores = forecast.head_to_head(1650.0, 1650.0, models, forecast.PARAMS.home, "neutral")["likely_scores"]
    assert [(s["a"], s["b"]) for s in scores] == [(1, 1), (0, 1), (1, 0)]
