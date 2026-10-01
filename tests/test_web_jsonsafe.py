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
