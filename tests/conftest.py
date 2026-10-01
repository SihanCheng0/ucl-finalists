import pytest
from synthetic import make_synthetic_dataset


@pytest.fixture
def synthetic_ds():
    return make_synthetic_dataset()


@pytest.fixture(scope="session")
def built():
    """Six synthetic seasons with model results, shared by the facts, analyst and report tests."""
    from ucl import model  # imported lazily: model.py is written after this file

    ds = make_synthetic_dataset(seasons=tuple(range(2021, 2027)))
    return ds, model.run(ds.team_seasons, ds.finals, ds.features)
