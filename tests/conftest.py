import pytest
from synthetic import make_synthetic_dataset

from ucl import config

config.BOOTSTRAP_SAMPLES = 60  # keep test runs fast; production uses the config default


@pytest.fixture(autouse=True)
def local_model_by_default(monkeypatch):
    """Tests ask LM Studio unless they choose OpenRouter themselves, whatever keys this shell happens to export."""
    monkeypatch.delenv(config.LLM_PROVIDER_ENV, raising=False)
    monkeypatch.delenv(config.OPENROUTER_KEY_ENV, raising=False)


@pytest.fixture
def synthetic_ds():
    return make_synthetic_dataset()


@pytest.fixture(scope="session")
def built():
    """Six synthetic seasons with model results, shared by the facts, analyst and report tests."""
    from ucl import model  # imported lazily: model.py is written after this file

    ds = make_synthetic_dataset(seasons=tuple(range(2021, 2027)))
    return ds, model.run(ds.team_seasons, ds.finals, ds.features)
