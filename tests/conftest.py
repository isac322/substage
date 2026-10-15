import pytest

from substage import config, llm


@pytest.fixture(autouse=True)
def _isolated_memory(tmp_path_factory, monkeypatch):
    """Tests never see the developer's real translator-memory corpus."""
    path = tmp_path_factory.mktemp("memory") / "memory.jsonl"
    monkeypatch.setattr(config, "memory_path", lambda: path)
    return path

@pytest.fixture(autouse=True)
def _isolated_budget(tmp_path, monkeypatch, request):
    """Keep legacy fake-client tests inside a fresh prospective ledger."""
    if request.path.name == "test_budget.py":
        return
    from dataclasses import replace

    from substage import budget

    real_reserve = budget.reserve
    ledger = tmp_path / "budget"

    def reserve_for_test(settings, model, stage, upper_usd, *, provider=None):
        if request.node.function.__name__ == "test_unconfigured_budgeted_call_is_blocked_before_network":
            return real_reserve(settings, model, stage, upper_usd, provider=provider)
        if settings.experiment_id is None:
            settings = replace(settings, experiment_id="pytest-default")
        return real_reserve(settings, model, stage, upper_usd, provider=provider)

    monkeypatch.setattr(
        budget,
        "ledger_path",
        lambda settings: ledger / (settings.experiment_id or "pytest-default") / "budget.sqlite3",
    )
    monkeypatch.setattr(budget, "reserve", reserve_for_test)


@pytest.fixture(autouse=True)
def _no_proxy_network(monkeypatch):
    """Production defaults route text calls to the OMP proxies: a test that
    builds a real proxy client without a fake transport fails loudly instead
    of calling the network."""
    def _blocked(url, headers, body, timeout):
        raise RuntimeError(f"network call in a test: {url}")

    monkeypatch.setattr(llm, "urllib_transport", _blocked)
