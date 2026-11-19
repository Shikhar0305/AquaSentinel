"""Shared test data: a small real WNTR simulation (6 scenarios, takes a few seconds)."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.simulate import simulate  # noqa: E402
from src.utils import load_params  # noqa: E402


@pytest.fixture(scope="session")
def params():
    p = copy.deepcopy(load_params())
    p["simulation"]["n_normal_scenarios"] = 4
    p["simulation"]["n_leak_scenarios"] = 4
    # make leaks clearly visible so tests are stable
    p["simulation"]["leak_area_min"] = 0.002
    p["simulation"]["leak_area_max"] = 0.004
    return p


@pytest.fixture(scope="session")
def sim_data(params):
    return simulate(params, 4, 4, 1.0, seed=1, prefix="test")


@pytest.fixture(scope="session")
def trained_model(params, sim_data):
    from src.model import build_model
    model = build_model(params)
    model.fit(sim_data[sim_data["leak"] == 0])
    return model
