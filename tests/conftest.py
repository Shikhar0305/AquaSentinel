"""Test data: a small real WNTR simulation (8 scenarios, a few seconds)."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_params  # noqa: E402
from src.data import simulate, validate  # noqa: E402
from src.model import build_model  # noqa: E402


@pytest.fixture(scope="session")
def params():
    p = copy.deepcopy(load_params())
    p["simulation"]["leak_area"] = [0.002, 0.004]   # clearly visible leaks keep the tests stable
    return p


@pytest.fixture(scope="session")
def raw_data(params):
    return simulate(params, n_normal=4, n_leak=4, demand_multiplier=1.0, seed=1, prefix="test")


@pytest.fixture(scope="session")
def sim_data(params, raw_data):
    return validate(raw_data, params)[0]   # fill power-cut gaps, like the real pipeline


@pytest.fixture(scope="session")
def model(params, sim_data):
    return build_model(params).fit(sim_data[sim_data["leak"] == 0])
