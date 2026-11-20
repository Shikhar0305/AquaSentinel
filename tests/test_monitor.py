import pandas as pd
import pytest

from monitor import validate_incoming_data


def _batch():
    return pd.DataFrame({
        "scenario_id": ["zone-01", "zone-01", "zone-01"],
        "timestamp": [0, 900, 1800],
        "valve_open": [0, 0, 0],
        "valve_minutes": [0, 15, 30],
        "inflow_lps": [15.0, 16.0, 17.0],
        "leak": [0, 0, 0],
    })


def test_validate_incoming_data_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_incoming_data(_batch().drop(columns=["leak"]))


def test_validate_incoming_data_rejects_duplicate_records():
    batch = pd.concat([_batch(), _batch().iloc[[1]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate scenario/timestamp"):
        validate_incoming_data(batch)
