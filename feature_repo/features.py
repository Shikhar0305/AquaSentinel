"""Feast definitions: one entity (the network zone / scenario) and one feature view."""
from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32
from feast.value_type import ValueType

PARQUET = str(Path(__file__).resolve().parent.parent / "data" / "features" / "sensor_features.parquet")

zone = Entity(name="zone", join_keys=["scenario_id"], value_type=ValueType.STRING,
              description="A metered zone of the network (one simulated scenario)")

sensor_source = FileSource(
    name="sensor_features_source",
    path=PARQUET,
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

sensor_features = FeatureView(
    name="sensor_features",
    entities=[zone],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="net_inflow", dtype=Float32, description="Total inflow from sources and tanks (L/s)"),
        Field(name="pressure_mean", dtype=Float32, description="Average of the pressure sensors (m)"),
        Field(name="pressure_min", dtype=Float32, description="Lowest pressure sensor reading (m)"),
    ],
    source=sensor_source,
    online=True,
    tags={"team": "aquasentinel"},
)
