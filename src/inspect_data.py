
from src.data_loader import load_dataset


def inspect_dataset(data):
    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)

    print("\nShape:", data.shape)

    print("\nColumns:")
    print(data.columns.tolist())

    print("\nNumber of scenarios (zones):")
    print(data["scenario_id"].nunique())

    print("\nScenarios with a leak:")
    print(data.groupby("scenario_id")["leak"].max().value_counts())

    print("\nRows per scenario:")
    print(data.groupby("scenario_id").size().describe())

    print("\nMissing values:")
    print(data.isnull().sum())

    print("\nLeak label share (1 = leak):")
    print(data["leak"].value_counts(normalize=True).round(3))

    print("\nTimestamp intervals (seconds):")
    intervals = (
        data.sort_values(["scenario_id", "timestamp"])
        .groupby("scenario_id")["timestamp"]
        .diff()
        .dropna()
    )
    print(intervals.value_counts().head())

    print("\nShare of time the supply valve is open:")
    print(round(data["valve_open"].mean(), 3))

    normal = data[data["leak"] == 0]

    print("\nInflow (L/s) without leaks, by valve status (0 = closed, 1 = open):")
    print(normal.groupby("valve_open")["inflow_lps"].describe().round(1))

    print("\nInflow (L/s) with a leak, by valve status:")
    print(data[data["leak"] == 1].groupby("valve_open")["inflow_lps"].describe().round(1))

    print("\nLeak hole sizes (m^2):")
    leaks = data[data["leak_area"] > 0].groupby("scenario_id")["leak_area"].first()
    print(leaks.describe())


if __name__ == "__main__":
    dataset = load_dataset()
    inspect_dataset(dataset)
