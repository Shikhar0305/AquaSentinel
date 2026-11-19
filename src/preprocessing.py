
import numpy as np
import pandas as pd


TARGET = "leak"

FEATURES = [
    "inflow_lps",
    "valve_open",
    "valve_minutes",
    "inflow_mean_4",
    "inflow_mean_12",
    "inflow_min_12",
    "inflow_change_4",
]


def validate_data(data):
    required_columns = [
        "scenario_id",
        "timestamp",
        "valve_open",
        "valve_minutes",
        "inflow_lps",
        "leak",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    if data[required_columns].isnull().any().any():
        raise ValueError("Missing values found in required columns")

    if data.duplicated(["scenario_id", "timestamp"]).any():
        raise ValueError("Duplicate scenario timestamp records found")

    data = data.sort_values(
        ["scenario_id", "timestamp"]
    ).copy()

    intervals = (
        data.groupby("scenario_id")["timestamp"]
        .diff()
        .dropna()
    )

    if not intervals.eq(900).all():
        raise ValueError("Unexpected timestamp intervals detected")

    return data


def create_features(data):
    data = validate_data(data)

    grouped = data.groupby("scenario_id")["inflow_lps"]

    # average inflow over the last hour (4 readings) and last 3 hours (12 readings)
    data["inflow_mean_4"] = grouped.transform(
        lambda values: values.rolling(4).mean()
    )

    data["inflow_mean_12"] = grouped.transform(
        lambda values: values.rolling(12).mean()
    )

    # lowest inflow in the last 3 hours:
    # when supply is off this should be small, a leak keeps it high
    data["inflow_min_12"] = grouped.transform(
        lambda values: values.rolling(12).min()
    )

    # change compared with one hour ago
    data["inflow_change_4"] = data["inflow_lps"] - grouped.shift(4)

    data = data.dropna(subset=FEATURES + [TARGET])

    return data


def scenario_split(data, test_ratio=0.3, seed=42):
    """
    Whole scenarios go either to training or to testing,
    so the model is tested on zones it has never seen.
    Leak and normal scenarios are split separately so both appear in the test set.
    """
    rng = np.random.default_rng(seed)

    has_leak = data.groupby("scenario_id")[TARGET].max()

    test_ids = []

    for ids in (has_leak[has_leak == 0].index, has_leak[has_leak == 1].index):
        ids = rng.permutation(ids)
        test_ids += list(ids[:max(1, round(len(ids) * test_ratio))])

    is_test = data["scenario_id"].isin(test_ids)

    train_data = data[~is_test]
    test_data = data[is_test]

    X_train = train_data[FEATURES]
    y_train = train_data[TARGET]

    X_test = test_data[FEATURES]
    y_test = test_data[TARGET]

    print("\nTrain scenarios:", train_data["scenario_id"].nunique())
    print("Test scenarios:", test_data["scenario_id"].nunique())
    print("Train rows:", len(X_train))
    print("Test rows:", len(X_test))
    print("Leak share in train:", round(y_train.mean(), 3))
    print("Leak share in test:", round(y_test.mean(), 3))

    return X_train, X_test, y_train, y_test


def to_frame(rows):
    return pd.DataFrame(rows)[FEATURES]
