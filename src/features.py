"""
Feature engineering.

Main idea - the WATER BALANCE:
    water going into the network (2 sources + 3 tanks) = consumption + leaks
So a leak shows up as extra inflow compared to a normal day at the same time.

Steps:
1. net_inflow = sum of flows out of the sources and tanks.
2. residual = net_inflow - normal net_inflow for that 15-min slot of the day
   (the "baseline profile", learned from normal training data).
3. Smooth the residual over 15 min, 1 h and 3 h. A short spike (someone filling a
   pool) and a slow, steady increase (a leak) look different across these windows.

What we tried first: residuals of all 8 pressure sensors as extra features.
They worked in winter data but in the summer data tank levels swing a lot, the
pressures became noisy and the model got much worse (F1 dropped below 0.2).
So pressure is now only used to suggest WHERE the leak might be (see model.explain).
"""
import pandas as pd

TIME_COL = "timestamp"
GROUP_COL = "scenario_id"


def add_time_slot(df, timestep_min=15):
    ts = pd.to_datetime(df[TIME_COL])
    return (ts.dt.hour * 60 + ts.dt.minute) // timestep_min


def add_water_balance(df, inflow_pipes):
    df = df.copy()
    df["net_inflow"] = df[[f"F_{p}" for p in inflow_pipes]].sum(axis=1)
    return df


def fit_baseline(df_normal, cols, timestep_min=15):
    """Median value of each column for every time slot of the day (96 slots for 15 min)."""
    slots = add_time_slot(df_normal, timestep_min)
    baseline = df_normal[cols].groupby(slots).median()
    expected = 24 * 60 // timestep_min
    if len(baseline) < expected:
        # fill gaps if some slots were missing in training data
        baseline = baseline.reindex(range(expected)).interpolate(limit_direction="both")
    return baseline


def residuals(df, baseline, cols, timestep_min=15):
    slots = add_time_slot(df, timestep_min)
    expected = baseline.loc[slots.values, cols].values
    return pd.DataFrame(df[cols].values - expected, columns=cols, index=df.index)


def _groups(df):
    # rolling windows must not mix different scenarios; live data is one stream
    if GROUP_COL in df.columns:
        return df[GROUP_COL]
    return pd.Series(0, index=df.index)


def make_features(df, baseline, windows=(1, 4, 12), timestep_min=15):
    """df must already have net_inflow (see add_water_balance)."""
    res = residuals(df, baseline, ["net_inflow"], timestep_min)["net_inflow"]
    groups = _groups(df)
    feats = {}
    for w in windows:
        name = f"inflow_resid_{w * timestep_min}min"
        feats[name] = res.groupby(groups).transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
    return pd.DataFrame(feats, index=df.index)
