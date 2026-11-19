"""
AquaSentinel control room (Streamlit).

Replays simulated sensor data through the API, as if readings were arriving live
every 15 minutes, and shows leak alerts, the network map, model registry and drift.

Run:  streamlit run dashboard/app.py
"""
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.utils import load_params, sensor_columns  # noqa: E402

st.set_page_config(page_title="AquaSentinel", page_icon="💧", layout="wide")

PARAMS = load_params()
SENSORS = sensor_columns(PARAMS)
API_URL = os.getenv("API_URL", "http://localhost:8000")
WINDOW = 16  # readings sent per request (4 h of history)

BLUE, RED, AMBER, GREEN, GREY = "#38bdf8", "#f43f5e", "#f59e0b", "#22c55e", "#64748b"

# ---------------------------------------------------------------- styling
st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.hero {display:flex; align-items:center; justify-content:space-between;
       background: linear-gradient(120deg, #0e7490 0%, #1e3a8a 100%);
       padding: 18px 26px; border-radius: 16px; margin-bottom: 18px;}
.hero h1 {color:white; font-size: 1.9rem; margin:0;}
.hero p {color:#cbd5e1; margin:0; font-size: .95rem;}
.chip {display:inline-block; padding:5px 12px; border-radius:999px; font-size:.8rem;
       margin-left:6px; background: rgba(255,255,255,.12); color:white;}
.card {background:#111a2e; border:1px solid #1f2b45; border-radius:14px; padding:14px 18px;}
.card .label {color:#94a3b8; font-size:.8rem; text-transform:uppercase; letter-spacing:.05em;}
.card .value {color:#f8fafc; font-size:1.6rem; font-weight:700; margin-top:2px;}
.card .sub {color:#64748b; font-size:.8rem;}
.status-ok {border-left: 5px solid #22c55e;}
.status-leak {border-left: 5px solid #f43f5e; background: #2a1020;}
.status-warn {border-left: 5px solid #f59e0b;}
</style>
""", unsafe_allow_html=True)


def card(label, value, sub="", status=""):
    st.markdown(f"""<div class="card {status}"><div class="label">{label}</div>
                <div class="value">{value}</div><div class="sub">{sub}</div></div>""",
                unsafe_allow_html=True)


# ---------------------------------------------------------------- API helpers
def api_get(path, key):
    try:
        r = requests.get(f"{API_URL}{path}", headers={"X-API-Key": key}, timeout=5)
        return r.json() if r.ok else None
    except requests.RequestException:
        return None


def api_predict(rows, key):
    body = {"readings": [{"timestamp": str(r["timestamp"]), "sensors": {s: float(r[s]) for s in SENSORS}}
                         for _, r in rows.iterrows()]}
    r = requests.post(f"{API_URL}/predict", json=body, headers={"X-API-Key": key}, timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data
def load_data(name):
    return pd.read_csv(ROOT / "data" / "raw" / name, parse_dates=["timestamp"])


@st.cache_data
def network_layout():
    import wntr
    wn = wntr.network.WaterNetworkModel(PARAMS["simulation"]["network"])
    nodes = {n: wn.get_node(n).coordinates for n in wn.node_name_list}
    edges = [(link.start_node_name, link.end_node_name) for _, link in wn.links()]
    return nodes, edges


def network_map(suspected=None, true_leak=None, show_truth=False):
    nodes, edges = network_layout()
    ex, ey = [], []
    for a, b in edges:
        ex += [nodes[a][0], nodes[b][0], None]
        ey += [nodes[a][1], nodes[b][1], None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#334155", width=1.5),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[c[0] for c in nodes.values()], y=[c[1] for c in nodes.values()],
                             mode="markers", marker=dict(size=4, color=GREY), text=list(nodes),
                             hovertemplate="junction %{text}<extra></extra>", showlegend=False))
    ps = PARAMS["sensors"]["pressure"]
    fig.add_trace(go.Scatter(x=[nodes[n][0] for n in ps], y=[nodes[n][1] for n in ps], mode="markers",
                             marker=dict(size=11, color=BLUE, symbol="diamond"), text=ps,
                             hovertemplate="pressure sensor %{text}<extra></extra>", name="pressure sensor"))
    if suspected and suspected in nodes:
        fig.add_trace(go.Scatter(x=[nodes[suspected][0]], y=[nodes[suspected][1]], mode="markers",
                                 marker=dict(size=34, color="rgba(244,63,94,.25)", line=dict(color=RED, width=2)),
                                 name="suspected area", hoverinfo="skip"))
    if show_truth and true_leak and true_leak in nodes:
        fig.add_trace(go.Scatter(x=[nodes[true_leak][0]], y=[nodes[true_leak][1]], mode="markers",
                                 marker=dict(size=14, color=AMBER, symbol="x"), name="actual leak (simulation)"))
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                      legend=dict(orientation="h", y=-0.02))
    return fig


def score_chart(hist, threshold, leak_start=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["timestamp"], y=hist["anomaly_score"], name="anomaly score",
                             line=dict(color=BLUE, width=2.5)))
    fig.add_hline(y=threshold, line=dict(color=RED, dash="dash"), annotation_text="threshold")
    alerts = hist[hist["leak_alert"]]
    fig.add_trace(go.Scatter(x=alerts["timestamp"], y=alerts["anomaly_score"], mode="markers",
                             marker=dict(color=RED, size=7), name="leak alert"))
    if leak_start is not None:
        fig.add_vline(x=leak_start, line=dict(color=AMBER, dash="dot"))
    fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.15))
    return fig


def inflow_chart(rows, hist):
    total = rows[[f"F_{p}" for p in PARAMS["sensors"]["inflow"]]].sum(axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rows["timestamp"], y=total, name="total inflow", line=dict(color=BLUE, width=2)))
    fig.add_trace(go.Bar(x=hist["timestamp"], y=hist["extra_inflow_lps"].clip(lower=0), name="extra vs normal (model)",
                         marker_color=[RED if a else GREY for a in hist["leak_alert"]], yaxis="y2", opacity=0.7))
    fig.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0), template="plotly_dark",
                      title="Water balance: inflow from sources + tanks", bargap=0,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis_title="L/s",
                      yaxis2=dict(overlaying="y", side="right", title="extra L/s", showgrid=False),
                      legend=dict(orientation="h", y=-0.25, font=dict(size=10)))
    return fig


def sensor_chart(rows, cols, title, ytitle):
    fig = go.Figure()
    for c in cols:
        fig.add_trace(go.Scatter(x=rows["timestamp"], y=rows[c], name=c, mode="lines", line=dict(width=1.6)))
    fig.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0), template="plotly_dark", title=title,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis_title=ytitle,
                      legend=dict(orientation="h", y=-0.25, font=dict(size=10)))
    return fig


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    api_key = st.text_input("API key", value=os.getenv("DASHBOARD_API_KEY", "operator-dev-key"), type="password")
    dataset = st.radio("Data", ["Winter (training season)", "Summer (+30% demand)"])
    data = load_data("sensor_data.csv" if dataset.startswith("Winter") else "drift_data.csv")
    scen = data.groupby("scenario_id").agg(leak=("leak", "max"), node=("leak_node", "first"),
                                           area=("leak_area", "first"))
    labels = {sid: f"{sid}  ·  {'💧 leak' if r.leak else 'normal'}" for sid, r in scen.iterrows()}
    default = int(scen["area"].values.argmax())  # start with the biggest leak - easiest to see
    scenario = st.selectbox("Scenario", list(labels), index=default, format_func=labels.get)
    speed = st.slider("Replay speed (readings / sec)", 2, 40, 12)
    show_truth = st.toggle("Show actual leak location", value=False,
                           help="Ground truth from the simulation - an operator would not know this.")

health = api_get("/health", api_key)
info = api_get("/model/info", api_key)
drift = api_get("/monitoring/drift", api_key)

api_chip = "🟢 API online" if health else "🔴 API offline"
ver_chip = f"model v{info['champion']['version']}" if info and info.get("champion") else "no model"
drift_chip = ("⚠️ drift detected" if drift and drift.get("drift_detected") else "✅ no drift") if drift else ""
st.markdown(f"""<div class="hero"><div><h1>💧 AquaSentinel</h1>
<p>AI leak detection for water distribution networks · EPANET Net3 · 15-min sensor readings</p></div>
<div><span class="chip">{api_chip}</span><span class="chip">{ver_chip}</span>
<span class="chip">{drift_chip}</span></div>
</div>""", unsafe_allow_html=True)

if not health:
    st.error(f"Cannot reach the API at {API_URL}. Start it with `uvicorn api.main:app --port 8000`.")
    st.stop()

tab_live, tab_models, tab_drift, tab_ops = st.tabs(["📡 Live monitor", "🧠 Models & lineage",
                                                    "📊 Data drift", "🛠️ Operations"])

# ---------------------------------------------------------------- live monitor
with tab_live:
    rows = data[data["scenario_id"] == scenario].reset_index(drop=True)
    true_node = scen.loc[scenario, "node"]
    true_node = str(int(true_node)) if pd.notna(true_node) and str(true_node) != "" else None
    leak_start = rows.loc[rows["leak"] == 1, "timestamp"].min() if rows["leak"].any() else None

    c1, c2, _ = st.columns([1, 1, 4])
    play = c1.button("▶ Replay scenario", type="primary", width="stretch")
    if c2.button("↺ Reset", width="stretch"):
        st.session_state.pop("hist", None)

    kpi = st.columns(4)
    kpi_slots = [k.empty() for k in kpi]
    banner = st.empty()
    left, right = st.columns([3, 2])
    with left:
        score_slot = st.empty()
        flow_slot = st.empty()
    with right:
        st.markdown("**Network map**")
        map_slot = st.empty()
        pres_slot = st.empty()

    def render(hist, upto, threshold):
        last = hist.iloc[-1]
        alert = bool(last["leak_alert"])
        if alert:
            status, css = "LEAK ALERT", "status-leak"
        elif last["is_anomaly"]:
            status, css = "Watching", "status-warn"
        else:
            status, css = "Normal", "status-ok"
        with kpi_slots[0].container():
            card("Status", status, str(last["timestamp"]), css)
        with kpi_slots[1].container():
            card("Extra inflow", f"{last['extra_inflow_lps']:.1f} L/s", "water above normal for this time")
        with kpi_slots[2].container():
            card("Anomaly score", f"{last['anomaly_score']:.3f}", f"threshold {threshold:.3f}")
        with kpi_slots[3].container():
            n_alerts = int(hist["leak_alert"].sum())
            card("Alerts in replay", n_alerts, f"{len(hist)} readings processed")
        if alert:
            banner.error(f"🚨 Possible leak - about {last['extra_inflow_lps']:.0f} L/s of unexplained inflow. "
                         f"Largest pressure drop near **{last['suspected_area']}**. Send a field crew to check.")
        else:
            banner.empty()
        score_slot.plotly_chart(score_chart(hist, threshold, leak_start), width="stretch", key=f"score{upto}")
        rows_now = rows.iloc[: upto + 1]
        pressure_cols = [c for c in SENSORS if c.startswith("P_")][:4]
        flow_slot.plotly_chart(inflow_chart(rows_now, hist), width="stretch", key=f"flow{upto}")
        suspected = last["suspected_area"].replace("junction ", "") if alert else None
        map_slot.plotly_chart(network_map(suspected, true_node, show_truth), width="stretch", key=f"map{upto}")
        pres_fig = sensor_chart(rows_now, pressure_cols, "Pressure (4 of 8 sensors)", "m")
        pres_slot.plotly_chart(pres_fig, width="stretch", key=f"pres{upto}")

    if play:
        hist_rows, threshold = [], 0.5
        try:
            for i in range(len(rows)):
                res = api_predict(rows.iloc[max(0, i - WINDOW + 1): i + 1], api_key)
                threshold = res["threshold"]
                hist_rows.append({**res["results"][-1], "model_version": res["model_version"]})
                hist = pd.DataFrame(hist_rows)
                hist["timestamp"] = pd.to_datetime(hist["timestamp"])
                if i % 2 == 0 or i == len(rows) - 1:
                    render(hist, i, threshold)
                time.sleep(1 / speed)
            st.session_state["hist"] = (scenario, hist, threshold)
        except requests.HTTPError as e:
            st.error(f"API error: {e.response.status_code} {e.response.text}")
    elif "hist" in st.session_state and st.session_state["hist"][0] == scenario:
        _, hist, threshold = st.session_state["hist"]
        render(hist, len(hist) - 1, threshold)
    else:
        st.info("Pick a scenario in the sidebar and press **Replay** to stream it through the API.")

# ---------------------------------------------------------------- models
with tab_models:
    if info and info.get("champion"):
        ch = info["champion"]
        cols = st.columns(4)
        with cols[0]:
            card("Champion", f"v{ch['version']}", f"exported {ch['exported_at'][:16]}", "status-ok")
        with cols[1]:
            card("Test ROC-AUC", ch.get("test_roc_auc") or "-", f"F1 {ch.get('test_f1')}")
        with cols[2]:
            chal = info.get("challenger")
            card("Challenger", f"v{chal['version']}" if chal else "none",
                 "shadow mode" if info["shadow_mode"] else f"canary {info['canary_percent']}%")
        with cols[3]:
            card("Recent anomaly rate", f"{100 * info['recent_anomaly_rate']:.1f}%",
                 f"last {info['recent_predictions']} predictions")
        st.markdown("#### Trained on")
        st.code(f"data file : {ch.get('data_file')}\ndata md5  : {ch.get('data_md5')}\nmlflow run: {ch.get('run_id')}")

    lineage = api_get("/model/lineage", api_key) or []
    st.markdown("#### Lineage log (governance)")
    if lineage:
        lin = pd.DataFrame(lineage)
        show = [c for c in ["time", "event", "version", "previous_version", "user", "reason",
                            "test_roc_auc", "data_file", "data_md5"] if c in lin.columns]
        st.dataframe(lin[show].iloc[::-1], width="stretch", hide_index=True)
    else:
        st.caption("No lineage events yet.")

# ---------------------------------------------------------------- drift
with tab_drift:
    if not drift or "drift_share" not in drift:
        st.info("No drift check has run yet. Run `python -m src.retrain` or the Airflow DAG.")
    else:
        cols = st.columns(3)
        with cols[0]:
            card("Drift detected", "YES" if drift["drift_detected"] else "NO", "",
                 "status-warn" if drift["drift_detected"] else "status-ok")
        with cols[1]:
            card("Drifted sensors", f"{drift['drifted_columns']} / {drift['total_columns']}",
                 f"share {drift['drift_share']:.2f}")
        with cols[2]:
            card("Compared", drift["current"].split("/")[-1], f"vs {drift['reference'].split('/')[-1]}")
        psi = pd.Series(drift["psi"]).sort_values()
        limit = PARAMS["drift"]["psi_threshold"]
        fig = go.Figure(go.Bar(x=psi.values, y=psi.index, orientation="h",
                               marker_color=[RED if v > limit else BLUE for v in psi.values]))
        fig.add_vline(x=limit, line=dict(color=AMBER, dash="dash"), annotation_text="PSI limit")
        fig.update_layout(height=420, template="plotly_dark", title="PSI per sensor (higher = more change)",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, width="stretch")
        report = ROOT / "reports" / "drift_report.html"
        if report.exists() and st.toggle("Show full Evidently report"):
            components.html(report.read_text(), height=900, scrolling=True)

# ---------------------------------------------------------------- operations
with tab_ops:
    st.markdown("#### Watchdog incidents")
    inc_file = ROOT / "logs" / "incidents.jsonl"
    if inc_file.exists():
        inc = pd.DataFrame([json.loads(x) for x in inc_file.read_text().splitlines() if x.strip()])
        st.dataframe(inc.iloc[::-1], width="stretch", hide_index=True)
    else:
        st.caption("No incidents recorded. Start the watchdog with `python -m ops.watchdog`.")

    st.markdown("#### API traffic (from logs/api.log)")
    log_file = ROOT / "logs" / "api.log"
    if log_file.exists():
        logs = pd.DataFrame([json.loads(x) for x in log_file.read_text().splitlines()[-5000:] if x.strip()])
        logs["time"] = pd.to_datetime(logs["time"])
        per_min = logs.set_index("time").resample("1min").agg({"latency_ms": "mean", "status": "count"})
        per_min.columns = ["avg latency (ms)", "requests"]
        st.line_chart(per_min.tail(120))
        anomalies = ROOT / "reports" / "log_anomalies.png"
        if anomalies.exists():
            st.image(str(anomalies), caption="Log anomaly detector output (python -m ops.log_anomaly)")
