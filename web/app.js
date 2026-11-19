// AquaSentinel control room.
// Plain JavaScript, no framework: every section below talks to one part of the API.

const $ = (sel) => document.querySelector(sel);
const COLORS = { water: "#0f6fa8", alarm: "#d9480f", pipe: "#a9bfcc", ink: "#0e2433", muted: "#5b7180", ok: "#2b8a5e" };

const state = {
  key: localStorage.getItem("aqua-key"),
  role: null,
  network: null,
  scenarios: [],
  data: null,          // readings of the selected scenario
  results: [],         // model output for each replayed reading
  threshold: null,
  timer: null,
  running: false,
  charts: {},
};

// ------------------------------------------------------------------ API helper
async function api(path, options = {}) {
  const res = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", "X-API-Key": state.key || "", ...(options.headers || {}) },
  });
  if (res.status === 401) { showSignin("Your API key is not valid. Please sign in again."); throw new Error("401"); }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`);
  return body;
}

function toast(message) {
  const t = $("#toast");
  t.textContent = message;
  t.hidden = false;
  clearTimeout(t._timer);
  t._timer = setTimeout(() => (t.hidden = true), 4500);
}

const fmtTime = (iso) => new Date(iso.replace(" ", "T")).toLocaleString("en-GB",
  { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });

// ------------------------------------------------------------------ sign in (role-based access)
function showSignin(error = "") {
  $("#signin").hidden = false;
  $("#signin-error").textContent = error;
}

async function signIn(key) {
  state.key = key;
  try {
    const me = await api("/api/me");
    state.role = me.role;
    localStorage.setItem("aqua-key", key);
    $("#signin").hidden = true;
    $("#role-chip").textContent = me.role;
    document.querySelectorAll(".admin-only").forEach((el) => (el.hidden = me.role !== "admin"));
    $("#play").disabled = me.role === "viewer";
    $("#play").title = me.role === "viewer" ? "Viewers cannot run predictions. Sign in as operator or admin." : "";
    await loadScenarios();
    route();
  } catch (e) {
    if (e.message !== "401") $("#signin-error").textContent = e.message;
  }
}

document.querySelectorAll(".role").forEach((b) => b.addEventListener("click", () => signIn(b.dataset.key)));
$("#signin-form").addEventListener("submit", (e) => { e.preventDefault(); signIn($("#custom-key").value.trim()); });
$("#signout").addEventListener("click", () => { localStorage.removeItem("aqua-key"); stopReplay(); showSignin(); });

// ------------------------------------------------------------------ navigation
function route() {
  const view = (location.hash || "#monitor").slice(1);
  document.querySelectorAll(".view").forEach((v) => (v.hidden = v.id !== `view-${view}`));
  document.querySelectorAll(".tabs a").forEach((a) => a.classList.toggle("active", a.dataset.view === view));
  if (!state.role) return;
  if (view === "models") loadModels();
  if (view === "drift") loadDrift();
  if (view === "operations") loadOperations();
}
window.addEventListener("hashchange", route);

async function checkHealth() {
  try {
    const h = await fetch("/health").then((r) => r.json());
    $("#api-state").className = "dot up";
    $("#model-chip").textContent = h.model_version ? `model v${h.model_version}` : "no model";
  } catch {
    $("#api-state").className = "dot down";
    $("#model-chip").textContent = "API offline";
  }
}

// ------------------------------------------------------------------ network map (SVG)
const svgNS = "http://www.w3.org/2000/svg";
function el(tag, attrs, parent) {
  const e = document.createElementNS(svgNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

async function drawNetwork() {
  const net = (state.network = await fetch("/api/network").then((r) => r.json()));
  const xs = Object.values(net.nodes).map((c) => c[0]);
  const ys = Object.values(net.nodes).map((c) => c[1]);
  const [minX, maxX, minY, maxY] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const pad = (maxX - minX) * 0.04;
  // map coordinates have y going up, SVG has y going down, so y is flipped
  net.point = (n) => [net.nodes[n][0], maxY + minY - net.nodes[n][1]];
  net.unit = (maxX - minX) / 120;   // size of one "dot" in map units

  const svg = $("#map");
  svg.innerHTML = "";
  svg.setAttribute("viewBox", `${minX - pad} ${minY - pad} ${maxX - minX + 2 * pad} ${maxY - minY + 2 * pad}`);
  const u = net.unit;

  const pipes = el("g", {}, svg);
  net.pipeEls = net.pipes.map(([a, b]) => {
    const [x1, y1] = net.point(a), [x2, y2] = net.point(b);
    return el("line", { x1, y1, x2, y2, class: "pipe", "vector-effect": "non-scaling-stroke", "data-a": a, "data-b": b }, pipes);
  });
  const nodes = el("g", {}, svg);
  for (const n of Object.keys(net.nodes)) {
    if (net.sources.includes(n) || net.tanks.includes(n)) continue;
    const [x, y] = net.point(n);
    el("circle", { cx: x, cy: y, r: u * 0.35, class: "junction" }, nodes);
  }
  for (const n of [...net.sources, ...net.tanks]) {
    const [x, y] = net.point(n);
    el("rect", { x: x - u * 0.8, y: y - u * 0.8, width: u * 1.6, height: u * 1.6, class: "source" }, nodes);
    const t = el("text", { x: x + u * 1.3, y: y + u * 0.5, class: "label", "font-size": u * 2.2 }, nodes);
    t.textContent = net.sources.includes(n) ? n : `Tank ${n}`;
  }
  for (const n of net.pressure_sensors) {
    const [x, y] = net.point(n);
    el("rect", { x: x - u * 0.8, y: y - u * 0.8, width: u * 1.6, height: u * 1.6, class: "sensor",
                 transform: `rotate(45 ${x} ${y})`, "vector-effect": "non-scaling-stroke" }, nodes);
    const t = el("text", { x: x + u * 1.3, y: y - u * 0.9, class: "label", "font-size": u * 2 }, nodes);
    t.textContent = n;
  }
  net.overlay = el("g", {}, svg);
}

function markMap(suspectNode, truthNode) {
  const net = state.network;
  if (!net) return;
  net.overlay.innerHTML = "";
  const u = net.unit;
  const radius = u * 9;
  net.pipeEls.forEach((p) => p.classList.remove("hot"));
  if (suspectNode && net.nodes[suspectNode]) {
    const [sx, sy] = net.point(suspectNode);
    // highlight pipes close to the suspected sensor
    net.pipeEls.forEach((p) => {
      const near = [p.dataset.a, p.dataset.b].every((n) => Math.hypot(net.point(n)[0] - sx, net.point(n)[1] - sy) < radius);
      p.classList.toggle("hot", near);
    });
    el("circle", { cx: sx, cy: sy, r: radius, class: "suspect", "vector-effect": "non-scaling-stroke" }, net.overlay);
  }
  if (truthNode && net.nodes[truthNode]) {
    const [x, y] = net.point(truthNode);
    const d = u * 1.4;
    el("path", { d: `M${x - d} ${y - d}L${x + d} ${y + d}M${x + d} ${y - d}L${x - d} ${y + d}`, class: "truth",
                 "vector-effect": "non-scaling-stroke" }, net.overlay);
  }
}

// ------------------------------------------------------------------ scenarios and replay
async function loadScenarios() {
  const season = $("#season").value;
  state.scenarios = await api(`/api/scenarios?season=${season}`);
  const select = $("#scenario");
  select.innerHTML = "";
  // biggest leaks first so the demo starts with an easy one
  const sorted = [...state.scenarios].sort((a, b) => b.area - a.area);
  for (const s of sorted) {
    const opt = document.createElement("option");
    opt.value = s.scenario_id;
    opt.textContent = s.leak ? `${s.scenario_id}  (leak)` : `${s.scenario_id}  (normal)`;
    select.appendChild(opt);
  }
  resetMonitor();
}

function currentScenario() {
  return state.scenarios.find((s) => s.scenario_id === $("#scenario").value);
}

function resetMonitor() {
  stopReplay();
  state.results = [];
  setStatus("normal", "Waiting", "Press Replay to stream this scenario's sensor readings through the model, one 15-minute reading at a time.");
  ["#ro-inflow", "#ro-score", "#ro-count", "#ro-alerts"].forEach((s) => ($(s).textContent = "–"));
  $("#clock").textContent = "–";
  updateTruth();
  drawScoreChart();
}

function updateTruth() {
  const s = currentScenario();
  const show = $("#show-truth").checked && s && s.leak;
  $("#truth-legend").hidden = !show;
  const last = state.results[state.results.length - 1];
  markMap(last && last.leak_alert ? last.suspected_node : null, show ? s.node : null);
}

function setStatus(kind, word, text) {
  $("#status").className = `status ${kind}`;
  $("#status-word").textContent = word;
  $("#status-text").textContent = text;
}

async function startReplay() {
  stopReplay();
  const s = currentScenario();
  state.data = await api(`/api/scenarios/${s.scenario_id}?season=${$("#season").value}`);
  state.results = [];
  $("#play").disabled = true;
  $("#stop").disabled = false;
  state.running = true;
  step(0);
}

function stopReplay() {
  clearTimeout(state.timer);
  state.running = false;
  $("#play").disabled = state.role === "viewer";
  $("#stop").disabled = true;
}

async function step(i) {
  const d = state.data;
  const from = Math.max(0, i - 15);   // send the last 16 readings (4 hours) so the model can smooth
  const readings = [];
  for (let j = from; j <= i; j++) {
    const sensors = {};
    for (const [name, values] of Object.entries(d.sensors)) sensors[name] = values[j];
    readings.push({ timestamp: d.timestamps[j], sensors });
  }
  try {
    const res = await api("/predict", { method: "POST", body: JSON.stringify({ readings }) });
    state.threshold = res.threshold;
    state.results.push({ ...res.results[res.results.length - 1], version: res.model_version });
    showReading(i);
  } catch (e) {
    stopReplay();
    toast(e.message);
    return;
  }
  if (!state.running) return;   // Stop was pressed
  if (i < d.timestamps.length - 1) {
    state.timer = setTimeout(() => step(i + 1), Number($("#speed").value));
  } else {
    finishReplay();
  }
}

function showReading(i) {
  const r = state.results[state.results.length - 1];
  const alerts = state.results.filter((x) => x.leak_alert).length;
  $("#clock").textContent = fmtTime(state.data.timestamps[i]);
  $("#ro-inflow").textContent = `${r.extra_inflow > 0 ? "+" : ""}${r.extra_inflow.toFixed(1)} L/s`;
  $("#ro-score").textContent = r.anomaly_score.toFixed(3);
  $("#ro-count").textContent = `${i + 1} / ${state.data.timestamps.length}`;
  $("#ro-alerts").textContent = alerts;

  if (r.leak_alert) {
    setStatus("leak", "Leak alert",
      `About ${Math.round(r.extra_inflow)} L/s more water is entering the network than normal for this time. ` +
      `The biggest pressure drop is near junction ${r.suspected_node}. Send a crew to check that area.`);
  } else if (r.is_anomaly) {
    setStatus("watch", "Unusual", "Inflow is higher than normal. An alert is raised if this lasts 45 minutes.");
  } else {
    setStatus("ok", "Normal", "Inflow matches a normal day at this time.");
  }
  updateTruth();
  updateScoreChart();
}

function finishReplay() {
  stopReplay();
  const s = currentScenario();
  const leakStart = state.data.leak.indexOf(1);
  const firstAlert = state.results.findIndex((r) => r.leak_alert);
  if (s.leak && firstAlert >= leakStart && leakStart >= 0) {
    toast(`Leak detected ${(firstAlert - leakStart) * 15} minutes after it started.`);
  } else if (s.leak) {
    toast("This leak was not detected. Small leaks can hide inside normal demand changes.");
  } else {
    toast(firstAlert >= 0 ? "False alarm on a scenario without a leak." : "No alerts. Correct: this scenario has no leak.");
  }
}

$("#play").addEventListener("click", startReplay);
$("#stop").addEventListener("click", stopReplay);
$("#season").addEventListener("change", loadScenarios);
$("#scenario").addEventListener("change", resetMonitor);
$("#show-truth").addEventListener("change", updateTruth);

// ------------------------------------------------------------------ charts
// draws a dotted vertical line at a given x index (used for "leak started")
const markerPlugin = {
  id: "marker",
  afterDatasetsDraw(chart, _args, opts) {
    if (opts.index == null || opts.index < 0) return;
    const x = chart.scales.x.getPixelForValue(opts.index);
    const { top, bottom } = chart.chartArea;
    const ctx = chart.ctx;
    ctx.save();
    ctx.strokeStyle = COLORS.ink;
    ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = COLORS.ink;
    ctx.font = "600 12px 'Barlow Semi Condensed', sans-serif";
    ctx.fillText(opts.label || "", x + 6, top + 12);
    ctx.restore();
  },
};

function baseOptions() {
  return {
    responsive: true, maintainAspectRatio: false, animation: false,
    interaction: { mode: "index", intersect: false },
    plugins: { legend: { position: "bottom", labels: { boxWidth: 12 } } },
    scales: { x: { grid: { display: false }, ticks: { maxTicksLimit: 10 } } },
  };
}

function drawScoreChart() {
  if (!window.Chart) return;
  state.charts.score?.destroy();
  const options = baseOptions();
  options.scales.y = { title: { display: true, text: "anomaly score" }, grace: "5%" };
  options.scales.y2 = { position: "right", title: { display: true, text: "extra inflow (L/s)" }, grid: { display: false } };
  options.plugins.marker = { index: -1, label: "leak starts" };
  state.charts.score = new Chart($("#score-chart"), {
    data: {
      labels: [],
      datasets: [
        { type: "line", label: "anomaly score", data: [], borderColor: COLORS.water, borderWidth: 2,
          pointRadius: [], pointBackgroundColor: COLORS.alarm, yAxisID: "y", tension: 0.2 },
        { type: "line", label: "alert threshold", data: [], borderColor: COLORS.alarm, borderDash: [6, 5],
          borderWidth: 1.5, pointRadius: 0, yAxisID: "y" },
        { type: "bar", label: "extra inflow", data: [], backgroundColor: "rgba(169,191,204,0.55)", yAxisID: "y2" },
      ],
    },
    options,
    plugins: [markerPlugin],
  });
}

function updateScoreChart() {
  const chart = state.charts.score;
  if (!chart) return;
  const n = state.results.length;
  chart.data.labels = state.data.timestamps.slice(0, n).map((t) => fmtTime(t));
  chart.data.datasets[0].data = state.results.map((r) => r.anomaly_score);
  chart.data.datasets[0].pointRadius = state.results.map((r) => (r.leak_alert ? 3 : 0));
  chart.data.datasets[1].data = state.results.map(() => state.threshold);
  chart.data.datasets[2].data = state.results.map((r) => r.extra_inflow);
  const leakStart = state.data.leak.indexOf(1);
  chart.options.plugins.marker.index = leakStart >= 0 && leakStart < n ? leakStart : -1;
  chart.update("none");
}

// ------------------------------------------------------------------ models page
async function loadModels() {
  const [info, lineage] = await Promise.all([api("/model/info"), api("/model/lineage")]);
  const c = info.champion;
  $("#champion").innerHTML = c ? `
    <p class="muted">In production</p>
    <p class="version">v${c.version}</p>
    <p class="since">since ${fmtTime(c.exported_at)}, trained by ${c.trained_by || "pipeline"}</p>
    <dl class="metrics">
      <div><dt>ROC-AUC on test data</dt><dd>${Number(c.roc_auc).toFixed(3)}</dd></div>
      <div><dt>Leaks detected</dt><dd>${Math.round(c.leaks_detected * 100)}%</dd></div>
      <div><dt>Typical delay</dt><dd>${Math.round(c.detection_delay_min)} min</dd></div>
      <div><dt>False alarms</dt><dd>${(c.false_alarm_rate * 100).toFixed(1)}%</dd></div>
      <div><dt>Flagged in last 500 readings</dt><dd>${(info.recent_anomaly_rate * 100).toFixed(1)}%</dd></div>
    </dl>
    <p class="trained-on">Trained on <strong>${c.data_file}</strong> (MD5 ${c.data_md5}).
      Waiting model: ${info.challenger ? "v" + info.challenger.version : "none"}.
      Shadow mode ${info.shadow_mode ? "on" : "off"}, canary ${info.canary_percent}% of requests.</p>`
    : `<p>No model in production yet. Run <code>python -m src.train</code>.</p>`;

  const names = { registered: "Registered", promoted: "Put into production", rolled_back: "Rolled back to", rejected: "Kept waiting" };
  $("#lineage").innerHTML = lineage.slice().reverse().map((e) => `
    <li class="${e.event}">
      <p class="what">${names[e.event] || e.event} v${e.version}</p>
      <p class="meta">${fmtTime(e.time)} by ${e.user}${e.reason ? ". " + e.reason : ""}</p>
      <p class="meta">Data: ${e.data_file}, ROC-AUC ${e.roc_auc ?? "–"}</p>
    </li>`).join("") || `<li><p class="meta">Nothing yet.</p></li>`;
}

async function adminAction(button, request, describe) {
  const out = $("#action-result");
  button.disabled = true;
  out.textContent = "Working…";
  try {
    out.textContent = describe(await request());
    await Promise.all([loadModels(), checkHealth()]);
  } catch (e) {
    out.textContent = e.message;
  } finally {
    button.disabled = false;
  }
}

$("#run-cycle").addEventListener("click", (e) => adminAction(e.target,
  () => api("/monitoring/run", { method: "POST" }),
  (r) => (r.drift_share !== undefined
    ? `${Math.round(r.drift_share * 100)}% of sensors drifted. ${r.action.charAt(0).toUpperCase() + r.action.slice(1)}.`
    : "No new data to check.")));
$("#promote").addEventListener("click", (e) => adminAction(e.target,
  () => api(`/model/promote/${$("#promote-version").value}`, { method: "POST" }),
  (r) => `Version ${r.champion} is now in production.`));
$("#rollback").addEventListener("click", (e) => adminAction(e.target,
  () => api("/model/rollback", { method: "POST" }),
  (r) => `Rolled back. Version ${r.rolled_back_to} is now in production.`));

// ------------------------------------------------------------------ drift page
async function loadDrift() {
  const d = await api("/monitoring/drift");
  if (!d.psi) return;
  const limit = 0.5;
  $("#psi-limit").textContent = limit;
  $("#drift-share").textContent = `${Math.round(d.drift_share * 100)}%`;
  $("#drift-summary").textContent = `Last check ${fmtTime(d.time)}: ${d.current} compared with ${d.reference}.`;
  $("#drift-action").textContent = d.action ? d.action.charAt(0).toUpperCase() + d.action.slice(1) + "." : "";

  const entries = Object.entries(d.psi).sort((a, b) => b[1] - a[1]);
  state.charts.psi?.destroy();
  const options = baseOptions();
  options.indexAxis = "y";
  options.plugins.legend = { display: false };
  options.scales = { x: { title: { display: true, text: "PSI (higher = more change)" } }, y: { grid: { display: false } } };
  state.charts.psi = new Chart($("#psi-chart"), {
    type: "bar",
    data: {
      labels: entries.map(([k]) => (k.startsWith("P_") ? `pressure ${k.slice(2)}` : `flow ${k.slice(2)}`)),
      datasets: [{ data: entries.map(([, v]) => v),
                   backgroundColor: entries.map(([, v]) => (v > limit ? COLORS.alarm : COLORS.water)) }],
    },
    options,
  });
}

// ------------------------------------------------------------------ operations page
async function loadOperations() {
  const ops = await api("/api/operations");
  const rows = (items, cols) => items.map((i) => `<tr>${cols.map((c) => `<td>${c(i)}</td>`).join("")}</tr>`).join("");
  $("#incidents tbody").innerHTML = rows(ops.incidents, [(i) => fmtTime(i.time), (i) => i.incident.replace(/_/g, " "), (i) => i.detail])
    || `<tr><td colspan="3" class="empty">No incidents. Start the watchdog with: python -m ops.watchdog</td></tr>`;
  $("#audit tbody").innerHTML = rows(ops.audit, [(i) => fmtTime(i.time), (i) => i.role, (i) => i.action, (i) => i.detail || ""])
    || `<tr><td colspan="4" class="empty">No admin actions yet.</td></tr>`;

  state.charts.traffic?.destroy();
  const options = baseOptions();
  options.scales.y = { title: { display: true, text: "requests / min" }, beginAtZero: true };
  options.scales.y2 = { position: "right", title: { display: true, text: "avg latency (ms)" }, grid: { display: false }, beginAtZero: true };
  state.charts.traffic = new Chart($("#traffic-chart"), {
    data: {
      labels: ops.traffic.map((t) => t.minute.slice(11)),
      datasets: [
        { type: "bar", label: "requests", data: ops.traffic.map((t) => t.requests), backgroundColor: "rgba(15,111,168,0.6)", yAxisID: "y" },
        { type: "line", label: "avg latency", data: ops.traffic.map((t) => t.latency), borderColor: COLORS.alarm, pointRadius: 0, yAxisID: "y2" },
      ],
    },
    options,
  });
}

// ------------------------------------------------------------------ start
window.addEventListener("DOMContentLoaded", async () => {
  if (window.Chart) {
    Chart.defaults.font.family = "Barlow, system-ui, -apple-system, 'Segoe UI', sans-serif";
    Chart.defaults.color = COLORS.muted;
  }
  await checkHealth();
  setInterval(checkHealth, 10000);
  await drawNetwork().catch(() => toast("Could not load the network map."));
  drawScoreChart();
  if (state.key) signIn(state.key); else showSignin();
  route();
});
