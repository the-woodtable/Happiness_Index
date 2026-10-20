import { initializeApp } from "https://www.gstatic.com/firebasejs/10.14.1/firebase-app.js";
import { getDatabase, ref, onValue, get } from "https://www.gstatic.com/firebasejs/10.14.1/firebase-database.js";
import { firebaseConfig } from "./config.js";

const db = getDatabase(initializeApp(firebaseConfig));
const $ = (id) => document.getElementById(id);

const OFFLINE_AFTER_MS = 30000;
const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;
const C = { purple: "#4f3a97", yellow: "#fed200", yellowEdge: "#be9600", coral: "#f2545b", coralEdge: "#aa2832", pale: "#ece4f4", white: "#ffffff" };
const SERIES = [
  { idx: 1, label: "Engaged", color: C.purple, edge: null },
  { idx: 2, label: "Smiled", color: C.yellow, edge: C.yellowEdge },
  { idx: 3, label: "Laughed", color: C.coral, edge: C.coralEdge },
];

let offset = 0;          // server clock minus this device's clock, ms
let connected = false;
let fatal = false;       // a read was refused or failed
let viewing = "";        // "" = live, otherwise a YYYY-MM-DD past day
let dayIndex = {};
let tickCount = 0;
const live = { status: null, summary: null, timeline: null, dayStartMs: null, day: null };
const hist = { summary: null, timeline: null, loaded: false, failed: false };

// ---- helpers ----
const serverNow = () => Date.now() + offset;
const dayStartFor = (d) => Date.parse(`${d}T04:00:00+08:00`);
const fmtSec = (v) => (typeof v === "number" ? `${v.toFixed(1)} sec` : "--");
const fmtDate = (d) =>
  new Date(`${d}T12:00:00+08:00`).toLocaleDateString("en-SG", {
    timeZone: "Asia/Singapore", weekday: "short", day: "numeric", month: "short", year: "numeric",
  });
const fmtTime = (ms) =>
  new Date(ms).toLocaleTimeString("en-SG", { timeZone: "Asia/Singapore", hour: "numeric", minute: "2-digit" });
function ago(ms) {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 5) return "just now";
  if (s < 60) return `${s} s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} min ago`;
  return `${Math.floor(m / 60)} h ago`;
}
function fmtClock(t) {  // t = seconds since 04:00 -> "HH:MM"
  const mins = Math.round((4 * 3600 + t) / 60) % 1440;
  return `${String(Math.floor(mins / 60)).padStart(2, "0")}:${String(mins % 60).padStart(2, "0")}`;
}
function normPoints(tl) {
  let pts = tl && tl.points ? (Array.isArray(tl.points) ? tl.points : Object.values(tl.points)) : [];
  pts = pts
    .map((p) => (Array.isArray(p) ? p : Object.values(p || {})))
    .filter((p) => p.length >= 4)
    .map((p) => p.map(Number))
    .sort((a, b) => a[0] - b[0]);
  if (!pts.length || pts[0][0] > 0) pts.unshift([0, 0, 0, 0]);
  return pts;
}

// ---- big-number blocks ----
document.querySelectorAll("[data-stat]").forEach((el) => {
  for (const cls of ["big", "of", "pct"]) {
    const s = document.createElement("span");
    s.className = cls;
    el.appendChild(s);
  }
});
function setStat(key, n, total) {
  const [big, of, pct] = document.querySelector(`[data-stat="${key}"]`).children;
  if (typeof n !== "number" || typeof total !== "number") {
    big.textContent = "--"; of.textContent = ""; pct.textContent = "";
    return;
  }
  big.textContent = String(n);
  of.textContent = `/ ${total}`;
  pct.textContent = total > 0 ? `(${Math.round((100 * n) / total)}%)` : "(--)";
}

// ---- which data is on screen ----
function currentView() {
  if (viewing) {
    const start = dayStartFor(viewing);
    const pts = normPoints(hist.timeline);
    const lastT = pts[pts.length - 1][0];
    const s = hist.summary;
    const end = s && typeof s.updatedMs === "number"
      ? Math.min(86400, Math.max(lastT, (s.updatedMs - start) / 1000))
      : lastT;
    return { summary: hist.summary, timeline: hist.timeline, end };
  }
  const start = live.dayStartMs ?? (live.day ? dayStartFor(live.day) : null);
  const end = start == null ? 0 : Math.max(0, (serverNow() - start) / 1000);
  return { summary: live.summary, timeline: live.timeline, end };
}

function render() {
  const { summary, timeline, end } = currentView();
  const s = summary || {};
  const n = typeof s.n === "number" ? s.n : null;
  setStat("stayed", s.stayed, n);
  setStat("smiles", s.smiles, n);
  setStat("laughs", s.laughs, n);
  setStat("shocked", s.shocked, n);
  $("staySub").textContent = `Stopped to look for ${typeof s.stayThresholdS === "number" ? s.stayThresholdS : 5} seconds +`;
  $("eye").textContent = fmtSec(s.eyeContactMedianS);
  $("reaction").textContent = fmtSec(s.reactionMedianS);
  $("tts").textContent = fmtSec(s.timeToHappinessMedianS);
  $("duration").textContent = fmtSec(s.happinessDurationMedianS);
  $("happiness").textContent = typeof s.happiness === "number" ? String(Math.round(s.happiness)) : "--";
  let total = null;
  if (typeof s.totalHappiness === "number") total = s.totalHappiness;
  else if (n === 0 || n === null) total = null;
  else if (!s.smiles) total = 0;
  else if (typeof s.happiness === "number") total = (s.smiles / n) * s.happiness;
  $("totalHappiness").textContent = total === null ? "--" : String(Math.round(total));
  drawGraph(timeline, end);
}

// ---- graph (plain SVG step chart) ----
function drawGraph(timeline, endSec) {
  const svg = $("graph");
  svg.textContent = "";
  const NS = "http://www.w3.org/2000/svg";
  const mk = (name, attrs, text) => {
    const e = document.createElementNS(NS, name);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    svg.appendChild(e);
    return e;
  };
  const W = 360, H = 230, L = 30, R = W - 10, T = 12, B = H - 26;
  const pts = normPoints(timeline);
  const last = pts[pts.length - 1];
  const end = Math.max(endSec || 0, last[0]);

  const legend = $("legend");
  legend.textContent = "";
  for (const s of SERIES) {
    const span = document.createElement("span");
    const sw = document.createElement("i");
    sw.style.background = s.color;
    span.appendChild(sw);
    span.appendChild(document.createTextNode(`${s.label} ${last[s.idx]}`));
    legend.appendChild(span);
  }

  const span = Math.max(end, 3600);
  const step = [600, 900, 1800, 3600, 7200, 10800, 14400, 21600].find((c) => span / c <= 5) ?? 21600;
  const xmax = Math.ceil(span / step) * step;
  const top = Math.max(last[1], last[2], last[3], 4);
  const ystep = [1, 2, 5, 10, 20, 50, 100, 200, 500].find((c) => top / c <= 5) ?? 500;
  const ymax = Math.max(ystep * 4, Math.ceil(top / ystep) * ystep);
  const X = (t) => L + ((R - L) * t) / xmax;
  const Y = (c) => B - ((B - T) * c) / ymax;

  mk("rect", { x: L, y: T, width: R - L, height: B - T, fill: C.pale });
  for (let c = 0; c <= ymax + 1e-9; c += ystep) {
    mk("line", { x1: L, x2: R, y1: Y(c), y2: Y(c), stroke: C.white, "stroke-width": 1 });
    mk("text", { x: L - 5, y: Y(c) + 4, "text-anchor": "end", "font-size": 10, fill: C.purple }, String(c));
  }
  for (let t = 0; t <= xmax + 1e-9; t += step) {
    mk("line", { x1: X(t), x2: X(t), y1: T, y2: B, stroke: C.white, "stroke-width": 1 });
    mk("text", { x: X(t), y: B + 14, "text-anchor": "middle", "font-size": 10, fill: C.purple }, fmtClock(t));
  }
  for (const s of SERIES) {
    let d = `M${X(pts[0][0])} ${Y(pts[0][s.idx])}`;
    for (let i = 1; i < pts.length; i++) {
      d += `L${X(pts[i][0])} ${Y(pts[i - 1][s.idx])}L${X(pts[i][0])} ${Y(pts[i][s.idx])}`;
    }
    d += `L${X(end)} ${Y(last[s.idx])}`;
    const common = { d, fill: "none", "stroke-linejoin": "round", "stroke-linecap": "round" };
    if (s.edge) mk("path", { ...common, stroke: s.edge, "stroke-width": 5 });
    mk("path", { ...common, stroke: s.color, "stroke-width": 3 });
    mk("circle", { cx: X(end), cy: Y(last[s.idx]), r: 4, fill: s.color, stroke: C.purple, "stroke-width": 1 });
  }
  mk("line", { x1: L, x2: L, y1: T, y2: B, stroke: C.purple, "stroke-width": 2 });
  mk("line", { x1: L, x2: R, y1: B, y2: B, stroke: C.purple, "stroke-width": 2 });
}

// ---- status pill, last updated, offline banner ----
function setPill(state, text) {
  const p = $("pill");
  p.className = `pill ${state}`;
  p.textContent = text;
}
function tick() {
  $("histBar").hidden = !viewing;
  if (viewing) {
    setPill("hist", "History");
    $("updated").textContent = "";
    $("offlineBanner").hidden = true;
    document.body.classList.remove("offline");
    $("histLabel").textContent = hist.failed
      ? `Could not load ${fmtDate(viewing)}`
      : hist.loaded && !hist.summary
        ? `No data for ${fmtDate(viewing)}`
        : `${fmtDate(viewing)} · final totals`;
    return;
  }
  const st = live.status;
  const hasTs = st && typeof st.ts === "number";
  const age = hasTs ? serverNow() - st.ts : null;
  let offline = false;
  if (fatal) {
    setPill("wait", "Can't load data");
  } else if (!connected) {
    setPill("wait", st ? "Reconnecting…" : "Connecting…");
  } else if (!st) {
    setPill("wait", "Waiting for camera");
  } else if (st.camera === "no_frames" || (hasTs && age > OFFLINE_AFTER_MS)) {
    offline = true;
    setPill("offline", "Camera offline");
  } else if (st.camera === "starting") {
    setPill("wait", "Camera starting");
  } else {
    setPill("live", "Live");
  }
  $("updated").textContent = !hasTs
    ? ""
    : offline
      ? `Last seen ${fmtTime(st.ts)}`
      : `Updated ${ago(age)}`;
  $("offlineBanner").hidden = !offline;
  document.body.classList.toggle("offline", offline);
}

// ---- history picker ----
function fillPicker() {
  const sel = $("dayPicker");
  sel.textContent = "";
  const add = (value, label, disabled) => {
    const o = document.createElement("option");
    o.value = value;
    o.textContent = label;
    if (disabled) o.disabled = true;
    sel.appendChild(o);
  };
  add("", "Live (today)");
  const days = Object.keys(dayIndex).filter((d) => DATE_RE.test(d) && d !== live.day).sort().reverse();
  if (!days.length) add("none", "No previous days yet", true);
  for (const d of days) {
    const x = dayIndex[d] || {};
    add(d, `${fmtDate(d)} · ${typeof x.n === "number" ? x.n : 0} people`);
  }
  sel.value = viewing && days.includes(viewing) ? viewing : "";
}
async function selectDay(d) {
  viewing = DATE_RE.test(d) ? d : "";
  hist.summary = hist.timeline = null;
  hist.loaded = hist.failed = false;
  render();
  tick();
  if (!viewing) return;
  const wanted = viewing;
  try {
    const [s, t] = await Promise.all([
      get(ref(db, `days/${wanted}/summary`)),
      get(ref(db, `days/${wanted}/timeline`)),
    ]);
    if (viewing !== wanted) return;
    hist.summary = s.val();
    hist.timeline = t.val();
    hist.loaded = true;
  } catch (e) {
    if (viewing !== wanted) return;
    hist.failed = true;
  }
  render();
  tick();
}
$("dayPicker").addEventListener("change", (e) => selectDay(e.target.value));
$("backLive").addEventListener("click", () => { $("dayPicker").value = ""; selectDay(""); });

// ---- live listeners ----
const onErr = () => { fatal = true; tick(); };
onValue(ref(db, ".info/serverTimeOffset"), (s) => { offset = s.val() || 0; });
onValue(ref(db, ".info/connected"), (s) => { connected = s.val() === true; tick(); });
onValue(ref(db, "live/status"), (s) => { live.status = s.val(); fatal = false; tick(); }, onErr);
onValue(ref(db, "live/summary"), (s) => { live.summary = s.val(); if (!viewing) render(); }, onErr);
onValue(ref(db, "live/timeline"), (s) => { live.timeline = s.val(); if (!viewing) render(); }, onErr);
onValue(ref(db, "live/dayStartMs"), (s) => { live.dayStartMs = s.val(); if (!viewing) render(); }, onErr);
onValue(ref(db, "live/day"), (s) => { live.day = s.val(); fillPicker(); if (!viewing) render(); }, onErr);
onValue(ref(db, "dayIndex"), (s) => { dayIndex = s.val() || {}; fillPicker(); }, onErr);

setInterval(() => {
  tick();
  if (!viewing && ++tickCount % 5 === 0) render();   // moves the graph's right edge
}, 1000);

fillPicker();
render();
tick();
