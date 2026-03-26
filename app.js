/* NeuroScan — app.js  (no CSV upload required) */

const API = "http://localhost:5000";

/* ── Samples ── */
const SAMPLES = {
  healthy: [
    119.992, 157.302, 74.997, 0.00784, 0.00007, 0.0037,
    0.00554, 0.01109, 0.04374, 0.426, 0.02182, 0.0313,
    0.02971, 0.06545, 0.02211, 21.033, 0.414783, 0.815285,
    -4.813031, 0.266482, 2.301442, 0.284654
  ],
  parkinsons: [
    162.568, 198.346, 77.63, 0.00502, 0.000003, 0.0028,
    0.00340, 0.00839, 0.01791, 0.168, 0.00910, 0.01273,
    0.01043, 0.02730, 0.01137, 23.064, 0.457492, 0.719819,
    -7.008949, 0.073069, 2.106750, 0.106396
  ]
};

/* ── Status poll ── */
async function checkStatus() {
  try {
    const r = await fetch(`${API}/status`);
    const d = await r.json();
    const dot = document.getElementById("statusDot");
    const txt = document.getElementById("statusTxt");

    if (d.scaler_ready) {
      dot.className = "status-dot ok";
      txt.textContent = `MODEL READY · ${d.model_type} ${d.kernel?.toUpperCase() ?? ""}`;
      document.getElementById("pillAcc").textContent =
        d.test_acc ? `${d.test_acc}%` : "—";
      document.getElementById("pillRows").textContent =
        d.dataset?.total ?? "—";
      document.getElementById("btnHint").textContent =
        "ENTER VALUES OR LOAD A SAMPLE ABOVE";
    } else {
      dot.className = "status-dot warn";
      txt.textContent = "BACKEND STARTING…";
      setTimeout(checkStatus, 3000);
    }
  } catch {
    document.getElementById("statusDot").className = "status-dot err";
    document.getElementById("statusTxt").textContent = "BACKEND OFFLINE";
    setTimeout(checkStatus, 5000);
  }
}

/* ── Helpers ── */
function loadSample(key) {
  SAMPLES[key].forEach((v, i) => {
    const el = document.getElementById(`f${i}`);
    if (el) el.value = v;
  });
}

function clearForm() {
  for (let i = 0; i < 22; i++) {
    const el = document.getElementById(`f${i}`);
    if (el) el.value = "";
  }
  document.getElementById("resultSection").style.display = "none";
}

/* ── Scan animation ── */
function showScan(on) {
  const ov = document.getElementById("scanOverlay");
  ov.style.display = on ? "flex" : "none";
  if (on) animateBar();
}

function animateBar() {
  const bar = document.getElementById("scanBar");
  const msgs = [
    "ANALYZING VOICE BIOMARKERS",
    "APPLYING STANDARD SCALER",
    "COMPUTING SVM DECISION",
    "GENERATING RESULT"
  ];
  let p = 0, m = 0;
  const iv = setInterval(() => {
    p = Math.min(p + Math.random() * 12, 95);
    bar.style.width = `${p}%`;
    if (p > (m + 1) * 23 && m < msgs.length - 1) {
      m++;
      document.getElementById("scanLbl").textContent = msgs[m];
    }
    if (p >= 95) clearInterval(iv);
  }, 120);
}

/* ── Gauge ── */
function animateGauge(pct, pos) {
  const arc    = document.getElementById("gArc");
  const needle = document.getElementById("gNeedle");
  const valEl  = document.getElementById("gaugeVal");
  const total  = 163;

  arc.setAttribute("stroke", pos ? "url(#gPos)" : "url(#gNeg)");
  const offset = total - (total * pct / 100);

  let cur = total, frame = 0;
  const target = offset;
  const step   = (total - target) / 40;
  const iv = setInterval(() => {
    cur = Math.max(cur - step, target);
    arc.setAttribute("stroke-dashoffset", cur.toFixed(1));
    const ang = -90 + (180 * pct / 100);
    needle.style.transform = `rotate(${-90 + (180 * (total - cur) / total)}deg)`;
    frame++;
    if (cur <= target) {
      clearInterval(iv);
      needle.style.transform = `rotate(${-90 + (180 * pct / 100)}deg)`;
    }
  }, 18);

  let displayed = 0;
  const iv2 = setInterval(() => {
    displayed = Math.min(displayed + pct / 40, pct);
    valEl.textContent = `${Math.round(displayed)}%`;
    if (displayed >= pct) clearInterval(iv2);
  }, 18);
}

/* ── Main analysis ── */
async function runAnalysis() {
  const features = [];
  let hasEmpty = false;

  for (let i = 0; i < 22; i++) {
    const el  = document.getElementById(`f${i}`);
    const val = parseFloat(el.value);
    el.classList.remove("bad");

    if (el.value.trim() === "" || isNaN(val)) {
      el.classList.add("bad");
      hasEmpty = true;
    } else {
      features.push(val);
    }
  }

  if (hasEmpty) {
    document.getElementById("btnHint").textContent =
      "⚠ FILL ALL 22 FIELDS — highlighted in red";
    return;
  }

  showScan(true);
  document.getElementById("analyzeBtn").disabled = true;

  try {
    const r = await fetch(`${API}/predict`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ features })
    });

    const data = await r.json();
    showScan(false);

    if (data.error) {
      alert("Server error: " + data.error);
      document.getElementById("analyzeBtn").disabled = false;
      return;
    }

    renderResult(data);
  } catch (e) {
    showScan(false);
    alert("Could not reach backend at " + API + ". Is app.py running?");
    document.getElementById("analyzeBtn").disabled = false;
  }
}

/* ── Render ── */
function renderResult(data) {
  const positive = data.prediction === 1;
  const section  = document.getElementById("resultSection");
  const card     = document.getElementById("resultCard");
  const title    = document.getElementById("resTitle");
  const desc     = document.getElementById("resDesc");
  const metrics  = document.getElementById("metricsGrid");
  const gVal     = document.getElementById("gaugeVal");

  card.className = `result-card ${positive ? "pos" : "neg"}`;
  title.textContent = positive ? "Parkinson's Detected" : "No Parkinson's Detected";
  desc.textContent  = positive
    ? "Voice biomarkers show patterns consistent with Parkinson's disease. The SVM model classified this profile as high-risk based on acoustic irregularities. Please consult a neurologist."
    : "Voice biomarkers appear within normal range. The SVM model classified this profile as low-risk. Continue regular health check-ups as recommended.";

  gVal.style.color = positive ? "var(--red)" : "var(--green)";

  metrics.innerHTML = `
    <div class="metric"><div class="metric-lbl">Prediction</div><div class="metric-val" style="color:${positive ? "var(--red)" : "var(--green)"}">${positive ? "POSITIVE" : "NEGATIVE"}</div></div>
    <div class="metric"><div class="metric-lbl">Confidence</div><div class="metric-val">${data.confidence}%</div></div>
    <div class="metric"><div class="metric-lbl">Decision Score</div><div class="metric-val">${data.decision_score}</div></div>
    <div class="metric"><div class="metric-lbl">Model</div><div class="metric-val">RBF-SVM</div></div>
    <div class="metric"><div class="metric-lbl">Features</div><div class="metric-val">22</div></div>
    <div class="metric"><div class="metric-lbl">Status</div><div class="metric-val" style="color:var(--green)">COMPLETE</div></div>
  `;

  section.style.display = "block";
  section.scrollIntoView({ behavior: "smooth", block: "start" });
  animateGauge(data.confidence, positive);
  document.getElementById("analyzeBtn").disabled = false;
  document.getElementById("btnHint").textContent = "ANALYSIS COMPLETE — SCROLL DOWN FOR RESULTS";
}

/* ── Init ── */
checkStatus();