/* ==========================================================================
   Dhara-Safe — frontend application logic (restructured)

   No build step: plain JS, fetches the FastAPI backend (app.py) which serves
   this file too via StaticFiles. If opened standalone, set window.DHARA_API_BASE.
   ========================================================================== */

const API_BASE = window.DHARA_API_BASE || "";
const REFRESH_MS = 30000;

let selectedLocationId = null;
let selectedLocationLatLon = null; // {lat, lon} for unmonitored/explored points
let mapMain, mapEmergency;
let mapMarkers = {};
let chartImportance, chartTrend;

// ---------------------------------------------------------------- utils

async function api(path, opts) {
  const res = await fetch(API_BASE + path, opts);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

function riskColor(level) {
  return { LOW: "#2fbf71", MEDIUM: "#f0b93d", HIGH: "#f2793a", CRITICAL: "#ea4b4b", UNKNOWN: "#6b7996" }[level] || "#6b7996";
}

function fmtTime(iso) {
  if (!iso) return "–";
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}

// ---------------------------------------------------------------- glossary tooltips (plain-language)

const GLOSSARY = {
  random_forest: "Random Forest: a machine-learning method that builds hundreds of simple decision trees and averages their votes — more reliable than any single tree.",
  formula: "This shows the exact math behind the risk %, with real numbers plugged in, instead of a hidden 'black box' number.",
  validation: "We tested the model against real, documented flood/landslide events from the past year to see how often it would have correctly flagged danger.",
  accuracy: "How often the model's predictions matched what actually happened, when tested against real historical events.",
  scs_cn: "SCS-CN (Soil Conservation Service Curve Number): a standard hydrology formula that estimates how much rainfall runs off the surface vs soaks into the ground.",
  confidence: "How much to trust an estimate — higher when there's more nearby real data to base it on.",
};

document.addEventListener("click", (e) => {
  const term = e.target.closest(".glossary-term");
  document.querySelectorAll(".glossary-popup").forEach((p) => p.remove());
  if (term) {
    const key = term.dataset.term;
    const popup = el("div", "glossary-popup", GLOSSARY[key] || "No definition available.");
    document.body.appendChild(popup);
    const rect = term.getBoundingClientRect();
    popup.style.top = (window.scrollY + rect.bottom + 6) + "px";
    popup.style.left = Math.min(window.scrollX + rect.left, window.innerWidth - 280) + "px";
    e.stopPropagation();
  }
});

// ---------------------------------------------------------------- GPS (works offline once fixed — see README)

function setGpsStatus(text, color) {
  const pill = document.getElementById("gps-status");
  pill.innerHTML = `<span style="width:6px;height:6px;border-radius:50%;background:${color};display:inline-block;"></span> GPS: ${text}`;
}

function getLocation() {
  return new Promise((resolve) => {
    if (!navigator.geolocation) { setGpsStatus("unsupported", "#ea4b4b"); resolve(null); return; }
    setGpsStatus("locating…", "#f0b93d");
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const coords = { lat: pos.coords.latitude, lon: pos.coords.longitude, ts: Date.now() };
        localStorage.setItem("dhara_last_gps", JSON.stringify(coords));
        setGpsStatus("live", "#2fbf71");
        resolve(coords);
      },
      () => {
        // GPS chip itself works without internet, just slower; if it still fails, use cache
        const cached = localStorage.getItem("dhara_last_gps");
        if (cached) {
          const c = JSON.parse(cached);
          const ageMin = Math.round((Date.now() - c.ts) / 60000);
          setGpsStatus(`cached, ${ageMin}min ago`, "#f0b93d");
          resolve(c);
        } else {
          setGpsStatus("unavailable", "#ea4b4b");
          resolve(null);
        }
      },
      { enableHighAccuracy: true, timeout: 20000, maximumAge: 300000 }
    );
  });
}

// ---------------------------------------------------------------- navigation

document.querySelectorAll(".nav-item").forEach((item) => {
  item.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((i) => i.classList.remove("active"));
    document.querySelectorAll(".page").forEach((p) => p.classList.remove("active"));
    item.classList.add("active");
    document.getElementById("page-" + item.dataset.page).classList.add("active");
    onPageShown(item.dataset.page);
  });
});

function onPageShown(page) {
  if (page === "home" && mapMain) setTimeout(() => mapMain.invalidateSize(), 50);
  if (page === "emergency") loadEmergency();
  if (page === "prediction") renderPrediction();
  if (page === "risk" && selectedLocationId) loadRiskDetail(selectedLocationId);
  if (page === "weather" && selectedLocationId) loadWeatherFor(selectedLocationId);
}

// ---------------------------------------------------------------- i18n (lightweight)

const I18N = {
  en: {
    "nav.home": "Dashboard", "nav.prediction": "Prediction", "nav.risk": "Risk Analysis",
    "nav.weather": "Live Weather", "nav.monitor": "Model Monitor", "nav.emergency": "Emergency & SOS",
    "nav.account": "Register",
    "home.title": "National Risk Overview",
    "home.subtitle": "Hyper-local flash-flood & landslide risk across monitored villages/wards. Refreshes every 15 minutes.",
  },
  hi: {
    "nav.home": "डैशबोर्ड", "nav.prediction": "भविष्यवाणी", "nav.risk": "जोखिम विश्लेषण",
    "nav.weather": "लाइव मौसम", "nav.monitor": "मॉडल मॉनिटर", "nav.emergency": "आपातकाल और एसओएस",
    "nav.account": "पंजीकरण",
    "home.title": "राष्ट्रीय जोखिम अवलोकन",
    "home.subtitle": "निगरानी किए गए गाँवों/वार्डों में हाइपर-लोकल बाढ़ और भूस्खलन जोखिम। हर 15 मिनट में अपडेट होता है।",
  },
};

document.querySelectorAll(".lang-toggle button").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".lang-toggle button").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    applyLang(btn.dataset.lang);
    localStorage.setItem("dhara_lang", btn.dataset.lang);
  });
});

function applyLang(lang) {
  const dict = I18N[lang] || I18N.en;
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    const key = node.getAttribute("data-i18n");
    if (dict[key]) node.textContent = dict[key];
  });
}

(function initLang() {
  const saved = localStorage.getItem("dhara_lang") || "en";
  const btn = document.querySelector(`.lang-toggle button[data-lang="${saved}"]`);
  if (btn) btn.click();
})();

// ---------------------------------------------------------------- search (top bar — feeds every page)

const searchInput = document.getElementById("search-input");
const searchResults = document.getElementById("search-results");
let searchDebounce;

searchInput.addEventListener("input", () => {
  clearTimeout(searchDebounce);
  const q = searchInput.value.trim();
  if (q.length < 2) { searchResults.classList.remove("open"); return; }
  searchDebounce = setTimeout(() => runSearch(q, searchResults, selectLocation), 300);
});

document.addEventListener("click", (e) => {
  if (!e.target.closest(".search-box")) searchResults.classList.remove("open");
});

async function runSearch(q, resultsBox, onPick) {
  try {
    const data = await api(`/api/search?q=${encodeURIComponent(q)}`);
    resultsBox.innerHTML = "";
    if (!data.results.length) {
      resultsBox.appendChild(el("div", "search-result-row", `<div class="loc-line2">${data.note}</div>`));
    } else {
      data.results.forEach((r) => {
        const row = el("div", "search-result-row");
        if (data.match_type === "monitored_grid") {
          row.innerHTML = `<div class="loc-line1">${r.village}, ${r.district}</div><div class="loc-line2">${r.state} · ${r.risk_level} ${r.risk_score ?? ""}%</div>`;
          row.addEventListener("click", () => onPick(r.id));
        } else if (data.match_type === "geocoded_live") {
          const live = r.live_data;
          const sub = live && live.mode === "interpolated_estimate"
            ? `Estimated ${live.estimated_risk_level} (not a full prediction)`
            : "Live data available";
          row.innerHTML = `<div class="loc-line1">${r.display_name}</div><div class="loc-line2">${sub}</div>`;
          row.addEventListener("click", () => onPick(null, { lat: r.lat, lon: r.lon, name: r.display_name }));
        } else {
          row.innerHTML = `<div class="loc-line2">${r.display_name || ""}</div>`;
        }
        resultsBox.appendChild(row);
      });
    }
    resultsBox.classList.add("open");
  } catch (e) { console.error(e); }
}

function selectLocation(id, explorePoint) {
  searchResults.classList.remove("open");
  searchInput.value = "";
  if (id) {
    selectedLocationId = id;
    selectedLocationLatLon = null;
  } else if (explorePoint) {
    selectedLocationId = null;
    selectedLocationLatLon = explorePoint;
  }
  // land on Prediction by default — the flagship "chance % + time left" view
  document.querySelector('.nav-item[data-page="prediction"]').click();
}

// ---------------------------------------------------------------- HOME (dashboard + map together)

async function loadDashboard() {
  try {
    const d = await api("/api/dashboard");
    localStorage.setItem("dhara_last_dashboard", JSON.stringify({ data: d, ts: Date.now() }));
    renderDashboard(d, false);
    document.getElementById("offline-banner").classList.remove("show");
  } catch (e) {
    const cached = localStorage.getItem("dhara_last_dashboard");
    if (cached) {
      const { data, ts } = JSON.parse(cached);
      renderDashboard(data, true);
      document.getElementById("offline-timestamp").textContent = new Date(ts).toLocaleString();
      document.getElementById("offline-banner").classList.add("show");
    }
  }
}

function renderDashboard(d, isOffline) {
  document.getElementById("count-low").textContent = d.risk_counts.LOW || 0;
  document.getElementById("count-medium").textContent = d.risk_counts.MEDIUM || 0;
  document.getElementById("count-high").textContent = d.risk_counts.HIGH || 0;
  document.getElementById("count-critical").textContent = d.risk_counts.CRITICAL || 0;

  document.getElementById("home-updated-callout").textContent = isOffline
    ? `Offline — showing cached data from earlier.`
    : `Pipeline last ran: ${new Date(d.generated_at).toLocaleString()} · ${d.monitored_locations} locations monitored · ${d.active_sos_alerts} active SOS alert(s)`;

  const list = document.getElementById("home-risk-list");
  list.innerHTML = "";
  d.highest_risk_areas.slice(0, 8).forEach((r) => {
    const row = el("div", "risk-row");
    row.innerHTML = `<span class="dot ${r.risk_level}"></span>
      <div class="place"><div class="name">${r.village}, ${r.district}</div><div class="sub">${r.state}</div></div>
      <div class="score">${r.risk_score}%</div><span class="badge ${r.risk_level}">${r.risk_level}</span>`;
    row.addEventListener("click", () => selectLocation(r.id));
    list.appendChild(row);
  });

  renderHomeMap(d.all_locations);
}

function popupHtml(loc) {
  return `<b>${loc.village}, ${loc.district}</b><br>${loc.state}<br>
    Risk: <b style="color:${riskColor(loc.risk_level)}">${loc.risk_level}</b> (${loc.risk_score ?? "–"}%)`;
}

function renderHomeMap(locations) {
  if (!mapMain) {
    mapMain = L.map("map").setView([22.5, 80], 4.4);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OpenStreetMap contributors", maxZoom: 19,
    }).addTo(mapMain);
  }
  locations.forEach((loc) => {
    const color = riskColor(loc.risk_level);
    if (mapMarkers[loc.id]) {
      mapMarkers[loc.id].setStyle({ color, fillColor: color });
      mapMarkers[loc.id].setPopupContent(popupHtml(loc));
    } else {
      const marker = L.circleMarker([loc.lat, loc.lon], {
        radius: 7, color, fillColor: color, fillOpacity: 0.85, weight: 1,
      }).addTo(mapMain);
      marker.bindPopup(popupHtml(loc));
      marker.on("click", () => selectLocation(loc.id));
      mapMarkers[loc.id] = marker;
    }
  });
}

// ---------------------------------------------------------------- PREDICTION (flagship — simple language)

async function renderPrediction() {
  const nameEl = document.getElementById("prediction-location-name");
  if (!selectedLocationId && !selectedLocationLatLon) {
    document.getElementById("prediction-empty").style.display = "block";
    document.getElementById("prediction-content").style.display = "none";
    return;
  }

  try {
    let risk, locationLabel;
    if (selectedLocationId) {
      const d = await api(`/api/risk/${selectedLocationId}`);
      risk = d.risk;
      locationLabel = `${d.location.village}, ${d.location.district}, ${d.location.state}`;
    } else {
      const d = await api(`/api/explore?lat=${selectedLocationLatLon.lat}&lon=${selectedLocationLatLon.lon}`);
      if (d.mode === "monitored") {
        selectedLocationId = d.location_id;
        return renderPrediction();
      }
      locationLabel = selectedLocationLatLon.name || `${selectedLocationLatLon.lat.toFixed(2)}, ${selectedLocationLatLon.lon.toFixed(2)}`;
      risk = {
        score: d.estimated_risk ? d.estimated_risk.value : null,
        level: d.estimated_risk_level, lead_time_hours: null,
        reasons: ["Estimated from nearby monitored areas — not an exact local prediction."],
        estimated: true,
      };
    }

    nameEl.textContent = locationLabel;
    document.getElementById("prediction-empty").style.display = "none";
    document.getElementById("prediction-content").style.display = "block";

    document.getElementById("pred-chance").textContent = risk.score != null ? `${Math.round(risk.score)}%` : "–";
    document.getElementById("pred-chance").style.color = riskColor(risk.level);
    document.getElementById("pred-band").innerHTML = `<span class="badge ${risk.level}">${risk.level}</span>`;

    const timeEl = document.getElementById("pred-time");
    const timeNote = document.getElementById("pred-time-note");
    if (risk.lead_time_hours) {
      timeEl.textContent = `~${risk.lead_time_hours} hours`;
      timeNote.textContent = "This is the model's estimate, not a guarantee — always follow official evacuation orders.";
    } else if (risk.level === "LOW" || risk.level === "MEDIUM") {
      timeEl.textContent = risk.level === "LOW" ? "No immediate concern" : "No countdown given";
      timeNote.textContent = risk.level === "LOW"
        ? "Conditions look calm right now. Keep checking back, especially during heavy rain."
        : "Risk is elevated but not clearly heading toward a disaster \u2014 we only estimate a countdown once risk reaches HIGH or CRITICAL, so this number isn't overclaiming certainty.";
    } else {
      timeEl.textContent = "Calculating\u2026";
      timeNote.textContent = "";
    }

    const plain = document.getElementById("pred-plain-text");
    const reasons = risk.reasons || [];
    if (risk.estimated) {
      plain.textContent = `${locationLabel} is not one of our closely-monitored spots, so this is an estimate based on nearby areas, not an exact local prediction. Estimated level: ${risk.level}.`;
    } else if (risk.level === "LOW") {
      plain.textContent = `Right now, ${locationLabel} looks safe. Our system checks rainfall, river levels, soil wetness and more every 15 minutes, and nothing concerning is showing up.`;
    } else {
      plain.textContent = `${locationLabel} currently has a ${Math.round(risk.score)}% chance of a flood or landslide. ` +
        (reasons.length ? `The main reasons: ${reasons.slice(0, 2).join("; ").toLowerCase()}.` : "");
    }

    document.getElementById("pred-accuracy-text").textContent =
      "In our tests against real past events, the model correctly flagged danger in roughly 7 out of 10 cases (see full numbers in Risk Analysis → Model Accuracy). No prediction system is perfect — always treat official warnings as final.";

    try {
      const dd = await api(`/api/dosdonts?hazard=flood`);
      document.getElementById("pred-quick-dos").innerHTML =
        `<ul class="reason-list">${dd.do.slice(0, 3).map((x) => `<li>${x}</li>`).join("")}</ul>`;
    } catch (e) { /* non-critical */ }

  } catch (e) {
    console.error(e);
    document.getElementById("prediction-empty").style.display = "block";
    document.getElementById("prediction-empty").textContent = "Couldn't load a prediction for this location yet.";
    document.getElementById("prediction-content").style.display = "none";
  }
}

document.getElementById("prediction-use-gps").addEventListener("click", async () => {
  const loc = await getLocation();
  if (loc) selectLocation(null, { lat: loc.lat, lon: loc.lon, name: "Your location" });
});

// ---------------------------------------------------------------- RISK ANALYSIS (detailed — city search first)

document.getElementById("risk-city-go").addEventListener("click", () => {
  const q = document.getElementById("risk-city-input").value.trim();
  if (q.length >= 2) runSearch(q, document.getElementById("risk-city-results"), (id, pt) => {
    if (id) { selectedLocationId = id; loadRiskDetail(id); }
    else if (pt) { selectedLocationLatLon = pt; renderPrediction(); document.querySelector('.nav-item[data-page="prediction"]').click(); }
  });
});
document.getElementById("risk-city-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") document.getElementById("risk-city-go").click();
});

async function loadRiskDetail(id) {
  try {
    const d = await api(`/api/risk/${id}`);
    document.getElementById("risk-city-prompt").style.display = "none";
    document.getElementById("risk-detail").style.display = "block";

    document.getElementById("risk-detail-title").textContent =
      `${d.location.village}, ${d.location.district}, ${d.location.state}`;
    document.getElementById("risk-detail-score").textContent = d.risk.score + "%";
    document.getElementById("risk-detail-score").style.color = riskColor(d.risk.level);
    document.getElementById("risk-detail-badge").innerHTML = `<span class="badge ${d.risk.level}">${d.risk.level}</span>`;
    document.getElementById("risk-detail-updated").textContent = "Updated " + fmtTime(d.risk.last_updated);
    document.getElementById("risk-flood-p").textContent = d.risk.flood_probability + "%";
    document.getElementById("risk-slide-p").textContent = d.risk.landslide_probability + "%";
    document.getElementById("risk-lead-time").textContent = d.risk.lead_time_hours
      ? `~${d.risk.lead_time_hours} hours` : "Not shown at LOW/MEDIUM — no clear hazard is imminent";
    const ltFormula = d.risk.formula && d.risk.formula.lead_time;
    document.getElementById("risk-lead-time-formula").innerHTML = ltFormula
      ? `<div class="mono" style="font-size:11px;margin-bottom:6px;">${ltFormula.formula}</div>
         <table class="data-table" style="font-size:12px;">
           ${ltFormula.steps.map((s) => `<tr><td>${s.label}</td><td class="mono">${s.value}${s.unit}</td></tr>`).join("")}
         </table>
         <p style="margin-top:6px;">${ltFormula.note}</p>` : "";

    const reasonsEl = document.getElementById("risk-reasons");
    reasonsEl.innerHTML = "";
    d.risk.reasons.forEach((r) => reasonsEl.appendChild(el("li", null, r)));

    renderImportanceChart(d.risk.feature_importance);
    renderTrendChart(d.risk_history);
    renderFormula(d.risk.formula);
    renderIoT(d.sensors, d.weather);
    loadValidation();
    loadModelInfo();
  } catch (e) {
    console.error(e);
  }
}

function renderFormula(formula) {
  const box = document.getElementById("risk-formula-block");
  if (!formula) { box.innerHTML = `<div class="text-muted text-sm">Formula not available yet.</div>`; return; }
  box.innerHTML = `
    <div class="mono text-sm" style="margin-bottom:12px;color:var(--text-secondary);">${formula.expression}</div>
    <table class="data-table">
      ${formula.steps.map((s) => `<tr><td>${s.label}</td><td class="mono">${s.value} ${s.unit}</td></tr>`).join("")}
    </table>
    <p class="text-sm text-muted" style="margin-top:10px;">${formula.why_this_weighting}</p>`;
}

function renderIoT(sensors, weather) {
  const box = document.getElementById("iot-readings");
  const tag = document.getElementById("iot-source-tag");
  if (!sensors) { box.innerHTML = `<div class="text-muted text-sm">No sensor data yet.</div>`; return; }
  tag.textContent = sensors.source || "DEMO";
  tag.style.background = (sensors.source || "").includes("LIVE") ? "var(--risk-low-soft)" : "var(--risk-medium-soft)";
  tag.style.color = (sensors.source || "").includes("LIVE") ? "var(--risk-low)" : "var(--risk-medium)";
  box.innerHTML = `
    <div class="stat-tile"><div class="num">${sensors.soil_moisture}%</div><div class="label">Soil moisture</div></div>
    <div class="stat-tile"><div class="num">${sensors.water_level}m</div><div class="label">Water level sensor <span class="demo-tag">DEMO</span></div></div>
    <div class="stat-tile"><div class="num">${sensors.iot_rainfall}mm</div><div class="label">IoT rain gauge <span class="demo-tag">DEMO</span></div></div>`;
}

function renderImportanceChart(data) {
  const ctx = document.getElementById("chart-importance");
  if (chartImportance) chartImportance.destroy();
  chartImportance = new Chart(ctx, {
    type: "bar",
    data: { labels: data.map((d) => d.feature), datasets: [{ data: data.map((d) => d.importance), backgroundColor: "#3d8bfd" }] },
    options: {
      indexAxis: "y", plugins: { legend: { display: false } },
      scales: { x: { ticks: { color: "#a9b8d6" }, grid: { color: "#24314f" } }, y: { ticks: { color: "#a9b8d6" }, grid: { display: false } } },
    },
  });
}

function renderTrendChart(history) {
  const ctx = document.getElementById("chart-trend");
  if (chartTrend) chartTrend.destroy();
  chartTrend = new Chart(ctx, {
    type: "line",
    data: { labels: history.map((h) => fmtTime(h.timestamp)), datasets: [{ data: history.map((h) => h.risk_score), borderColor: "#3d8bfd", tension: 0.3, pointRadius: 2 }] },
    options: {
      plugins: { legend: { display: false } },
      scales: { y: { min: 0, max: 100, ticks: { color: "#a9b8d6" }, grid: { color: "#24314f" } }, x: { ticks: { color: "#a9b8d6" }, grid: { display: false } } },
    },
  });
}

async function loadValidation() {
  try {
    const d = await api("/api/validation/2025");
    const b = d.random_forest_backtest;
    const box = document.getElementById("validation-block");
    box.innerHTML = `
      <p class="text-sm text-muted">${d.validated_on}</p>
      <div class="grid grid-2">
        ${["flood", "landslide"].map((k) => `
          <div>
            <div class="panel-title" style="margin-bottom:6px;">${k.toUpperCase()}</div>
            <table class="data-table">
              <tr><td>Events tested</td><td>${b[k].events_tested}</td></tr>
              <tr><td>Correctly detected</td><td>${b[k].correctly_detected}</td></tr>
              <tr><td>Missed events</td><td>${b[k].missed_events}</td></tr>
              <tr><td>False alarms</td><td>${b[k].false_alarms}</td></tr>
              <tr><td>Precision</td><td>${b[k].precision}</td></tr>
              <tr><td>Recall</td><td>${b[k].recall}</td></tr>
              <tr><td>F1</td><td>${b[k].f1}</td></tr>
            </table>
          </div>`).join("")}
      </div>`;
  } catch (e) { console.error(e); }
}

async function loadModelInfo() {
  try {
    const d = await api("/api/model-info");
    const box = document.getElementById("model-info-block");
    box.innerHTML = `
      <table class="data-table">
        <tr><td>Algorithm</td><td>${d.algorithm}</td></tr>
        <tr><td>Model class (from the actual object)</td><td class="mono">${d.model_type}</td></tr>
        <tr><td>Number of decision trees</td><td class="mono">${d.n_estimators}</td></tr>
        <tr><td>Max tree depth</td><td class="mono">${d.max_depth}</td></tr>
        <tr><td>Input features</td><td class="mono">${d.n_features}</td></tr>
        <tr><td>Training rows</td><td class="mono">${d.training_data_rows ?? "n/a"}</td></tr>
        <tr><td>Monitored locations</td><td class="mono">${d.monitored_locations}</td></tr>
        <tr><td>Trained at</td><td class="mono">${d.trained_at ? new Date(d.trained_at).toLocaleString() : "n/a"}</td></tr>
        <tr><td>scikit-learn (train time / now)</td><td class="mono">${d.sklearn_version_used_to_train} / ${d.sklearn_version_running_now}</td></tr>
      </table>
      <p class="text-sm text-muted" style="margin-top:8px;">${d.note}</p>`;
  } catch (e) { console.error(e); }
}

// ---------------------------------------------------------------- LIVE WEATHER (pure live data)

async function loadWeatherFor(id) {
  try {
    const d = await api(`/api/weather/${id}`);
    document.getElementById("weather-empty").style.display = "none";
    document.getElementById("weather-card").style.display = "block";
    document.getElementById("w-temp").textContent = d.temperature.toFixed(1);
    document.getElementById("w-rain24").textContent = d.rainfall_24h.toFixed(1);
    document.getElementById("w-humidity").textContent = d.humidity.toFixed(0);
    document.getElementById("w-forecast").textContent = d.rainfall_forecast_6h.toFixed(1);
    document.getElementById("w-wind").textContent = (d.wind_speed ?? 0).toFixed(1);
    document.getElementById("w-gusts").textContent = (d.wind_gusts ?? 0).toFixed(1);
    document.getElementById("w-dewpoint").textContent = (d.dew_point ?? 0).toFixed(1);
    document.getElementById("w-pressure").textContent = (d.pressure ?? 0).toFixed(0);
    document.getElementById("w-uv").textContent = (d.uv_index ?? 0).toFixed(1);
    document.getElementById("w-visibility").textContent = ((d.visibility ?? 10000) / 1000).toFixed(1);
    document.getElementById("w-cloud").textContent = (d.cloud_cover ?? 0).toFixed(0);
    document.getElementById("w-winddir").textContent = (d.wind_direction ?? 0).toFixed(0);
    document.getElementById("w-extreme-alert").style.display = d.extreme_rainfall_alert ? "block" : "none";
    document.getElementById("w-source-info").textContent = `Source: ${d.source}`;
  } catch (e) { console.error(e); }
}

// ---------------------------------------------------------------- MODEL MONITOR

async function loadMonitor() {
  try {
    const [activity, status] = await Promise.all([api("/api/pipeline/activity?limit=25"), api("/api/pipeline/status")]);
    const feed = document.getElementById("activity-feed");
    feed.innerHTML = "";
    activity.activity.forEach((a) => {
      const row = el("div", "activity-row");
      row.innerHTML = `<span class="t">${fmtTime(a.timestamp)}</span><span class="s">${a.step}</span><span>${a.detail}</span>`;
      feed.appendChild(row);
    });
    const list = document.getElementById("source-status-list");
    list.innerHTML = "";
    status.sources.forEach((s) => {
      const row = el("div", "source-row");
      row.innerHTML = `<span>${s.source}</span><span class="status-chip ${s.status}">${s.status}</span>`;
      list.appendChild(row);
    });
    document.getElementById("next-cycle-label").textContent = `pipeline · every ${status.next_cycle_minutes} min`;
  } catch (e) { console.error(e); }
  loadHistoricalValidation();
}

async function loadHistoricalValidation(refresh) {
  const summaryBox = document.getElementById("hist-validation-summary");
  const tableBox = document.getElementById("hist-validation-table");
  const noteBox = document.getElementById("hist-validation-note");
  tableBox.innerHTML = `<div class="text-muted text-sm">${refresh ? "Running fresh check against real historical weather\u2026 (needs internet, ~15-30s)" : "Loading\u2026"}</div>`;
  try {
    const d = await api(`/api/historical-validation${refresh ? "?refresh=true" : ""}`);
    const s = d.summary;
    summaryBox.innerHTML = `
      <div class="stat-tile"><div class="num">${s.total_events}</div><div class="label">Real events tested</div></div>
      <div class="stat-tile"><div class="num">${s.successfully_scored}</div><div class="label">Successfully scored</div></div>
      <div class="stat-tile ${s.accuracy_pct >= 70 ? "low" : s.accuracy_pct >= 40 ? "medium" : "high"}"><div class="num">${s.accuracy_pct != null ? s.accuracy_pct + "%" : "\u2013"}</div><div class="label">Correctly flagged HIGH/CRITICAL</div></div>
      <div class="stat-tile"><div class="num">${s.fetch_errors}</div><div class="label">Fetch errors (needs internet)</div></div>`;

    if (s.successfully_scored === 0) {
      tableBox.innerHTML = `<div class="callout warn">Couldn't reach Open-Meteo's Archive API from this machine (no internet, or blocked). This check needs real outbound internet to run \u2014 try again on a connected network. Real event citations are listed below regardless.</div>`;
    } else {
      tableBox.innerHTML = "";
    }
    const table = el("table", "data-table");
    table.innerHTML = `<tr><th>Event</th><th>Date</th><th>Actual</th><th>Predicted</th><th>Match</th><th>Source</th></tr>` +
      d.events.map((e) => `<tr>
        <td>${e.place}, ${e.state}</td><td class="mono">${e.date}</td>
        <td><span class="badge ${e.actual_severity}">${e.actual_severity}</span></td>
        <td>${e.predicted_level ? `<span class="badge ${e.predicted_level}">${e.predicted_level}</span> ${e.predicted_score}%` : `<span class="text-muted">fetch failed</span>`}</td>
        <td>${e.matched === true ? "✅" : e.matched === false ? "❌" : "\u2013"}</td>
        <td><a href="${e.source}" target="_blank" rel="noopener">verify ↗</a></td>
      </tr>`).join("");
    tableBox.appendChild(table);
    noteBox.textContent = `${d.methodology} ${d.from_cache ? "(showing cached results \u2014 click 'Run fresh check' to re-verify live)" : ""}`;
  } catch (e) {
    tableBox.innerHTML = `<div class="callout warn">Couldn't load validation results: ${e.message}</div>`;
  }
}
document.getElementById("hist-validation-refresh").addEventListener("click", () => loadHistoricalValidation(true));

// ---------------------------------------------------------------- EMERGENCY & SOS (merged)

async function loadEmergency() {
  const center = selectedLocationId ? await api(`/api/risk/${selectedLocationId}`).then((d) => d.location)
    : selectedLocationLatLon || { lat: 28.6, lon: 77.2 };
  try {
    const d = await api(`/api/hospitals?lat=${center.lat}&lon=${center.lon}`);
    const source = d.facilities[0]?.source || "";
    document.getElementById("facility-source-tag").innerHTML = source.includes("Demo")
      ? `<span class="demo-tag">${source}</span>` : `<span class="text-muted text-sm">via ${source}</span>`;

    if (!mapEmergency) {
      mapEmergency = L.map("map-emergency").setView([center.lat, center.lon], 12);
      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { attribution: "&copy; OpenStreetMap contributors", maxZoom: 19 }).addTo(mapEmergency);
    } else {
      mapEmergency.setView([center.lat, center.lon], 12);
    }
    const icons = { hospital: "🏥", clinic: "🏥", police: "👮", fire_station: "🚒" };
    const list = document.getElementById("facility-list");
    list.innerHTML = "";
    d.facilities.forEach((f) => {
      L.marker([f.lat, f.lon]).addTo(mapEmergency).bindPopup(`${icons[f.type] || "📍"} ${f.name}`);
      const row = el("div", "risk-row");
      row.innerHTML = `<span>${icons[f.type] || "📍"}</span><div class="place"><div class="name">${f.name}</div><div class="sub">${f.type}</div></div>`;
      list.appendChild(row);
    });
  } catch (e) { console.error(e); }

  try {
    const c = await api("/api/emergency-contacts");
    const box = document.getElementById("contacts-list");
    box.innerHTML = "";
    c.contacts.forEach((ct) => {
      const row = el("div", "source-row");
      row.innerHTML = `<span>${ct.name}</span><span class="mono">${ct.number}</span>`;
      box.appendChild(row);
    });
  } catch (e) { console.error(e); }

  loadActiveSOS();
  loadCriticalAreaUsers(center);
}

async function loadCriticalAreaUsers(center) {
  const box = document.getElementById("critical-area-users");
  try {
    let isCritical = false;
    if (selectedLocationId) {
      const d = await api(`/api/risk/${selectedLocationId}`);
      isCritical = d.risk.level === "CRITICAL";
    }
    if (!isCritical) {
      box.innerHTML = `<div class="text-muted text-sm">No CRITICAL area selected yet.</div>`;
      return;
    }
    const d = await api(`/api/nearby-users?lat=${center.lat}&lon=${center.lon}&radius_km=5`);
    box.innerHTML = "";
    if (!d.nearby_users.length) {
      box.innerHTML = `<div class="text-muted text-sm">This area is CRITICAL, but no other registered users are nearby yet.</div>`;
      return;
    }
    d.nearby_users.forEach((u) => {
      const row = el("div", "risk-row");
      row.innerHTML = `<span class="dot CRITICAL"></span><div class="place"><div class="name">User ${u.id.slice(0, 6)}</div><div class="sub">${u.distance_km}km away · ${u.precision}</div></div>`;
      box.appendChild(row);
    });
  } catch (e) { box.innerHTML = `<div class="text-muted text-sm">Couldn't load nearby users.</div>`; }
}

document.getElementById("sos-trigger").addEventListener("click", () => {
  const name = document.getElementById("sos-name").value || "Anonymous";
  const place = document.getElementById("sos-place").value || "Unknown location";
  const notify_phone = document.getElementById("sos-notify-phone").value.trim() || null;
  const resultBox = document.getElementById("sos-result");
  resultBox.innerHTML = `<div class="text-muted text-sm">Requesting location permission…</div>`;

  navigator.geolocation.getCurrentPosition(
    async (pos) => {
      const { latitude, longitude } = pos.coords;
      try {
        const d = await api("/api/sos", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name, place, lat: latitude, lon: longitude, notify_phone }),
        });
        const directNote = notify_phone ? `<br>Registered number SMS: ${d.direct_contact_sms_sent ? "sent" : "not sent (" + d.channels.sms + ")"}` : "";
        resultBox.innerHTML = `<div class="callout warn">${d.message}<br><br>
          <span class="text-sm">${d.notified_nearby_users.length} nearby user(s) matched. SMS: ${d.channels.sms}${directNote}</span></div>`;
        loadActiveSOS();
      } catch (e) { resultBox.innerHTML = `<div class="callout warn">Could not send SOS: ${e.message}</div>`; }
    },
    (err) => {
      resultBox.innerHTML = `<div class="callout warn">Location permission denied or unavailable (${err.message}). Try the offline SMS button below instead.</div>`;
    }
  );
});

// Offline-capable SOS: pre-fills the phone's own SMS app using cellular signal, no internet needed
function updateSmsLink() {
  const name = document.getElementById("sos-name").value || "Someone";
  const place = document.getElementById("sos-place").value || "unknown location";
  const cached = localStorage.getItem("dhara_last_gps");
  let coordsText = "location unknown — please share manually";
  if (cached) {
    const c = JSON.parse(cached);
    coordsText = `https://maps.google.com/?q=${c.lat},${c.lon}`;
  }
  const body = encodeURIComponent(`SOS: ${name} needs help at ${place}. Location: ${coordsText}`);
  document.getElementById("sos-sms-link").href = `sms:112?body=${body}`;
}
document.getElementById("sos-name").addEventListener("input", updateSmsLink);
document.getElementById("sos-place").addEventListener("input", updateSmsLink);

async function loadActiveSOS() {
  try {
    const d = await api("/api/sos/active");
    const list = document.getElementById("active-sos-list");
    list.innerHTML = d.alerts.length ? "" : `<div class="text-muted text-sm">No active alerts.</div>`;
    d.alerts.forEach((a) => {
      const row = el("div", "risk-row");
      row.innerHTML = `<span class="dot CRITICAL"></span><div class="place"><div class="name">${a.name}</div><div class="sub">${a.place} · ${fmtTime(a.timestamp)}</div></div>`;
      list.appendChild(row);
    });
  } catch (e) { console.error(e); }
}


// ---------------------------------------------------------------- REGISTER (simplified — no donation)

document.getElementById("reg-submit").addEventListener("click", async () => {
  const name = document.getElementById("reg-name").value.trim();
  const phone = document.getElementById("reg-phone").value.trim();
  const email = document.getElementById("reg-email").value.trim();
  const box = document.getElementById("reg-result");
  if (!name || !phone) { box.innerHTML = `<div class="callout warn">Name and mobile number are required.</div>`; return; }
  try {
    const loc = await getLocation();
    const d = await api("/api/register", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, phone, email, lat: loc?.lat, lon: loc?.lon }),
    });
    localStorage.setItem("dhara_user_id", d.user_id);
    box.innerHTML = `<div class="callout">Registered! We'll text ${phone} if a disaster is predicted near you.</div>`;
  } catch (e) { box.innerHTML = `<div class="callout warn">Registration failed: ${e.message}</div>`; }
});

// ---------------------------------------------------------------- FLOATING DHARA-AI CHAT (predictions + news + Q&A)

const chatPanel = document.getElementById("chat-panel");
document.getElementById("chat-float-btn").addEventListener("click", () => {
  chatPanel.classList.toggle("open");
  if (chatPanel.classList.contains("open") && !chatPanel.dataset.greeted) {
    chatPanel.dataset.greeted = "1";
    const log = document.getElementById("chat-log");
    log.appendChild(el("div", "chat-msg bot", "Hi! Ask me about a state/village's risk, search news (e.g. \"news Kerala floods\"), or ask general safety questions."));
  }
});
document.getElementById("chat-panel-close").addEventListener("click", () => chatPanel.classList.remove("open"));

document.getElementById("chat-send").addEventListener("click", sendChat);
document.getElementById("chat-input").addEventListener("keydown", (e) => { if (e.key === "Enter") sendChat(); });

async function sendChat() {
  const input = document.getElementById("chat-input");
  const q = input.value.trim();
  if (!q) return;
  const log = document.getElementById("chat-log");
  log.appendChild(el("div", "chat-msg user", q));
  input.value = "";
  log.scrollTop = log.scrollHeight;

  const isNewsQuery = /\bnews\b/i.test(q);
  try {
    if (isNewsQuery) {
      const topic = q.replace(/\bnews\b/gi, "").trim() || "flood India";
      const d = await api(`/api/news?q=${encodeURIComponent(topic)}`);
      if (!d.configured) {
        log.appendChild(el("div", "chat-msg bot", d.note));
      } else if (d.summary) {
        log.appendChild(el("div", "chat-msg bot", d.summary.replace(/\n/g, "<br>")));
      } else if (d.articles.length) {
        log.appendChild(el("div", "chat-msg bot", d.articles.slice(0, 3).map((a) => `• ${a.title}`).join("<br>")));
      } else {
        log.appendChild(el("div", "chat-msg bot", "No recent news found for that."));
      }
    } else {
      const d = await api(`/api/bot?q=${encodeURIComponent(q)}`);
      log.appendChild(el("div", "chat-msg bot", d.answer));
    }
  } catch (e) {
    log.appendChild(el("div", "chat-msg bot", "Sorry, I couldn't reach the backend."));
  }
  log.scrollTop = log.scrollHeight;
}

// ---------------------------------------------------------------- init + polling

async function refreshAll() {
  await loadDashboard();
  if (document.getElementById("page-monitor").classList.contains("active")) loadMonitor();
  if (selectedLocationId) {
    if (document.getElementById("page-risk").classList.contains("active")) loadRiskDetail(selectedLocationId);
    if (document.getElementById("page-prediction").classList.contains("active")) renderPrediction();
    if (document.getElementById("page-weather").classList.contains("active")) loadWeatherFor(selectedLocationId);
  }
}

window.addEventListener("online", refreshAll);
window.addEventListener("offline", () => document.getElementById("offline-banner").classList.add("show"));

loadDashboard();
loadMonitor();
getLocation(); // populate GPS status pill early, cache for offline SOS use
setInterval(refreshAll, REFRESH_MS);
