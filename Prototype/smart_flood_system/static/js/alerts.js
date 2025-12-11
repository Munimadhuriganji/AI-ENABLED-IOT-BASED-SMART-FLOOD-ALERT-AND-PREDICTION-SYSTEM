// static/js/alerts.js
// Handles Alert History page behaviour for FloodWatch

let alertsCache = [];

// Map risk label -> badge class (keep in sync with style.css)
function riskClass(risk) {
  if (!risk) return "badge bg-secondary";
  const r = String(risk).toLowerCase();
  if (r.startsWith("high")) return "badge bg-danger";
  if (r.startsWith("medium")) return "badge bg-warning text-dark";
  if (r.startsWith("low")) return "badge bg-success";
  return "badge bg-secondary";
}

// Fetch alerts JSON from backend
async function fetchAlerts() {
  try {
    const res = await fetch("/api/alerts");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    // backend may return either an array or {alerts: [...]}
    const list = Array.isArray(data) ? data : (data.alerts || []);
    alertsCache = list;
    renderAlerts();
    updateMetrics();
  } catch (err) {
    console.error("Failed to fetch alerts:", err);
  }
}

// Update the four metrics boxes
function updateMetrics() {
  const total = alertsCache.length;
  let high = 0, medium = 0, low = 0;

  alertsCache.forEach(a => {
    const risk = String(a.risk || "").toLowerCase();
    if (risk.startsWith("high") || risk.startsWith("critical")) {
      high++;
    } else if (risk.startsWith("medium")) {
      medium++;
    } else if (risk.startsWith("low")) {
      low++;
    }
  });

  // Update the metric elements
  const totalEl = document.getElementById("metric-total");
  const highEl = document.getElementById("metric-high");
  const mediumEl = document.getElementById("metric-medium");
  const lowEl = document.getElementById("metric-low");
  const countEl = document.getElementById("alerts-count");

  if (totalEl) totalEl.textContent = total;
  if (highEl) highEl.textContent = high;
  if (mediumEl) mediumEl.textContent = medium;
  if (lowEl) lowEl.textContent = low;
  if (countEl) countEl.textContent = total;
}

// Apply search + filter and render table body
function renderAlerts() {
  const tbody = document.getElementById("alerts-tbody");
  if (!tbody) return;

  const filterSelect = document.getElementById("risk-filter");
  const searchInput = document.getElementById("search-node");
  const statsEl = document.getElementById("alert-stats");

  const riskFilter = filterSelect ? filterSelect.value : "all";
  const query = searchInput ? searchInput.value.trim().toLowerCase() : "";

  let filtered = alertsCache;

  if (riskFilter !== "all") {
    filtered = filtered.filter(a =>
      String(a.risk || "")
        .toLowerCase()
        .startsWith(riskFilter)
    );
  }

  if (query) {
    filtered = filtered.filter(a =>
      String(a.node_id || "")
        .toLowerCase()
        .includes(query)
    );
  }

  // update tiny stats label
  if (statsEl) {
    statsEl.textContent = `${filtered.length} of ${alertsCache.length} alerts shown`;
  }

  // build rows
  tbody.innerHTML = "";
  if (!filtered.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 5;
    td.className = "text-center text-muted";
    td.textContent = "No alerts matching your criteria.";
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }

  filtered.forEach((a, idx) => {
    const tr = document.createElement("tr");

    const tdIdx = document.createElement("td");
    tdIdx.textContent = String(idx + 1);

    const tdNode = document.createElement("td");
    tdNode.textContent = a.node_id || "-";

    const tdRisk = document.createElement("td");
    const spanRisk = document.createElement("span");
    spanRisk.className = riskClass(a.risk);
    spanRisk.textContent = a.risk || "Unknown";
    tdRisk.appendChild(spanRisk);

    const tdProb = document.createElement("td");
    if (a.prob !== undefined && a.prob !== null) {
      const p = Number(a.prob);
      tdProb.textContent = isNaN(p) ? "-" : p.toFixed(2);
    } else {
      tdProb.textContent = "-";
    }

    const tdTime = document.createElement("td");
    tdTime.textContent = a.ts || a.time || "-";

    const tdReason = document.createElement("td");
    tdReason.textContent = a.reason || "-";

    tr.appendChild(tdIdx);
    tr.appendChild(tdNode);
    tr.appendChild(tdRisk);
    tr.appendChild(tdProb);
    tr.appendChild(tdTime);
    tr.appendChild(tdReason);
    tbody.appendChild(tr);
  });
}

// Hook up filters and auto-refresh
document.addEventListener("DOMContentLoaded", () => {
  const filterSelect = document.getElementById("risk-filter");
  const searchInput = document.getElementById("search-node");

  if (filterSelect) {
    filterSelect.addEventListener("change", renderAlerts);
  }
  if (searchInput) {
    searchInput.addEventListener("input", () => {
      // tiny debounce
      if (searchInput._timer) clearTimeout(searchInput._timer);
      searchInput._timer = setTimeout(renderAlerts, 200);
    });
  }

  // Initial load + polling
  fetchAlerts();
  setInterval(fetchAlerts, 5000);
});
