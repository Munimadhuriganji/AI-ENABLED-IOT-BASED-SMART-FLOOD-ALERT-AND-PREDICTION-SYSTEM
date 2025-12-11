// static/js/dashboard.js
// Improved dashboard script: click snapshot -> update level + rainfall charts
// Requires Chart.js (v3+) loaded in the page (see instructions below).

(() => {
  const API_LATEST = "/latest-readings";
  const API_TS = (node) => `/timeseries/${encodeURIComponent(node)}`;
  const AUTO_REFRESH_MS = 5000;

  // Chart instances
  let levelChart = null;
  let rainChart = null;
  let refreshTimer = null;
  let currentNode = "node1";

  // UTIL: format ISO -> short label
  function shortLabel(iso) {
    try {
      const d = new Date(iso);
      // show hours:minutes or date if older
      const now = Date.now();
      if (Math.abs(now - d.getTime()) < 24 * 3600 * 1000) {
        return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      }
      return d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit" });
    } catch (e) {
      return iso;
    }
  }

  // Downsample / aggregate series into <= maxPoints for tidy charts
  function aggregateSeries(timestamps, values, maxPoints = 50, method = "sum") {
    const n = values.length;
    if (n <= maxPoints) {
      return { ts: timestamps.slice(), vals: values.slice() };
    }
    const bucketSize = Math.ceil(n / maxPoints);
    const outTs = [];
    const outVals = [];
    for (let i = 0; i < n; i += bucketSize) {
      const sliceTs = timestamps.slice(i, i + bucketSize);
      const sliceVals = values.slice(i, i + bucketSize);
      // pick middle timestamp as label
      outTs.push(sliceTs[Math.floor(sliceTs.length / 2)]);
      if (method === "sum") {
        outVals.push(sliceVals.reduce((a, b) => a + (Number(b) || 0), 0));
      } else {
        outVals.push(sliceVals.reduce((a, b) => a + (Number(b) || 0), 0) / sliceVals.length);
      }
    }
    return { ts: outTs, vals: outVals };
  }

  // Create or update the level line chart (with EWMA datasets)
  function renderLevelChart(labels, data, ewmaMean = [], ewmaThreshold = []) {
    const ctx = document.getElementById("chart-level").getContext("2d");
    const formattedLabels = labels.map(shortLabel);

    const datasets = [
      {
        label: "Water Level (m)",
        data: data,
        fill: false,
        tension: 0.15,
        pointRadius: 3,
        borderWidth: 2,
        borderColor: "rgba(54, 162, 235, 1)",
        backgroundColor: "rgba(54, 162, 235, 0.2)",
      },
      {
        label: "EWMA",
        data: ewmaMean,
        fill: false,
        tension: 0.15,
        pointRadius: 0,
        borderWidth: 2,
        borderColor: "rgba(75, 192, 192, 1)",
        borderDash: [4, 2],
      },
      {
        label: "EWMA Threshold",
        data: ewmaThreshold,
        fill: false,
        tension: 0.15,
        pointRadius: 0,
        borderWidth: 2,
        borderColor: "rgba(255, 99, 132, 1)",
        borderDash: [6, 3],
      },
    ];

    if (levelChart) {
      levelChart.data.labels = formattedLabels;
      levelChart.data.datasets = datasets;
      levelChart.update();
      return;
    }

    levelChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: formattedLabels,
        datasets: datasets,
      },
      options: {
        maintainAspectRatio: false,
        scales: {
          x: {
            ticks: {
              autoSkip: true,
              maxRotation: 0,
              minRotation: 0,
              maxTicksLimit: 10,
            },
          },
          y: {
            title: { display: true, text: "Level (m)" },
            beginAtZero: false,
          },
        },
        plugins: {
          legend: { display: true, position: "top" },
          tooltip: { mode: "index", intersect: false },
        },
      },
    });
  }

  // Create or update the rainfall bar chart (aggregated)
  function renderRainChart(labels, data) {
    const ctx = document.getElementById("chart-rain").getContext("2d");
    const formattedLabels = labels.map(shortLabel);

    const dataset = {
      label: "Rain (mm)",
      data: data,
      borderWidth: 0.5,
      barPercentage: 0.9,
      categoryPercentage: 0.9,
    };

    if (rainChart) {
      rainChart.data.labels = formattedLabels;
      rainChart.data.datasets = [dataset];
      rainChart.update();
      return;
    }

    rainChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: formattedLabels,
        datasets: [dataset],
      },
      options: {
        maintainAspectRatio: false,
        scales: {
          x: {
            ticks: {
              autoSkip: true,
              maxRotation: 90,
              minRotation: 0,
              maxTicksLimit: 12,
            },
          },
          y: {
            beginAtZero: true,
            title: { display: true, text: "Rain (mm)" },
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: { mode: "index", intersect: false },
        },
      },
    });
  }

  // Fetch and show the timeseries for a node
  async function updateChartsForNode(node) {
    try {
      const resp = await fetch(API_TS(node));
      if (!resp.ok) {
        console.warn("timeseries fetch failed", resp.status);
        return;
      }
      const js = await resp.json();
      const series = js.series || {};
      const timestamps = (series.timestamps || []).slice(); // oldest->newest expected
      const levels = (series.levels || []).map((v) => (v === null || v === undefined ? NaN : Number(v)));
      const rains = (series.rains || []).map((v) => (v === null || v === undefined ? 0 : Number(v)));
      const ewmaMean = (series.ewma_mean || []).map((v) => (v === null || v === undefined ? NaN : Number(v)));
      const ewmaThreshold = (series.ewma_threshold || []).map((v) => (v === null || v === undefined ? NaN : Number(v)));

      // If timestamps missing or empty, try series array of objects
      if ((!timestamps || timestamps.length === 0) && Array.isArray(js.series)) {
        const objs = js.series;
        for (const o of objs) {
          timestamps.push(o.ts);
          levels.push(o.level !== undefined && o.level !== null ? Number(o.level) : NaN);
          rains.push(o.rain !== undefined && o.rain !== null ? Number(o.rain) : 0);
        }
      }

      // Trim NaNs and ensure equal lengths
      const cleanT = [];
      const cleanL = [];
      const cleanR = [];
      const cleanEwmaMean = [];
      const cleanEwmaThreshold = [];
      for (let i = 0; i < Math.max(levels.length, rains.length, timestamps.length); i++) {
        const t = timestamps[i] || timestamps[timestamps.length - 1] || new Date().toISOString();
        const lv = i < levels.length ? levels[i] : (levels.length ? levels[levels.length - 1] : NaN);
        const rv = i < rains.length ? rains[i] : 0;
        const em = i < ewmaMean.length ? ewmaMean[i] : NaN;
        const et = i < ewmaThreshold.length ? ewmaThreshold[i] : NaN;
        cleanT.push(t);
        cleanL.push(Number.isFinite(lv) ? lv : NaN);
        cleanR.push(Number.isFinite(rv) ? rv : 0);
        cleanEwmaMean.push(Number.isFinite(em) ? em : NaN);
        cleanEwmaThreshold.push(Number.isFinite(et) ? et : NaN);
      }

      // For level chart: sample down to max 200 points (preserve shape)
      let levelLabels = cleanT;
      let levelVals = cleanL;
      let ewmaMeanVals = cleanEwmaMean;
      let ewmaThresholdVals = cleanEwmaThreshold;
      if (levelVals.length > 200) {
        const agg = aggregateSeries(levelLabels, levelVals, 200, "avg");
        const aggEm = aggregateSeries(levelLabels, cleanEwmaMean, 200, "avg");
        const aggEt = aggregateSeries(levelLabels, cleanEwmaThreshold, 200, "avg");
        levelLabels = agg.ts;
        levelVals = agg.vals;
        ewmaMeanVals = aggEm.vals;
        ewmaThresholdVals = aggEt.vals;
      }

      // For rain chart: aggregate to max 50 bars (sum makes sense for rainfall)
      let rainLabels = cleanT;
      let rainVals = cleanR;
      if (rainVals.length > 50) {
        const aggR = aggregateSeries(rainLabels, rainVals, 50, "sum");
        rainLabels = aggR.ts;
        rainVals = aggR.vals;
      }

      renderLevelChart(levelLabels, levelVals, ewmaMeanVals, ewmaThresholdVals);
      renderRainChart(rainLabels, rainVals);
    } catch (e) {
      console.error("updateChartsForNode error", e);
    }
  }

  // Fetch latest readings and populate snapshot card(s)
  async function refreshLatest() {
    try {
      const resp = await fetch(API_LATEST);
      if (!resp.ok) {
        console.warn("latest-readings fetch failed", resp.status);
        return;
      }
      const js = await resp.json();
      const readings = js.readings || (js.latest ? [js.latest] : []);
      // Update the live snapshot display(s) - elements with .node-snapshot and data-node
      const snapshotEls = document.querySelectorAll(".node-snapshot");
      snapshotEls.forEach((el) => {
        const node = el.dataset.node;
        // find matching reading
        const found = readings.find((r) => r.node_id === node) || readings[0] || null;
        if (found) {
          const lvl = (found.level === null || found.level === undefined) ? "—" : Number(found.level).toFixed(2);
          const rain = (found.rain === null || found.rain === undefined) ? "—" : Number(found.rain).toFixed(2);
          const ts = found.ts ? new Date(found.ts).toISOString() : "";
          const snapNode = el.querySelector('.snap-node');
          if (snapNode) snapNode.textContent = `Node: ${found.node_id}`;
          const snapLevel = el.querySelector('.snap-level');
          if (snapLevel) snapLevel.textContent = `Level: ${lvl} m`;
          const snapRain = el.querySelector('.snap-rain');
          if (snapRain) snapRain.textContent = `Rain: ${rain} mm`;
          const snapTs = el.querySelector('.snap-ts');
          if (snapTs) snapTs.textContent = ts;
        }
      });

      // if no user-selected node, set currentNode to first reading's node
      if ((!currentNode || currentNode === "") && readings.length) {
        currentNode = readings[0].node_id;
      }

      // If the current node not present in readings but readings exist, set to first
      if (readings.length && !readings.some(r => r.node_id === currentNode)) {
        currentNode = readings[0].node_id;
      }

      // Update charts for currentNode
      await updateChartsForNode(currentNode);

    } catch (e) {
      console.error("refreshLatest error", e);
    }
  }

  // Attach click handlers to snapshot elements to change node
  function attachSnapshotClicks() {
    document.addEventListener("click", (ev) => {
      // If clicked inside element with data-node, pick it up
      let el = ev.target;
      while (el && el !== document.body) {
        if (el.dataset && el.dataset.node) {
          const node = el.dataset.node;
          if (node && node !== currentNode) {
            currentNode = node;
            // visual feedback: add .selected to snaps
            document.querySelectorAll(".node-snapshot").forEach(e => e.classList.remove("selected"));
            const chosen = document.querySelector(`.node-snapshot[data-node="${node}"]`);
            if (chosen) chosen.classList.add("selected");
            updateChartsForNode(node);
          }
          break;
        }
        el = el.parentElement;
      }
    });
  }

  // Initialization
  function init() {
    // Ensure canvas elements exist (create fallback if missing)
    if (!document.getElementById("chart-level") || !document.getElementById("chart-rain")) {
      console.error("chart canvas elements #chart-level and #chart-rain required in dashboard.html");
      return;
    }

    attachSnapshotClicks();
    refreshLatest();
    refreshAlerts();
    updateMLSummary();
    // periodic refresh
    if (refreshTimer) clearInterval(refreshTimer);
    refreshTimer = setInterval(() => {
      refreshLatest();
      refreshAlerts();
    }, AUTO_REFRESH_MS);
  }

  // Fetch and display alerts in the dashboard sidebar
  async function refreshAlerts() {
    try {
      const resp = await fetch("/api/alerts");
      if (!resp.ok) {
        console.warn("alerts fetch failed", resp.status);
        return;
      }
      const js = await resp.json();
      const alerts = js.alerts || [];

      const listEl = document.getElementById("latest-alerts-list");
      const countEl = document.getElementById("latest-alerts-count");

      if (!listEl) return;

      // Update count
      if (countEl) countEl.textContent = alerts.length;

      if (alerts.length === 0) {
        listEl.innerHTML = '<li class="empty-state text-muted small">No alerts recorded yet.</li>';
        return;
      }

      // Show latest 10 alerts
      const recentAlerts = alerts.slice(0, 10);
      listEl.innerHTML = recentAlerts.map(a => {
        const riskClass = a.risk === "High" ? "text-danger" : (a.risk === "Medium" ? "text-warning" : "text-success");
        const ts = a.ts ? new Date(a.ts).toLocaleString() : "";
        return `<li class="mb-2 pb-2 border-bottom">
          <div class="d-flex justify-content-between">
            <strong class="${riskClass}">${a.risk || "Unknown"}</strong>
            <small class="text-muted">${a.node_id || ""}</small>
          </div>
          <small class="text-muted">${ts}</small>
          ${a.reason ? `<div class="small">${a.reason}</div>` : ""}
        </li>`;
      }).join("");
    } catch (e) {
      console.error("refreshAlerts error", e);
    }
  }

  // Update ML Summary section
  function updateMLSummary() {
    const summaryEl = document.getElementById("ml-summary");
    if (!summaryEl) return;

    // Display basic ML status - in production this would fetch from an API
    summaryEl.innerHTML = `
      <div class="small">
        <div class="d-flex justify-content-between mb-1">
          <span>ML Model:</span>
          <span class="text-success">Active</span>
        </div>
        <div class="d-flex justify-content-between mb-1">
          <span>EWMA Detector:</span>
          <span class="text-success">Active</span>
        </div>
        <div class="d-flex justify-content-between">
          <span>Static Rules:</span>
          <span class="text-success">Active</span>
        </div>
      </div>
    `;
  }

  // Run on DOM ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
