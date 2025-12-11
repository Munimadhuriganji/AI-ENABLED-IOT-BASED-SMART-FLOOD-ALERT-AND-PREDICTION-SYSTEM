#!/usr/bin/env python3
# app.py - corrected for Gunicorn/Container startup

from __future__ import annotations
import os
import io
import sqlite3
import json
import logging
from datetime import datetime, timezone
from functools import wraps

from flask import Flask, request, jsonify, render_template, send_file, g, abort
from werkzeug.utils import secure_filename

# ML deps
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Optional Twilio
TWILIO_ENABLED = False
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_ENABLED = True
except Exception:
    TWILIO_ENABLED = False

# -----------------------
# Config / environment
# -----------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "floodwatch.db")

ADMIN_KEY = os.environ.get("ADMIN_KEY", "supersecret123")

TWILIO_SID = os.environ.get("TWILIO_SID", "")
TWILIO_TOKEN = os.environ.get("TWILIO_TOKEN", "")
TWILIO_FROM = os.environ.get("TWILIO_FROM", "")
SMS_TO = os.environ.get("SMS_TO", "")

EWMA_ALPHA = float(os.environ.get("EWMA_ALPHA", "0.1"))
EWMA_K = float(os.environ.get("EWMA_K", "3.0"))

MODEL_PATH = os.path.join(PROJECT_ROOT, "dt_model.pkl")
CALIBRATED_MODEL_PATH = os.path.join(PROJECT_ROOT, "calibrated_model.pkl")

# Logging
LOG_PATH = os.path.join(PROJECT_ROOT, "floodwatch.log")
logger = logging.getLogger("floodwatch")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(LOG_PATH)
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh.setFormatter(fmt)
ch.setFormatter(fmt)
# Avoid duplicate handlers when reloading
if not logger.handlers:
    logger.addHandler(fh)
    logger.addHandler(ch)

# -----------------------
# DB helpers
# -----------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        g.db.row_factory = sqlite3.Row
    return g.db

def init_db():
    logger.info("Initializing DB and checking tables...")
    db = sqlite3.connect(DB_PATH)
    c = db.cursor()
    # Create sensor table: store raw readings (allow NULLs to avoid strict insert failures)
    c.execute("""
      CREATE TABLE IF NOT EXISTS sensor (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        node_id TEXT,
        level REAL,
        rain REAL,
        pre_alert INTEGER DEFAULT 0,
        ts TEXT
      )
    """)
    # Alerts table
    c.execute("""
      CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        node_id TEXT,
        risk TEXT,
        prob REAL,
        ts TEXT,
        reason TEXT
      )
    """)
    # indexes
    c.execute("CREATE INDEX IF NOT EXISTS idx_sensor_node_ts ON sensor(node_id, ts);")
    c.execute("CREATE INDEX IF NOT EXISTS idx_alerts_node_ts ON alerts(node_id, ts);")
    db.commit()
    db.close()
    logger.info("DB ready at %s", DB_PATH)

# teardown is registered below after app is created

# -----------------------
# Flask app
# -----------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["JSON_SORT_KEYS"] = False

# proper teardown registered on the app instance
@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()

# -----------------------
# Model loading & helpers
# -----------------------
model = None
model_is_classifier = False
label_map = {0: "NoRisk", 1: "Low", 2: "High"}  # default

def try_load_model():
    global model, model_is_classifier, label_map
    loaded = False
    for p in (CALIBRATED_MODEL_PATH, MODEL_PATH):
        if os.path.exists(p):
            try:
                model = joblib.load(p)
                logger.info("Loaded ML model from %s", p)
                loaded = True
                break
            except Exception as e:
                logger.exception("Failed to load model %s: %s", p, e)
    if not loaded:
        logger.warning("No ML model found (dt_model.pkl / calibrated_model.pkl). ML inference will be disabled.")
        model = None
        return

    model_is_classifier = hasattr(model, "predict")
    try:
        if hasattr(model, "classes_"):
            classes = list(model.classes_)
            # if classes are ints, map to default labels if possible
            if all(isinstance(c, (int, np.integer)) for c in classes):
                label_map_local = {}
                for c in classes:
                    label_map_local[int(c)] = label_map.get(int(c), str(c))
                label_map = label_map_local
            else:
                # non-int classes -> we'll treat predictions as strings
                label_map = None
            logger.info("Model 'classes_' found: %s", classes)
    except Exception:
        logger.exception("Could not read model.classes_")

def safe_predict_prob(X):
    """
    Return (pred_label, prob) where:
     - pred_label can be int OR string depending on model.
     - prob is float for 'critical/high' class where available.
    """
    if model is None:
        return None, None

    Xarr = np.array(X).reshape(1, -1)
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(Xarr)[0]
            idx = int(np.argmax(probs))
            pred = model.classes_[idx] if hasattr(model, "classes_") else idx
            prob_of_crit = None
            if hasattr(model, "classes_"):
                for i, cls in enumerate(model.classes_):
                    if (isinstance(cls, (int, np.integer)) and int(cls) == 2) or (isinstance(cls, str) and str(cls).lower().startswith("high")):
                        prob_of_crit = float(probs[i])
                        break
            if prob_of_crit is None:
                prob_of_crit = float(max(probs))
            return pred, float(prob_of_crit)
        else:
            pred = model.predict(Xarr)[0]
            return pred, 1.0
    except Exception:
        logger.exception("safe_predict_prob failed")
        try:
            pred = model.predict(Xarr)[0]
            return pred, 1.0
        except Exception:
            return None, None

# -----------------------
# Utility functions
# -----------------------
def now_iso():
    return datetime.now(timezone.utc).isoformat()

def compute_ewma_threshold(recent_levels, alpha=EWMA_ALPHA, k=EWMA_K):
    mu = None
    var = 0.0
    if not recent_levels:
        return {"mu": None, "sigma": None, "threshold": None}
    for v in recent_levels:
        if mu is None:
            mu = float(v)
            var = 0.0
        else:
            diff = float(v) - mu
            mu = alpha * float(v) + (1 - alpha) * mu
            var = alpha * (diff * diff) + (1 - alpha) * var
    sigma = (var ** 0.5) if var > 0 else 0.0
    threshold = mu + k * sigma
    return {"mu": mu, "sigma": sigma, "threshold": threshold}

def fetch_recent_levels(node_id, limit=30):
    db = get_db()
    c = db.cursor()
    c.execute("SELECT ts, level, rain FROM sensor WHERE node_id=? ORDER BY ts DESC LIMIT ?", (node_id, limit))
    rows = c.fetchall()
    return [dict(r) for r in reversed(rows)]

# simple admin decorator
def require_admin(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        key = request.args.get("admin_key") or request.headers.get("X-ADMIN-KEY") or request.form.get("admin_key")
        if not key or key != ADMIN_KEY:
            logger.warning("Unauthorized admin attempt from %s", request.remote_addr)
            return abort(401)
        return f(*args, **kwargs)
    return wrap

# -----------------------
# Routes: UI pages
# -----------------------
@app.route("/")
@app.route("/dashboard")
def index():
    return render_template("dashboard.html")

@app.route("/alerts")
def alerts_page():
    return render_template("alerts.html")

# ----------------------------------------------------------------------
# API endpoint for alerts (JSON)
# ----------------------------------------------------------------------
@app.route("/api/alerts")
def alerts_api():
    """Return recent alerts as JSON.
    The front‑end expects either a list or an object with an ``alerts`` key.
    """
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute("SELECT node_id, risk, prob, ts, reason FROM alerts ORDER BY ts DESC")
        rows = cur.fetchall()
        alerts = [dict(r) for r in rows]
        # Return a consistent shape – both formats are accepted by alerts.js
        return jsonify({"alerts": alerts})
    except Exception as e:
        logger.exception("Failed to fetch alerts JSON")
        return jsonify({"alerts": []}), 500

# health check
@app.route("/health")
def health():
    return jsonify({"status": "ok", "db_exists": os.path.exists(DB_PATH)})

# -----------------------
# API: ingestion + queries
# -----------------------
@app.route("/ingest", methods=["POST"])
def ingest():
    db = get_db()
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "invalid json"}), 400

    node = payload.get("node_id") or payload.get("node") or "node1"
    try:
        level = float(payload.get("level")) if payload.get("level") is not None else None
    except Exception:
        level = None
    try:
        rain = float(payload.get("rain")) if payload.get("rain") is not None else None
    except Exception:
        rain = None
    pre_alert = 1 if payload.get("pre_alert") else 0
    ts = payload.get("ts") or now_iso()

    try:
        c = db.cursor()
        c.execute("INSERT INTO sensor(node_id, level, rain, pre_alert, ts) VALUES (?,?,?,?,?)",
                  (node, level, rain, pre_alert, ts))
        db.commit()
    except Exception:
        logger.exception("DB insert failed")
        return jsonify({"error": "db insert failed"}), 500

    recent = fetch_recent_levels(node, limit=60)
    levels = [r["level"] for r in recent if r["level"] is not None]
    ewma = compute_ewma_threshold(levels[-60:], alpha=EWMA_ALPHA, k=EWMA_K)

    prob = None
    predicted_label_name = None
    try:
        if levels:
            level_now = levels[-1]
            level_first = levels[0] if len(levels) >= 1 else level_now
            delta = float(level_now) - float(level_first)
            rain_sum = 0.0
            try:
                rain_sum = sum([float(r["rain"] or 0.0) for r in recent])
            except Exception:
                rain_sum = 0.0
            feat = [level_now, delta, rain_sum]
            pred_raw, prob = safe_predict_prob(feat)
            if pred_raw is None:
                predicted_label_name = None
            else:
                if isinstance(pred_raw, (int, np.integer)):
                    predicted_label_name = label_map.get(int(pred_raw), str(pred_raw)) if label_map else str(pred_raw)
                else:
                    predicted_label_name = str(pred_raw)
        else:
            predicted_label_name = None
    except Exception:
        logger.exception("ML inference failed")

    risk = "Low"
    reason = ""
    try:
        if prob is not None:
            if prob >= 0.9:
                risk = "High"
                reason = f"ML prob {prob:.2f}"
            elif prob >= 0.6:
                risk = "Medium"
                reason = f"ML prob {prob:.2f}"
            else:
                risk = "Low"
                reason = f"ML prob {prob:.2f}"
        try:
            if ewma.get("threshold") is not None and level is not None:
                if float(level) > float(ewma["threshold"]):
                    risk = "High"
                    reason = (reason + " | EWMA breach").strip(" |")
        except Exception:
            pass

        if pre_alert:
            if risk == "Low":
                risk = "Medium"
            reason = (reason + " | pre_alert").strip(" |")
    except Exception:
        logger.exception("Risk determination failed")

    if risk in ("Medium", "High"):
        try:
            c = db.cursor()
            c.execute("INSERT INTO alerts(node_id, risk, prob, ts, reason) VALUES (?,?,?,?,?)",
                      (node, risk, float(prob) if prob is not None else None, ts, reason))
            db.commit()
            send_alert_notification(node, risk, prob)
        except Exception:
            logger.exception("Failed to insert alert row")

    return jsonify({"status": "ok", "ewma": ewma, "predicted": predicted_label_name, "prob": prob})

def send_alert_notification(node, risk, prob):
    text = f"ALERT {risk} @{node} prob={prob if prob is not None else 'N/A'}"
    if TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM and SMS_TO and TWILIO_ENABLED:
        try:
            client = TwilioClient(TWILIO_SID, TWILIO_TOKEN)
            client.messages.create(body=text, from_=TWILIO_FROM, to=SMS_TO)
            logger.info("[SMS] Sent Twilio alert: %s", text)
            return
        except Exception as e:
            logger.error("Twilio send failed: %s. Falling back to stub.", e)
    logger.info("[SMS-STUB] -> %s: %s", SMS_TO or "+XXXXXXXX", text)

# Latest readings endpoint
@app.route("/latest-readings", methods=["GET"])
def latest_readings():
    db = get_db()
    c = db.cursor()
    try:
        c.execute("""
            SELECT node_id, level, rain, pre_alert, ts FROM (
              SELECT node_id, level, rain, pre_alert, ts,
                     ROW_NUMBER() OVER (PARTITION BY node_id ORDER BY ts DESC) rn
              FROM sensor
            ) WHERE rn=1
        """)
        rows = c.fetchall()
        readings = [dict(r) for r in rows]
    except Exception:
        c.execute("SELECT node_id, level, rain, pre_alert, ts FROM sensor ORDER BY ts DESC")
        rows = c.fetchall()
        latest = {}
        for r in rows:
            node = r["node_id"]
            if node not in latest:
                latest[node] = r
        readings = list(latest.values())

    latest = readings[0] if readings else None
    node = request.args.get("node") or (latest["node_id"] if latest else None) or "node1"
    series = fetch_recent_levels(node, limit=200)
    timestamps = [r["ts"] for r in series]
    levels = [r["level"] for r in series]
    rains = [r["rain"] for r in series]
    c2 = db.cursor()
    c2.execute("SELECT node_id, risk, prob, ts, reason FROM alerts ORDER BY ts DESC LIMIT 50")
    alerts = [dict(r) for r in c2.fetchall()]

    # Compute EWMA for the last N levels
    ewma_mu = []
    ewma_thr = []
    mu = None
    var = 0.0
    alpha = EWMA_ALPHA
    k = EWMA_K

    for v in levels:
        if v is None:
            ewma_mu.append(None)
            ewma_thr.append(None)
            continue
        if mu is None:
            mu = float(v)
            var = 0.0
        else:
            diff = float(v) - mu
            mu = alpha * float(v) + (1 - alpha) * mu
            var = alpha * (diff * diff) + (1 - alpha) * var

        sigma = (var ** 0.5) if var > 0 else 0.0
        thr = mu + k * sigma

        ewma_mu.append(mu)
        ewma_thr.append(thr)

    return jsonify({
        "readings": readings,
        "latest": latest,
        "series": {
            "timestamps": timestamps,
            "levels": levels,
            "rains": rains,
            "ewma_mean": ewma_mu,
            "ewma_threshold": ewma_thr
        },
        "alerts": alerts,
        "ml_summary": None
    })

@app.route("/get-alerts", methods=["GET"])
def get_alerts():
    db = get_db()
    c = db.cursor()
    c.execute("SELECT node_id, risk, prob, ts, reason FROM alerts ORDER BY ts DESC LIMIT 1000")
    rows = c.fetchall()
    alerts = [dict(r) for r in rows]
    return jsonify({"alerts": alerts})

@app.route("/timeseries/<node_id>", methods=["GET"])
def timeseries(node_id):
    series = fetch_recent_levels(node_id, limit=500)
    
    # Extract levels and compute EWMA
    levels = [r["level"] for r in series]
    timestamps = [r["ts"] for r in series]
    rains = [r["rain"] for r in series]
    
    ewma_mu = []
    ewma_thr = []
    mu = None
    var = 0.0
    alpha = EWMA_ALPHA
    k = EWMA_K

    for v in levels:
        if v is None:
            ewma_mu.append(None)
            ewma_thr.append(None)
            continue
        if mu is None:
            mu = float(v)
            var = 0.0
        else:
            diff = float(v) - mu
            mu = alpha * float(v) + (1 - alpha) * mu
            var = alpha * (diff * diff) + (1 - alpha) * var

        sigma = (var ** 0.5) if var > 0 else 0.0
        thr = mu + k * sigma

        ewma_mu.append(mu)
        ewma_thr.append(thr)

    return jsonify({
        "series": {
            "timestamps": timestamps,
            "levels": levels,
            "rains": rains,
            "ewma_mean": ewma_mu,
            "ewma_threshold": ewma_thr
        }
    })

@app.route("/admin/ewma-plot")
def ewma_plot():
    node = request.args.get("node", "node1")
    n = int(request.args.get("n", "200"))
    recent = fetch_recent_levels(node, limit=n)
    levels = [r["level"] for r in recent if r["level"] is not None]
    timestamps = [r["ts"] for r in recent]

    if not levels:
        plt.figure(figsize=(8,3))
        plt.text(0.5, 0.5, "No data for node "+node, ha="center", va="center")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        plt.close()
        return send_file(buf, mimetype="image/png", download_name=f"ewma_{node}.png")

    mu = None
    var = 0.0
    ewma_vals = []
    thr_vals = []
    alpha = EWMA_ALPHA
    k = EWMA_K
    for v in levels:
        if mu is None:
            mu = float(v)
            var = 0.0
        else:
            diff = float(v) - mu
            mu = alpha * float(v) + (1 - alpha) * mu
            var = alpha * (diff * diff) + (1 - alpha) * var
        sigma = (var ** 0.5) if var > 0 else 0.0
        thr = mu + k * sigma
        ewma_vals.append(mu)
        thr_vals.append(thr)

    plt.figure(figsize=(10,3.5))
    plt.plot(range(len(levels)), levels, label="Level", linewidth=1.6)
    plt.plot(range(len(ewma_vals)), ewma_vals, label=f"EWMA (α={alpha})", linewidth=1.2)
    plt.plot(range(len(thr_vals)), thr_vals, label=f"Threshold (k={k})", linestyle="--", linewidth=1.2)
    plt.legend()
    plt.title(f"EWMA & threshold — {node}")
    plt.xlabel("sample index (oldest->newest)")
    plt.ylabel("level")
    plt.grid(alpha=0.12)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    plt.close()
    return send_file(buf, mimetype="image/png", download_name=f"ewma_{node}.png")

@app.route("/export-db")
def export_db():
    if os.path.exists(DB_PATH):
        return send_file(DB_PATH, as_attachment=True, download_name="floodwatch.db")
    else:
        return jsonify({"error": "db not found"}), 404

@app.route("/plots/<path:filename>")
def serve_plots(filename):
    plots_dir = os.path.join(PROJECT_ROOT, "plots")
    filepath = os.path.join(plots_dir, secure_filename(filename))
    if os.path.exists(filepath):
        return send_file(filepath)
    return abort(404)

# -----------------------
# Initialize at import-time (works for Gunicorn)
# -----------------------
try:
    init_db()
except Exception:
    logger.exception("init_db failed during import")

try:
    try_load_model()
except Exception:
    logger.exception("try_load_model failed during import")

# -----------------------
# Run dev server only when executed directly
# -----------------------
if __name__ == "__main__":
    # do a quick check
    logger.info("Starting Flask dev server (debug=False)")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
