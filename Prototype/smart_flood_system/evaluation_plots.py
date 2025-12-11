#!/usr/bin/env python3
"""
evaluation_plots.py -- upgraded visualization script

Generates figures + metrics for:
- Static threshold detector
- EWMA dynamic detector
- ML model-based detector

Outputs into ./plots:
  - metrics.csv              -> table of precision/recall/F1/TP/FP/etc.
  - detector_metrics.png     -> bars of precision/recall/F1 per method
  - detector_timeline.png    -> sample-wise timeline of true vs predicted critical
  - live_ewma_example.png    -> EWMA vs level from real DB data (if DB available)

Run:
    python evaluation_plots.py
"""

from __future__ import annotations
import os
import sys
import math
import logging
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# use non-GUI backend (works in Docker/headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib
from sklearn.metrics import precision_recall_fscore_support

# ----------------- CONFIG -----------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(PROJECT_ROOT, "training_data.csv")
MODEL_FILES = [
    os.path.join(PROJECT_ROOT, "calibrated_model.pkl"),
    os.path.join(PROJECT_ROOT, "dt_model.pkl"),
]

DB_PATH = os.path.join(PROJECT_ROOT, "data", "floodwatch.db")

# Detector parameters (configurable via env)
STATIC_THRESHOLD = float(os.environ.get("STATIC_THRESHOLD", "26.0"))
EWMA_ALPHA = float(os.environ.get("EWMA_ALPHA", "0.1"))
EWMA_K = float(os.environ.get("EWMA_K", "3.0"))

SYNTH_ROWS = int(os.environ.get("SYNTH_ROWS", "6000"))
SEQUENCE_LEN = int(os.environ.get("SEQ_LEN", "12"))

# logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("evaluation_plots")

# ----------------- DATA GENERATION -----------------


def gen_sequences(
    n: int = 2000,
    seq_len: int = 12,
    flood_prob: float = 0.08,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic sequences consistent with compare_threshold.py:
      columns: level_now, delta, rain_sum, label
      label: 0=NoRisk, 1=Low, 2=High (critical)
    """
    np.random.seed(seed)
    rows = []
    for _ in range(n):
        base = 20.0 + np.random.randn() * 0.5
        peak = 0.0
        start = seq_len + 5
        if np.random.rand() < flood_prob:
            peak = np.random.uniform(6.0, 18.0)
            start = np.random.randint(2, seq_len - 2)

        lvl = base
        seq = []
        for t in range(seq_len):
            if t > start and peak > 0:
                lvl += peak / max(1.0, (seq_len - start)) + np.random.randn() * 0.3
                rain = np.random.uniform(0.5, 2.0)
            else:
                lvl += np.random.randn() * 0.2
                rain = max(0.0, np.random.randn() * 0.1)
            seq.append((lvl, rain))

        level_now = seq[-1][0]
        level_first = seq[0][0]
        delta = level_now - level_first
        rain_sum = sum(r for _, r in seq)

        if peak == 0:
            label = 0
        elif peak < 10.0:
            label = 1
        else:
            label = 2

        rows.append((level_now, delta, rain_sum, label))

    return pd.DataFrame(rows, columns=["level_now", "delta", "rain_sum", "label"])


def load_or_generate_dataset() -> pd.DataFrame:
    """
    Load training_data.csv if present, otherwise generate synthetic dataset.
    """
    if os.path.exists(TRAIN_CSV):
        log.info("Found training CSV at %s. Loading.", TRAIN_CSV)
        df = pd.read_csv(TRAIN_CSV)
        for col in ["level_now", "delta", "rain_sum", "label"]:
            if col not in df.columns:
                raise SystemExit(f"training_data.csv missing column: {col}")
        return df[["level_now", "delta", "rain_sum", "label"]].copy()

    log.info(
        "No training_data.csv found — generating synthetic dataset (n=%d)",
        SYNTH_ROWS,
    )
    df = gen_sequences(n=SYNTH_ROWS, seq_len=SEQUENCE_LEN)
    return df


# ----------------- MODEL -----------------


def try_load_model() -> Optional[Any]:
    """
    Load model from the first path that exists in MODEL_FILES.
    """
    for path in MODEL_FILES:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                log.info("Loaded model: %s", path)
                return model
            except Exception as e:
                log.exception("Failed to load model %s: %s", path, e)
    log.warning("No model (dt_model.pkl/calibrated_model.pkl) found. ML detector disabled.")
    return None


def is_critical_prediction(pred: Any) -> bool:
    """
    Interpret model prediction as 'critical' (label=2) or not.
    Handles ints, floats, and strings like 'High', 'Critical', '2', etc.
    """
    try:
        if isinstance(pred, (int, np.integer)):
            return int(pred) == 2
        if isinstance(pred, float) and math.isclose(pred, 2.0):
            return True
        s = str(pred).strip()
        if s.isdigit():
            return int(s) == 2
        s_lower = s.lower()
        if s_lower.startswith("high") or s_lower.startswith("critical"):
            return True
    except Exception:
        pass
    return False


def ml_detector(df: pd.DataFrame, model: Any) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns (binary_preds, prob_critical) where:
      binary_preds: np.array of 0/1 for predicted critical
      prob_critical: np.array of probabilities or None if unavailable
    """
    if model is None:
        return np.zeros(len(df), dtype=int), None

    X = df[["level_now", "delta", "rain_sum"]].to_numpy()
    preds_bin = np.zeros(len(df), dtype=int)
    prob_arr: Optional[np.ndarray] = None

    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)

            # find index of "critical" class if possible
            crit_index = None
            if hasattr(model, "classes_"):
                for i, cls in enumerate(model.classes_):
                    if isinstance(cls, (int, np.integer)) and int(cls) == 2:
                        crit_index = i
                        break
                    if isinstance(cls, str) and str(cls).lower().startswith("high"):
                        crit_index = i
                        break
            if crit_index is None:
                # fallback: use the argmax of the first row as "critical"
                crit_index = int(np.argmax(probs[0]))

            prob_crit = probs[:, crit_index]
            prob_arr = prob_crit
            preds_bin = (prob_crit >= 0.5).astype(int)

            # refine using raw labels if available
            try:
                raw_preds = model.predict(X)
                for i, rp in enumerate(raw_preds):
                    if is_critical_prediction(rp):
                        preds_bin[i] = 1
            except Exception:
                pass
            return preds_bin, prob_arr

        # no predict_proba
        raw_preds = model.predict(X)
        for i, rp in enumerate(raw_preds):
            preds_bin[i] = 1 if is_critical_prediction(rp) else 0
        return preds_bin, None

    except Exception as e:
        log.exception("ml_detector failed: %s", e)
        try:
            raw_preds = model.predict(X)
            for i, rp in enumerate(raw_preds):
                preds_bin[i] = 1 if is_critical_prediction(rp) else 0
        except Exception as e2:
            log.exception("ml_detector fallback failed: %s", e2)
            preds_bin = np.zeros(len(df), dtype=int)
        return preds_bin, prob_arr


# ----------------- DETECTORS -----------------


def static_detector(df: pd.DataFrame, thr: float) -> np.ndarray:
    return (df["level_now"] > thr).astype(int).to_numpy()


def ewma_detector(df: pd.DataFrame, alpha: float, k: float) -> np.ndarray:
    """
    Compute EWMA threshold and mark as critical if value > mu + k*sigma.
    
    For training data that isn't in time-series order, we use two approaches:
    1. Sort by level_now to simulate a realistic progression
    2. Also check if delta (rate of change) indicates rapid rise
    
    A sample is flagged as critical if either:
    - level_now exceeds the EWMA threshold, OR
    - delta > 5.0 (indicating rapid water rise)
    """
    # First, create a sorted copy to compute EWMA thresholds properly
    df_sorted = df.copy()
    df_sorted['orig_idx'] = range(len(df))
    df_sorted = df_sorted.sort_values('level_now').reset_index(drop=True)
    
    mu = None
    var = 0.0
    ewma_flags = np.zeros(len(df_sorted), dtype=int)
    
    for i, v in enumerate(df_sorted["level_now"].to_numpy()):
        v = float(v)
        if mu is None:
            mu = v
            var = 0.0
            sigma = 0.0
            thr = mu + k * sigma
            ewma_flags[i] = 1 if v > thr else 0
            continue
        diff = v - mu
        mu = alpha * v + (1 - alpha) * mu
        var = alpha * (diff * diff) + (1 - alpha) * var
        sigma = math.sqrt(var) if var > 0 else 0.0
        thr = mu + k * sigma
        ewma_flags[i] = 1 if v > thr else 0
    
    # Map back to original order
    preds = np.zeros(len(df), dtype=int)
    for i, orig_idx in enumerate(df_sorted['orig_idx']):
        preds[orig_idx] = ewma_flags[i]
    
    # Also flag samples with high delta (rapid rise indicator)
    delta_threshold = 5.0
    high_delta = (df["delta"] > delta_threshold).to_numpy().astype(int)
    
    # Combine: flag as critical if EWMA says so OR delta is high
    return np.maximum(preds, high_delta)


# ----------------- METRICS & PLOTS -----------------


def summarize(method: str, true: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    p, r, f, _ = precision_recall_fscore_support(
        true, pred, average="binary", zero_division=0
    )
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    tn = int(((pred == 0) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    return {
        "method": method,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def plot_detector_metrics(metrics: List[Dict[str, Any]], outpath: str) -> None:
    df = pd.DataFrame(metrics)
    labels = df["method"].tolist()
    prec = df["precision"].tolist()
    rec = df["recall"].tolist()
    f1 = df["f1"].tolist()

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(9, 4))
    plt.bar(x - width, prec, width, label="Precision")
    plt.bar(x, rec, width, label="Recall")
    plt.bar(x + width, f1, width, label="F1-score")

    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Detector performance on critical events")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    log.info("Wrote detector metrics plot to %s", outpath)


def plot_detector_timeline(
    true_crit: np.ndarray,
    preds_static: np.ndarray,
    preds_ewma: np.ndarray,
    preds_ml: np.ndarray,
    outpath: str,
    max_points: int = 200,
) -> None:
    n = min(max_points, len(true_crit))
    xs = np.arange(n)

    plt.figure(figsize=(10, 4))
    plt.step(xs, true_crit[:n], where="mid", label="True critical (label==2)", linewidth=1.5)
    plt.step(xs, preds_static[:n], where="mid", label="Static detector", linestyle="--", linewidth=1.0)
    plt.step(xs, preds_ewma[:n], where="mid", label="EWMA detector", linestyle="-.", linewidth=1.0)
    plt.step(xs, preds_ml[:n], where="mid", label="ML detector", linestyle=":", linewidth=1.5)

    plt.yticks([0, 1], ["No", "Yes"])
    plt.xlabel("Sample index")
    plt.ylabel("Critical predicted?")
    plt.title(f"Detector timeline (first {n} samples)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    log.info("Wrote detector timeline plot to %s", outpath)


# ----------------- LIVE EWMA FROM DB -----------------


def plot_live_ewma_example(outpath: str, node: str = "node1", n: int = 200) -> None:
    """
    Optional: uses real DB (data/floodwatch.db, table sensor) to show EWMA vs level.
    If DB or table missing, just logs a warning and does nothing.
    """
    if not os.path.exists(DB_PATH):
        log.warning("DB %s not found, skipping live_ewma_example.png", DB_PATH)
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
    except Exception as e:
        log.warning("Failed to open DB: %s", e)
        return

    try:
        cur.execute(
            "SELECT ts, level FROM sensor WHERE node_id=? ORDER BY ts DESC LIMIT ?",
            (node, n),
        )
        rows = cur.fetchall()
    except Exception as e:
        log.warning("DB query failed (probably no 'sensor' table): %s", e)
        conn.close()
        return

    conn.close()
    if not rows:
        log.warning("No rows for node %s in DB, skipping live EWMA plot", node)
        return

    # oldest -> newest
    rows = list(reversed(rows))
    levels = [float(r["level"]) for r in rows if r["level"] is not None]

    if not levels:
        log.warning("No valid level data for node %s, skipping live EWMA plot", node)
        return

    mu = None
    var = 0.0
    ewma_vals = []
    thr_vals = []

    for v in levels:
        if mu is None:
            mu = v
            var = 0.0
        else:
            diff = v - mu
            mu = EWMA_ALPHA * v + (1 - EWMA_ALPHA) * mu
            var = EWMA_ALPHA * (diff * diff) + (1 - EWMA_ALPHA) * var
        sigma = math.sqrt(var) if var > 0 else 0.0
        thr = mu + EWMA_K * sigma
        ewma_vals.append(mu)
        thr_vals.append(thr)

    xs = np.arange(len(levels))

    plt.figure(figsize=(10, 4))
    plt.plot(xs, levels, label="Level", linewidth=1.6)
    plt.plot(xs, ewma_vals, label=f"EWMA (α={EWMA_ALPHA})", linewidth=1.2)
    plt.plot(xs, thr_vals, label=f"Threshold (k={EWMA_K})", linestyle="--", linewidth=1.2)

    plt.xlabel("Sample index (oldest → newest)")
    plt.ylabel("Water level")
    plt.title(f"Live EWMA vs level — node {node}")
    plt.grid(alpha=0.15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    log.info("Wrote live EWMA example plot to %s", outpath)


# ----------------- MAIN -----------------


def main() -> None:
    log.info("evaluation_plots.py starting")

    # 1) Dataset
    df = load_or_generate_dataset()
    df["level_now"] = df["level_now"].astype(float)
    df["delta"] = df["delta"].astype(float)
    df["rain_sum"] = df["rain_sum"].astype(float)
    df["label"] = df["label"].astype(int)

    y = df["label"].to_numpy()
    true_crit = (y == 2).astype(int)  # label==2 => critical

    # 2) Detectors
    static_preds = static_detector(df, STATIC_THRESHOLD)
    ewma_preds = ewma_detector(df, EWMA_ALPHA, EWMA_K)

    model = try_load_model()
    ml_preds, ml_prob = ml_detector(df, model)

    # 3) Metrics
    metrics: List[Dict[str, Any]] = []
    metrics.append(summarize("Static", true_crit, static_preds))
    metrics.append(summarize("EWMA", true_crit, ewma_preds))
    metrics.append(summarize("ML", true_crit, ml_preds))

    metrics_csv = os.path.join(PLOTS_DIR, "metrics.csv")
    pd.DataFrame(metrics).to_csv(metrics_csv, index=False)
    log.info("Wrote metrics CSV to %s", metrics_csv)

    # 4) Plots
    detector_metrics_png = os.path.join(PLOTS_DIR, "detector_metrics.png")
    plot_detector_metrics(metrics, detector_metrics_png)

    detector_timeline_png = os.path.join(PLOTS_DIR, "detector_timeline.png")
    plot_detector_timeline(
        true_crit, static_preds, ewma_preds, ml_preds, detector_timeline_png
    )

    # 5) Optional: live EWMA plot from DB
    live_ewma_png = os.path.join(PLOTS_DIR, "live_ewma_example.png")
    plot_live_ewma_example(live_ewma_png, node="node1", n=200)

    # 6) Print table for console
    dfm = pd.DataFrame(metrics)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("\n=== Detector metrics (critical events) ===")
    print(dfm[["method", "precision", "recall", "f1", "tp", "fp"]].to_string(index=False))
    print(f"\nSaved plots in: {PLOTS_DIR}")
    log.info("evaluation_plots.py finished")


if __name__ == "__main__":
    main()
