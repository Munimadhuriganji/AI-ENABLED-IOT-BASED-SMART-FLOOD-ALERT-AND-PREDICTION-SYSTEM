#!/usr/bin/env python3
"""
compare_threshold.py -- upgraded evaluation script

What it does (short):
- Loads an ML model (dt_model.pkl or calibrated_model.pkl) if present.
- Loads real training_data.csv if available; otherwise generates a
  physics-guided synthetic dataset (configurable size).
- Runs three detectors:
    * STATIC threshold detector (level_now > STATIC)
    * EWMA dynamic detector (alpha, k)
    * ML classifier (model.predict / predict_proba)
- Computes precision/recall/f1 on the *critical* class (label==2).
- Writes plots/ewma_vs_static.png and plots/metrics.csv (metrics table)
- Prints a small summary to stdout.

Drop this file in your project root and run:
    python compare_threshold.py
"""

from __future__ import annotations
import os
import sys
import math
import json
import logging
from typing import List, Tuple, Any, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# ---------- Config ----------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_FILES = [os.path.join(PROJECT_ROOT, "calibrated_model.pkl"),
               os.path.join(PROJECT_ROOT, "dt_model.pkl")]
TRAIN_CSV = os.path.join(PROJECT_ROOT, "training_data.csv")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Detector params (tweak via env)
STATIC_THRESHOLD = float(os.environ.get("STATIC", "26.0"))
EWMA_ALPHA = float(os.environ.get("EWMA_ALPHA", "0.1"))
EWMA_K = float(os.environ.get("EWMA_K", "3.0"))

SYNTH_ROWS = int(os.environ.get("SYNTH_ROWS", "6000"))
SEQUENCE_LEN = int(os.environ.get("SEQ_LEN", "12"))

MODEL = None

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("compare_threshold")

# ---------- Data generation & helpers ----------


def gen_sequences(n: int = 2000, seq_len: int = 12, flood_prob: float = 0.08, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic sequences and convert to tabular features:
      level_now, delta (level_now - level_first), rain_sum, label
    label: 0 = NoRisk, 1 = Low, 2 = High (critical)
    """
    np.random.seed(seed)
    rows = []
    for i in range(n):
        base = 20.0 + np.random.randn() * 0.5
        peak = 0.0
        start = seq_len + 5
        if np.random.rand() < flood_prob:
            peak = np.random.uniform(6.0, 18.0)
            start = np.random.randint(2, seq_len - 2)
        lvl = base
        seq = []
        for t in range(seq_len):
            if t > start and peak > 0.0:
                lvl += peak / max(1.0, (seq_len - start)) + np.random.randn() * 0.3
                rain = np.random.uniform(0.5, 2.0)
            else:
                lvl += np.random.randn() * 0.2
                rain = max(0.0, np.random.randn() * 0.1)
            seq.append((lvl, rain))
        X_now = seq[-1][0]
        X_first = seq[0][0]
        delta = X_now - X_first
        rain_sum = sum(r for _, r in seq)
        # define label mapping consistent with training script:
        # if peak == 0 => 0, else (1 if peak < 10 else 2)
        label = 0
        if peak > 0:
            label = 1 if peak < 10.0 else 2
        rows.append((X_now, delta, rain_sum, label))
    return pd.DataFrame(rows, columns=["level_now", "delta", "rain_sum", "label"])


def load_or_generate_dataset() -> pd.DataFrame:
    """
    Try to read TRAIN_CSV if present, otherwise generate synthetic dataset.
    Expected CSV format (if present): columns level_now, delta, rain_sum, label
    """
    if os.path.exists(TRAIN_CSV):
        log.info("Found training CSV: %s. Loading.", TRAIN_CSV)
        df = pd.read_csv(TRAIN_CSV)
        # basic sanity checks
        for col in ["level_now", "delta", "rain_sum", "label"]:
            if col not in df.columns:
                raise SystemExit(f"training_data.csv missing column: {col}")
        return df[["level_now", "delta", "rain_sum", "label"]].copy()
    else:
        log.info("No training_data.csv found — generating synthetic dataset (n=%d).", SYNTH_ROWS)
        df = gen_sequences(n=SYNTH_ROWS, seq_len=SEQUENCE_LEN)
        return df


# ---------- Model utilities ----------

def try_load_model() -> Optional[Any]:
    """
    Load the first available model from MODEL_FILES. Returns model or None.
    """
    for p in MODEL_FILES:
        if os.path.exists(p):
            try:
                m = joblib.load(p)
                log.info("Loaded model: %s", p)
                return m
            except Exception as e:
                log.exception("Failed to load model %s: %s", p, e)
    log.warning("No model file found at %s. ML detector will be disabled.", MODEL_FILES)
    return None


def is_critical_prediction(pred: Any, model: Any = None) -> bool:
    """
    Given model prediction output, determine if it indicates 'critical' (label==2).
    Accepts int-like labels or string labels like 'High'/'high' or '2' strings.
    """
    try:
        # numeric labels
        if isinstance(pred, (int, np.integer)):
            return int(pred) == 2
        # float near 2
        if isinstance(pred, float) and math.isclose(pred, 2.0):
            return True
        # string labels: check if startswith 'high' or equal '2'
        s = str(pred).strip()
        if s.isdigit():
            return int(s) == 2
        if s.lower().startswith("high") or s.lower().startswith("critical"):
            return True
    except Exception:
        pass
    return False


# ---------- Detectors ----------

def static_detector(df: pd.DataFrame, static_thr: float = STATIC_THRESHOLD) -> np.ndarray:
    # returns binary array: 1 => predicted critical
    return (df["level_now"] > static_thr).astype(int).to_numpy()


def ewma_detector(df: pd.DataFrame, alpha: float = EWMA_ALPHA, k: float = EWMA_K) -> np.ndarray:
    """
    Compute EWMA (online) threshold per-row reading and mark as critical if value > mu + k*sigma.
    
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


def ml_detector(df: pd.DataFrame, model: Any) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Run ML model predictions. Returns (binary_preds, proba_of_critical_if_available)
    binary_preds: 1 if predicted critical (label==2), 0 otherwise
    proba: numpy array of probability-of-critical (or None if not available)
    """
    if model is None:
        return np.zeros(len(df), dtype=int), None

    X = df[["level_now", "delta", "rain_sum"]].to_numpy()
    # try predict_proba first
    proba_arr = None
    preds_bin = np.zeros(len(df), dtype=int)
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            # try to find index corresponding to 'critical' class (2 or 'High')
            class_index = None
            if hasattr(model, "classes_"):
                for i, cls in enumerate(model.classes_):
                    if (isinstance(cls, (int, np.integer)) and int(cls) == 2) or (isinstance(cls, str) and str(cls).lower().startswith("high")):
                        class_index = i
                        break
            # fallback: use highest-prob index as 'positive'
            if class_index is None:
                class_index = np.argmax(probs, axis=1)[0]  # not ideal, but fallback
            proba_of_crit = probs[:, class_index]
            proba_arr = proba_of_crit
            # binary preds: use threshold 0.5 on that prob
            preds_bin = (proba_of_crit >= 0.5).astype(int)
            # BUT we will also sanity-check using model.predict() class labels
            try:
                raw_preds = model.predict(X)
                for i, rp in enumerate(raw_preds):
                    if is_critical_prediction(rp, model=model):
                        preds_bin[i] = 1
            except Exception:
                pass
            return preds_bin, proba_arr
        else:
            raw_preds = model.predict(X)
            for i, rp in enumerate(raw_preds):
                preds_bin[i] = 1 if is_critical_prediction(rp, model=model) else 0
            return preds_bin, None
    except Exception as e:
        log.exception("ML detector failed: %s", e)
        # fallback to plain predict if predict_proba raised
        try:
            raw_preds = model.predict(X)
            for i, rp in enumerate(raw_preds):
                preds_bin[i] = 1 if is_critical_prediction(rp, model=model) else 0
            return preds_bin, None
        except Exception as e2:
            log.exception("ML fallback predict failed: %s", e2)
            return np.zeros(len(df), dtype=int), None


# ---------- Metrics & plotting ----------

def summarize(name: str, true: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    p, r, f, _ = precision_recall_fscore_support(true, pred, average="binary", zero_division=0)
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    tn = int(((pred == 0) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    return {"method": name, "precision": float(p), "recall": float(r), "f1": float(f), "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def write_metrics_csv(metrics: List[Dict[str, Any]], outpath: str):
    df = pd.DataFrame(metrics)
    df.to_csv(outpath, index=False)
    log.info("Wrote metrics table to %s", outpath)


def plot_tp_fp(metrics: List[Dict[str, Any]], outpath: str):
    labels = [m["method"] for m in metrics]
    tp = [m["tp"] for m in metrics]
    fp = [m["fp"] for m in metrics]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, tp, width, label="TP")
    plt.bar(x + width / 2, fp, width, label="FP")
    plt.xticks(x, labels)
    plt.ylabel("Count")
    plt.title("Detector comparison (critical events): TP vs FP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    log.info("Wrote TP/FP plot to %s", outpath)


# ---------- Main ----------

def main():
    log.info("compare_threshold.py starting")
    df = load_or_generate_dataset()
    # ensure types
    df["level_now"] = df["level_now"].astype(float)
    df["delta"] = df["delta"].astype(float)
    df["rain_sum"] = df["rain_sum"].astype(float)
    df["label"] = df["label"].astype(int)

    X = df[["level_now", "delta", "rain_sum"]].to_numpy()
    y = df["label"].to_numpy()
    true_crit = (y == 2).astype(int)

    # detectors
    static_preds = static_detector(df, STATIC_THRESHOLD)
    ewma_preds = ewma_detector(df, alpha=EWMA_ALPHA, k=EWMA_K)

    # ML
    global MODEL
    MODEL = try_load_model()
    ml_preds, ml_proba = ml_detector(df, MODEL)

    # Summaries
    metrics = []
    metrics.append(summarize("Static", true_crit, static_preds))
    metrics.append(summarize("EWMA", true_crit, ewma_preds))
    metrics.append(summarize("ML", true_crit, ml_preds))

    # Output metrics CSV
    metrics_csv = os.path.join(PLOTS_DIR, "metrics.csv")
    write_metrics_csv(metrics, metrics_csv)

    # Plot TP vs FP
    tp_fp_png = os.path.join(PLOTS_DIR, "ewma_vs_static.png")
    plot_tp_fp(metrics, tp_fp_png)

    # Print summary table
    dfm = pd.DataFrame(metrics)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("\n=== Detector metrics (on critical events) ===")
    print(dfm[["method", "precision", "recall", "f1", "tp", "fp"]].to_string(index=False))
    print(f"\nSaved plots to: {PLOTS_DIR}")
    log.info("Done.")

    # Extra: save a JSON summary for dashboards
    summary_json = os.path.join(PLOTS_DIR, "metrics_summary.json")
    with open(summary_json, "w") as f:
        json.dump({"metrics": metrics}, f, indent=2)
    log.info("Wrote JSON summary to %s", summary_json)


if __name__ == "__main__":
    main()
