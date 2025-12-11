#!/usr/bin/env python3
"""
model_training.py

Trains the ML model for FloodWatch using:
- Either training_data.csv (if present),
- Or a synthetic, physics-guided dataset (generated here).

Outputs:
- dt_model.pkl                     -> baseline decision tree model
- training_data.csv (if synthetic) -> so you can reuse the same data
- training_metrics.txt             -> simple metrics summary
- plots/feature_importances.png    -> bar chart of feature importance

Run:
    python model_training.py
"""

from __future__ import annotations
import os
import sys
import math
import logging
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

# optional calibration (best-effort)
try:
    from sklearn.calibration import CalibratedClassifierCV
    HAVE_CAL = True
except Exception:
    HAVE_CAL = False

# ----------------- CONFIG -----------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(PROJECT_ROOT, "training_data.csv")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

BASELINE_MODEL_PATH = os.path.join(PROJECT_ROOT, "dt_model.pkl")
CAL_MODEL_PATH = os.path.join(PROJECT_ROOT, "calibrated_model.pkl")
METRICS_TXT = os.path.join(PROJECT_ROOT, "training_metrics.txt")

SYNTH_ROWS = int(os.environ.get("SYNTH_ROWS", "6000"))
SEQ_LEN = int(os.environ.get("SEQ_LEN", "12"))
FLOOD_PROB = float(os.environ.get("FLOOD_PROB", "0.08"))
RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("model_training")


# ----------------- DATA GENERATION -----------------


def gen_sequences(
    n: int = 2000,
    seq_len: int = 12,
    flood_prob: float = 0.08,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic sequences consistent with evaluation scripts.

    Columns:
      - level_now: last level in the sequence
      - delta: level_now - first_level
      - rain_sum: total rain in sequence
      - label: 0=NoRisk, 1=Low, 2=High (critical)
    """
    np.random.seed(seed)
    rows = []
    for _ in range(n):
        base = 20.0 + np.random.randn() * 0.5
        peak = 0.0
        start = seq_len + 5

        # decide if this sequence has a 'flood event'
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

        level_now = seq[-1][0]
        level_first = seq[0][0]
        delta = level_now - level_first
        rain_sum = sum(r for _, r in seq)

        if peak == 0:
            label = 0           # NoRisk
        elif peak < 10.0:
            label = 1           # Low
        else:
            label = 2           # High/Critical

        rows.append((level_now, delta, rain_sum, label))

    return pd.DataFrame(rows, columns=["level_now", "delta", "rain_sum", "label"])


def load_or_create_training_data() -> pd.DataFrame:
    """
    Load training_data.csv if it exists, otherwise create a synthetic dataset and save it.
    """
    if os.path.exists(TRAIN_CSV):
        log.info("Found training_data.csv. Loading existing dataset.")
        df = pd.read_csv(TRAIN_CSV)
        for col in ["level_now", "delta", "rain_sum", "label"]:
            if col not in df.columns:
                raise SystemExit(f"training_data.csv missing column: {col}")
        return df[["level_now", "delta", "rain_sum", "label"]].copy()

    log.info("No training_data.csv found — generating synthetic dataset (n=%d)", SYNTH_ROWS)
    df = gen_sequences(n=SYNTH_ROWS, seq_len=SEQ_LEN, flood_prob=FLOOD_PROB, seed=RANDOM_STATE)
    df.to_csv(TRAIN_CSV, index=False)
    log.info("Wrote synthetic training_data.csv with shape %s", df.shape)
    return df


# ----------------- TRAINING -----------------


def train_decision_tree(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> Tuple[DecisionTreeClassifier, dict]:
    """
    Train the baseline decision tree classifier and return model + metrics dict.
    """
    # Simple train/test split (time-agnostic). For real project, use time-based split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y,
    )

    clf = DecisionTreeClassifier(
        max_depth=5,  # interpretable
        min_samples_leaf=20,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # overall metrics
    report = classification_report(y_test, y_pred, digits=3)
    cm = confusion_matrix(y_test, y_pred)

    # focus on critical class (label == 2)
    crit_true = (y_test == 2).astype(int)
    crit_pred = (y_pred == 2).astype(int)
    p, r, f, _ = precision_recall_fscore_support(
        crit_true, crit_pred, average="binary", zero_division=0
    )

    metrics = {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "critical_precision": float(p),
        "critical_recall": float(r),
        "critical_f1": float(f),
    }

    return clf, metrics


def maybe_calibrate(
    clf: DecisionTreeClassifier,
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    """
    Best-effort probability calibration. If something fails, just log and move on.
    """
    if not HAVE_CAL:
        log.warning("sklearn.calibration.CalibratedClassifierCV not available; skipping calibration.")
        return

    try:
        log.info("Attempting probability calibration (CalibratedClassifierCV, isotonic)...")
        # smaller sample for speed
        # You can use full data, but this is enough for a mini-project
        X_cal, _, y_cal, _ = train_test_split(
            X, y, test_size=0.5, random_state=RANDOM_STATE, stratify=y
        )

        cal = CalibratedClassifierCV(
            estimator=clf,  # <--- correct param for new sklearn
            method="isotonic",
            cv=3,
        )
        cal.fit(X_cal, y_cal)
        joblib.dump(cal, CAL_MODEL_PATH)
        log.info("Saved calibrated model to %s", CAL_MODEL_PATH)
    except Exception as e:
        log.warning(
            "Calibration failed - falling back to uncalibrated DT only: %s", e
        )


# ----------------- PLOTTING -----------------


def plot_feature_importances(
    clf: DecisionTreeClassifier,
    feature_names: list[str],
    outpath: str,
) -> None:
    if not hasattr(clf, "feature_importances_"):
        log.warning("Classifier has no feature_importances_. Skipping plot.")
        return

    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(6, 4))
    plt.bar(range(len(importances)), importances[idx])
    plt.xticks(range(len(importances)), [feature_names[i] for i in idx], rotation=20)
    plt.ylabel("Importance")
    plt.title("Decision Tree Feature Importances")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    log.info("Saved feature importances plot to %s", outpath)


# ----------------- MAIN -----------------


def main() -> None:
    log.info("Training pipeline started.")

    df = load_or_create_training_data()
    log.info("Dataset shape: %s", df.shape)

    # ensure types
    df["level_now"] = df["level_now"].astype(float)
    df["delta"] = df["delta"].astype(float)
    df["rain_sum"] = df["rain_sum"].astype(float)
    df["label"] = df["label"].astype(int)

    X = df[["level_now", "delta", "rain_sum"]].to_numpy()
    y = df["label"].to_numpy()

    clf, metrics = train_decision_tree(X, y)
    joblib.dump(clf, BASELINE_MODEL_PATH)
    log.info("Saved baseline decision tree model to %s", BASELINE_MODEL_PATH)

    # Optional probability calibration
    maybe_calibrate(clf, X, y)

    # Save metrics text
    with open(METRICS_TXT, "w", encoding="utf-8") as f:
        f.write("=== FloodWatch Model Training Metrics ===\n\n")
        f.write("Dataset shape: {}\n".format(df.shape))
        f.write("Features: level_now, delta, rain_sum\n")
        f.write("Labels: 0=NoRisk, 1=Low, 2=High(Critical)\n\n")

        f.write("---- Classification report (all classes) ----\n")
        f.write(metrics["classification_report"])
        f.write("\n\n")

        f.write("---- Confusion matrix (rows: true, cols: pred) ----\n")
        f.write(str(metrics["confusion_matrix"]))
        f.write("\n\n")

        f.write("---- Critical class (label==2) ----\n")
        f.write(f"precision: {metrics['critical_precision']:.3f}\n")
        f.write(f"recall:    {metrics['critical_recall']:.3f}\n")
        f.write(f"f1-score:  {metrics['critical_f1']:.3f}\n")

    log.info("Wrote training metrics to %s", METRICS_TXT)

    # Plot feature importances
    feat_plot = os.path.join(PLOTS_DIR, "feature_importances.png")
    plot_feature_importances(
        clf,
        feature_names=["level_now", "delta", "rain_sum"],
        outpath=feat_plot,
    )

    log.info("Training pipeline finished.")
    print("-> Baseline model:", BASELINE_MODEL_PATH)
    if os.path.exists(CAL_MODEL_PATH):
        print("-> Calibrated model:", CAL_MODEL_PATH)
    print("-> Metrics:", METRICS_TXT)
    print("-> Feature importances plot:", feat_plot)


if __name__ == "__main__":
    main()
