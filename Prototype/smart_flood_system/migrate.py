#!/usr/bin/env python3
"""
migrate.py

One-time / occasional script to (re)create the SQLite schema for FloodWatch.

Even though app.py can auto-create tables on startup, this is useful when:
- You deleted data/floodwatch.db
- You want to be sure tables exist before running simulate_esp.py
- You want a clean DB before demos

Run:
    python migrate.py
"""

from __future__ import annotations
import os
import sqlite3
import logging
from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "floodwatch.db"

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("migrate")


def init_db() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    log.info("Using DB at: %s", DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()

        # Main sensor readings table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sensor (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id   TEXT    NOT NULL,
                level     REAL    NOT NULL,
                rain      REAL    NOT NULL,
                pre_alert INTEGER DEFAULT 0,
                ts        TEXT    NOT NULL
            );
            """
        )

        # Alerts table (ML/EWMA decisions)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT    NOT NULL,
                risk    TEXT    NOT NULL,
                prob    REAL,
                ts      TEXT    NOT NULL,
                reason  TEXT
            );
            """
        )

        # Helpful indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sensor_node_ts ON sensor(node_id, ts);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_alerts_node_ts ON alerts(node_id, ts);")

        conn.commit()
        log.info("DB schema ensured (sensor + alerts tables).")

    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    log.info("Done.")
