#!/usr/bin/env python3
"""
simulate_esp.py

Simulates ESP32 + ultrasonic level sensor + rain gauge sending data
to the FloodWatch backend at /ingest.

Usage examples:

    # single node, normal speed (runs until Ctrl+C)
    python simulate_esp.py

    # single node, faster "rapid" mode
    python simulate_esp.py --mode rapid

    # burst of exactly 50 samples (then stop)
    python simulate_esp.py --mode burst --count 50

    # send to custom URL as node2
    python simulate_esp.py --url http://127.0.0.1:5000/ingest --node node2

    # simulate three nodes together
    python simulate_esp.py --nodes node1,node2,node3 --mode rapid --count 300

    # stop automatically after 60 seconds
    python simulate_esp.py --max-seconds 60
"""

from __future__ import annotations
import argparse
import time
import random
from typing import Tuple, Dict, Any, List

import requests

DEFAULT_URL = "http://127.0.0.1:5000/ingest"


def gen_next_reading(
    level: float,
    rain: float,
    flood_state: Dict[str, Any],
    mode: str = "normal",
) -> Tuple[float, float, bool, Dict[str, Any]]:
    """
    Generate next (level, rain, pre_alert) based on a simple state machine.

    flood_state: carries whether we are in a "flood event" ramp.
    """

    # base noise level around ~20
    if level < 10.0:
        level = 20.0 + random.random() * 2.0

    # chance to start a "flood ramp" if not already in one
    if not flood_state.get("active", False):
        start_prob = 0.03 if mode == "rapid" else 0.01
        if random.random() < start_prob:
            flood_state = {
                "active": True,
                "steps_left": random.randint(10, 30),
                "rise_per_step": (
                    random.uniform(0.8, 1.6)
                    if mode == "rapid"
                    else random.uniform(0.5, 1.2)
                ),
            }

    # apply ramp if active
    if flood_state.get("active", False):
        level += flood_state["rise_per_step"] + random.uniform(-0.2, 0.3)
        rain = max(0.0, rain + random.uniform(0.5, 2.0))
        flood_state["steps_left"] -= 1
        if flood_state["steps_left"] <= 0:
            flood_state["active"] = False
    else:
        # calm behaviour
        level += random.uniform(-0.3, 0.3)
        # small random rain
        rain = max(0.0, rain + random.uniform(-0.1, 0.3))

    # clamp to sane range
    level = max(0.0, level)
    rain = max(0.0, min(rain, 50.0))

    # simple pre-alert flag
    pre_alert = level > 26.0 or (flood_state.get("active", False) and level > 24.0)

    return level, rain, pre_alert, flood_state


def simulate(
    url: str,
    node_ids: List[str],
    mode: str = "normal",
    count: int | None = None,
    base_sleep: float = 2.0,
    max_seconds: float | None = None,
) -> None:
    """
    Core simulation loop.

    - node_ids: list of node IDs to simulate (1..N).
    - count: total number of samples across ALL nodes. None = infinite.
    - max_seconds: stop after this many seconds. None = no time limit.
    """
    print(f"[SIM] Sending data to {url}")
    print(f"[SIM] Nodes: {', '.join(node_ids)}  |  mode={mode}")
    if count is not None:
        print(f"[SIM] Max samples (total across nodes): {count}")
    if max_seconds is not None:
        print(f"[SIM] Max runtime: {max_seconds} seconds")

    if mode == "rapid":
        sleep_time = 0.5
    elif mode == "burst":
        sleep_time = 0.2
    else:
        sleep_time = base_sleep

    # state per node
    node_state: Dict[str, Dict[str, Any]] = {}
    for nid in node_ids:
        node_state[nid] = {
            "level": 20.0 + random.uniform(-1.0, 1.0),
            "rain": 0.0,
            "flood_state": {"active": False},
        }

    total_sent = 0
    start_time = time.time()

    try:
        while True:
            for nid in node_ids:
                st = node_state[nid]
                level, rain, pre_alert, fstate = gen_next_reading(
                    st["level"], st["rain"], st["flood_state"], mode=mode
                )
                st["level"], st["rain"], st["flood_state"] = level, rain, fstate

                payload = {
                    "node_id": nid,
                    "level": round(level, 2),
                    "rain": round(rain, 2),
                    "pre_alert": bool(pre_alert),
                }

                total_sent += 1

                try:
                    resp = requests.post(url, json=payload, timeout=5)
                    status = resp.status_code
                    try:
                        data = resp.json()
                    except Exception:
                        data = resp.text
                    print(f"[{total_sent:04d}] {nid} -> {payload}  Resp: {status}  -> {data}")
                except Exception as e:
                    print(f"[{total_sent:04d}] {nid} ERROR sending: {e}")

                # stop on count
                if count is not None and total_sent >= count:
                    print(f"[SIM] Reached count={count}. Stopping.")
                    return

                # stop on time limit
                if max_seconds is not None and (time.time() - start_time) >= max_seconds:
                    print(f"[SIM] Reached max_seconds={max_seconds}. Stopping.")
                    return

            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[SIM] Stopped by user (Ctrl+C). Total samples sent:", total_sent)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate ESP32 flood sensors sending to /ingest"
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Ingest URL (default: %(default)s)",
    )
    parser.add_argument(
        "--node",
        default="node1",
        help="Single node ID (default: %(default)s). Ignored if --nodes is used.",
    )
    parser.add_argument(
        "--nodes",
        default=None,
        help="Comma-separated list of node IDs (e.g. node1,node2,node3). "
             "If set, overrides --node.",
    )
    parser.add_argument(
        "--mode",
        default="normal",
        choices=["normal", "rapid", "burst"],
        help="Simulation mode (default: %(default)s)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Total number of samples to send across all nodes (default: infinite)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=2.0,
        help="Base sleep seconds for normal mode (default: %(default)s)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Maximum runtime in seconds (default: no limit)",
    )

    args = parser.parse_args()

    if args.nodes:
        node_ids = [s.strip() for s in args.nodes.split(",") if s.strip()]
    else:
        node_ids = [args.node]

    simulate(
        url=args.url,
        node_ids=node_ids,
        mode=args.mode,
        count=args.count,
        base_sleep=args.sleep,
        max_seconds=args.max_seconds,
    )


if __name__ == "__main__":
    main()

#python simulate_esp.py --nodes node1,node2,node3 --mode rapid --count 300
## python simulate_esp.py --mode rapid --max-seconds 60


