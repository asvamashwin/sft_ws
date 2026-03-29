"""Safety event logger for Med-Sentinel 360.

Writes per-cycle safety data to:
  - CSV files (for offline analysis)
  - Protobuf SafetyState messages (for real-time telemetry via bridge)

CSV columns:
  timestamp, d_min, S_p, velocity_scale, zone, v_robot, v_human,
  closest_link, closest_human, max_contact_force, protective_stop
"""

from __future__ import annotations

import csv
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from med_sentinel.safety.ssm_controller import SSMState, SafetyZone
from med_sentinel.safety.pfl_monitor import PFLState

CSV_HEADER = [
    "timestamp",
    "d_min",
    "S_p",
    "S_h",
    "S_r",
    "S_s",
    "velocity_scale",
    "zone",
    "v_robot",
    "v_human",
    "closest_link",
    "closest_human_joint",
    "max_contact_force",
    "max_force_link",
    "protective_stop",
]


class SafetyLogger:
    """Logs SSM and PFL data to CSV and optionally to the Protobuf bridge."""

    def __init__(self, config: Dict[str, Any], session_name: str = ""):
        log_cfg = config.get("safety", {}).get("logging", {})
        self._output_dir = log_cfg.get("output_dir", "safety_logs")
        self._csv_enabled = log_cfg.get("csv_enabled", True)
        self._proto_enabled = log_cfg.get("proto_enabled", True)

        if not session_name:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_name = session_name

        self._csv_writer: Optional[csv.writer] = None
        self._csv_file = None
        self._csv_path: Optional[str] = None
        self._record_count = 0
        self._records: List[Dict[str, Any]] = []
        self._bridge_callback = None

    def start(self):
        """Open the CSV file and write the header."""
        if not self._csv_enabled:
            return

        os.makedirs(self._output_dir, exist_ok=True)
        self._csv_path = os.path.join(
            self._output_dir, f"safety_{self._session_name}.csv"
        )
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(CSV_HEADER)
        print(f"[SafetyLogger] Logging to {self._csv_path}")

    def stop(self):
        """Close the CSV file and print summary."""
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
            print(
                f"[SafetyLogger] Closed log: {self._csv_path} "
                f"({self._record_count} records)"
            )

    def set_bridge_callback(self, callback):
        """Register a callback to push SafetyState protos to the bridge.

        The callback should accept a serialized SafetyState bytes object.
        """
        self._bridge_callback = callback

    def log(self, ssm_state: SSMState, pfl_state: PFLState):
        """Log one cycle of safety data."""
        now = time.time()

        record = {
            "timestamp": now,
            "d_min": ssm_state.d_min,
            "S_p": ssm_state.S_p,
            "S_h": ssm_state.S_h,
            "S_r": ssm_state.S_r,
            "S_s": ssm_state.S_s,
            "velocity_scale": ssm_state.velocity_scale,
            "zone": ssm_state.zone.value,
            "v_robot": ssm_state.v_robot,
            "v_human": ssm_state.v_human,
            "closest_link": ssm_state.closest_link,
            "closest_human_joint": ssm_state.closest_human_joint,
            "max_contact_force": pfl_state.max_force,
            "max_force_link": pfl_state.max_force_link,
            "protective_stop": ssm_state.protective_stop or pfl_state.protective_stop,
        }

        self._records.append(record)
        self._record_count += 1

        if self._csv_writer:
            self._csv_writer.writerow([record[col] for col in CSV_HEADER])
            if self._record_count % 100 == 0:
                self._csv_file.flush()

        if self._proto_enabled and self._bridge_callback:
            self._push_proto(ssm_state, pfl_state)

    def _push_proto(self, ssm_state: SSMState, pfl_state: PFLState):
        """Serialize safety state to protobuf and push via bridge callback."""
        try:
            from med_sentinel.bridge import med_sentinel_pb2 as pb

            msg = pb.SafetyState()
            now = time.time()
            msg.stamp.seconds = int(now)
            msg.stamp.nanos = int((now - int(now)) * 1e9)
            msg.sequence = self._record_count
            msg.d_min = ssm_state.d_min
            msg.S_p = ssm_state.S_p
            msg.velocity_scale = ssm_state.velocity_scale
            msg.closest_link = ssm_state.closest_link
            msg.closest_human = ssm_state.closest_human_joint
            msg.max_contact_force = pfl_state.max_force
            msg.protective_stop = (
                ssm_state.protective_stop or pfl_state.protective_stop
            )
            msg.v_robot = ssm_state.v_robot
            msg.v_human = ssm_state.v_human

            zone_map = {
                SafetyZone.GREEN: pb.ZONE_GREEN,
                SafetyZone.YELLOW: pb.ZONE_YELLOW,
                SafetyZone.RED: pb.ZONE_RED,
            }
            msg.zone = zone_map.get(ssm_state.zone, pb.ZONE_GREEN)

            self._bridge_callback(msg.SerializeToString())
        except Exception as e:
            print(f"[SafetyLogger] Proto push error: {e}")

    # ------------------------------------------------------------------
    # Post-session analysis
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Compute summary statistics from logged records."""
        if not self._records:
            return {"record_count": 0}

        d_mins = [r["d_min"] for r in self._records]
        forces = [r["max_contact_force"] for r in self._records]
        scales = [r["velocity_scale"] for r in self._records]
        stops = [r["protective_stop"] for r in self._records]

        zone_counts = {"GREEN": 0, "YELLOW": 0, "RED": 0}
        for r in self._records:
            zone_counts[r["zone"]] = zone_counts.get(r["zone"], 0) + 1

        stop_times = self._compute_stop_times()

        return {
            "record_count": len(self._records),
            "d_min_min": float(np.min(d_mins)),
            "d_min_mean": float(np.mean(d_mins)),
            "max_contact_force": float(np.max(forces)),
            "mean_velocity_scale": float(np.mean(scales)),
            "total_protective_stops": sum(stops),
            "zone_distribution": zone_counts,
            "stop_times_ms": stop_times,
            "avg_stop_time_ms": float(np.mean(stop_times)) if stop_times else 0.0,
        }

    def _compute_stop_times(self) -> List[float]:
        """Find stop-time durations: time from zone entering RED to velocity=0."""
        stop_times = []
        in_stop = False
        stop_start = 0.0

        for r in self._records:
            if r["protective_stop"] and not in_stop:
                in_stop = True
                stop_start = r["timestamp"]
            elif not r["protective_stop"] and in_stop:
                in_stop = False

            if in_stop and r["velocity_scale"] == 0.0 and stop_start > 0:
                dt_ms = (r["timestamp"] - stop_start) * 1000.0
                stop_times.append(dt_ms)
                stop_start = r["timestamp"]

        return stop_times

    @property
    def records(self) -> List[Dict[str, Any]]:
        return list(self._records)

    @property
    def record_count(self) -> int:
        return self._record_count
