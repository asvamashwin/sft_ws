"""Protobuf serialization helpers for Med-Sentinel 360.

Converts between native Python/numpy types and protobuf messages
for zero-copy-friendly wire transport over WebSockets.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from med_sentinel.bridge import med_sentinel_pb2 as pb


def make_timestamp() -> pb.Timestamp:
    """Create a Timestamp message from the current wall clock."""
    now = time.time()
    ts = pb.Timestamp()
    ts.seconds = int(now)
    ts.nanos = int((now - int(now)) * 1e9)
    return ts


def timestamp_to_float(ts: pb.Timestamp) -> float:
    """Convert a Timestamp message to a float (seconds since epoch)."""
    return ts.seconds + ts.nanos * 1e-9


def build_robot_state(
    sequence: int,
    robot_name: str,
    joint_names: List[str],
    joint_positions: np.ndarray,
    joint_velocities: np.ndarray,
    joint_efforts: np.ndarray,
    ee_position: Optional[np.ndarray] = None,
    ee_orientation: Optional[np.ndarray] = None,
    base_position: Optional[np.ndarray] = None,
    base_orientation: Optional[np.ndarray] = None,
    status: int = pb.ROBOT_STATUS_IDLE,
    sim_time: float = 0.0,
) -> pb.RobotState:
    """Build a RobotState protobuf from numpy arrays."""
    state = pb.RobotState()
    state.stamp.CopyFrom(make_timestamp())
    state.sequence = sequence
    state.robot_name = robot_name
    state.status = status
    state.sim_time = sim_time

    js = state.joint_state
    js.name.extend(joint_names)
    js.position.extend(joint_positions.tolist())
    js.velocity.extend(joint_velocities.tolist())
    js.effort.extend(joint_efforts.tolist())

    if ee_position is not None and ee_orientation is not None:
        ee = state.end_effector
        ee.pose.position.x = float(ee_position[0])
        ee.pose.position.y = float(ee_position[1])
        ee.pose.position.z = float(ee_position[2])
        ee.pose.orientation.w = float(ee_orientation[0])
        ee.pose.orientation.x = float(ee_orientation[1])
        ee.pose.orientation.y = float(ee_orientation[2])
        ee.pose.orientation.z = float(ee_orientation[3])

    if base_position is not None and base_orientation is not None:
        bp = state.base_pose
        bp.position.x = float(base_position[0])
        bp.position.y = float(base_position[1])
        bp.position.z = float(base_position[2])
        bp.orientation.w = float(base_orientation[0])
        bp.orientation.x = float(base_orientation[1])
        bp.orientation.y = float(base_orientation[2])
        bp.orientation.z = float(base_orientation[3])

    return state


def parse_control_command(data: bytes) -> Dict:
    """Deserialize a ControlCommand from wire bytes into a plain dict."""
    cmd = pb.ControlCommand()
    cmd.ParseFromString(data)
    return {
        "stamp": timestamp_to_float(cmd.stamp),
        "sequence": cmd.sequence,
        "mode": cmd.mode,
        "joint_targets": list(cmd.joint_targets),
        "max_velocity": cmd.max_velocity,
        "max_accel": cmd.max_accel,
        "stop": cmd.stop,
    }


def build_control_command(
    sequence: int,
    joint_targets: List[float],
    mode: int = pb.CONTROL_MODE_POSITION,
    max_velocity: float = 0.5,
    max_accel: float = 0.5,
    stop: bool = False,
) -> pb.ControlCommand:
    """Build a ControlCommand protobuf message."""
    cmd = pb.ControlCommand()
    cmd.stamp.CopyFrom(make_timestamp())
    cmd.sequence = sequence
    cmd.mode = mode
    cmd.joint_targets.extend(joint_targets)
    cmd.max_velocity = max_velocity
    cmd.max_accel = max_accel
    cmd.stop = stop
    return cmd


def build_ping(sequence: int) -> pb.PingPong:
    """Build a PingPong message for latency measurement."""
    ping = pb.PingPong()
    ping.client_send.CopyFrom(make_timestamp())
    ping.sequence = sequence
    return ping


def stamp_ping_server(ping_bytes: bytes) -> pb.PingPong:
    """Add server timestamps to a received PingPong message."""
    ping = pb.PingPong()
    ping.ParseFromString(ping_bytes)
    ping.server_recv.CopyFrom(make_timestamp())
    ping.server_send.CopyFrom(make_timestamp())
    return ping
