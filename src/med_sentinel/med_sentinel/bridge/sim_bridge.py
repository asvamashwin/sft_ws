"""Isaac Sim <-> WebSocket bridge for Med-Sentinel 360.

Runs alongside the simulation loop, sampling robot state at 100Hz,
serializing via Protobuf, and pushing to the FastAPI WebSocket server.
Also pulls incoming ControlCommands and applies them to the robot.

Usage:
    Called from the main simulation loop, not standalone.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

from med_sentinel.bridge import med_sentinel_pb2 as pb
from med_sentinel.bridge.proto_handler import build_robot_state
from med_sentinel.bridge.server import (
    bridge as bridge_state,
    get_latest_command,
    update_robot_state,
)

FRANKA_JOINT_NAMES = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7",
    "panda_finger_joint1", "panda_finger_joint2",
]


class SimBridge:
    """Bridges Isaac Sim robot state to the WebSocket server.

    Call `sample_and_push()` once per sim step from the simulation loop.
    Call `pull_command()` to get the latest ControlCommand.
    """

    def __init__(self, robot, world, config: Dict[str, Any]):
        self._robot = robot
        self._world = world
        self._config = config
        self._sequence = 0
        self._robot_name = config["robot"].get("type", "franka")
        self._joint_names = FRANKA_JOINT_NAMES
        self._last_sample_time = 0.0
        self._sample_interval = 1.0 / 100.0  # 100 Hz

    def sample_and_push(self) -> bool:
        """Sample robot state and push to the bridge server.

        Returns True if a sample was taken (rate-limited to 100Hz).
        """
        now = time.perf_counter()
        if (now - self._last_sample_time) < self._sample_interval:
            return False
        self._last_sample_time = now

        positions = self._robot.get_joint_positions()
        velocities = self._robot.get_joint_velocities()

        efforts = np.zeros_like(positions)

        status = pb.ROBOT_STATUS_IDLE
        vel_norm = np.linalg.norm(velocities)
        if vel_norm > 0.01:
            status = pb.ROBOT_STATUS_MOVING

        sim_time = self._world.current_time if hasattr(self._world, 'current_time') else 0.0

        state = build_robot_state(
            sequence=self._sequence,
            robot_name=self._robot_name,
            joint_names=self._joint_names,
            joint_positions=positions,
            joint_velocities=velocities,
            joint_efforts=efforts,
            status=status,
            sim_time=sim_time,
        )

        serialized = state.SerializeToString()
        update_robot_state(serialized)
        self._sequence += 1
        return True

    def pull_command(self) -> Optional[Dict]:
        """Get the latest control command from a connected web client."""
        return get_latest_command()

    def apply_command(self, controller) -> bool:
        """Pull command and apply it to the robot controller.

        Args:
            controller: FrankaController instance with set_joint_positions().

        Returns True if a command was applied.
        """
        cmd = self.pull_command()
        if cmd is None:
            return False

        if cmd.get("stop", False):
            current = self._robot.get_joint_positions()
            controller.set_joint_positions(current.tolist())
            return True

        targets = cmd.get("joint_targets", [])
        if not targets:
            return False

        mode = cmd.get("mode", pb.CONTROL_MODE_POSITION)
        if mode == pb.CONTROL_MODE_POSITION:
            controller.set_joint_positions(targets)

        return True


def start_server_thread(host: str = "0.0.0.0", port: int = 8765) -> threading.Thread:
    """Start the FastAPI WebSocket server in a background daemon thread."""
    from med_sentinel.bridge.server import run_server

    thread = threading.Thread(
        target=run_server,
        args=(host, port),
        daemon=True,
        name="med_sentinel_bridge",
    )
    thread.start()
    time.sleep(0.5)
    return thread
