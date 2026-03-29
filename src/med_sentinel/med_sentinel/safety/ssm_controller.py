"""ISO 15066 Speed and Separation Monitoring (SSM) controller.

Implements the protective separation distance formula:

    S_p = S_h + S_r + S_s + C + Z_d + Z_r

and classifies the current state into safety zones:

    GREEN  -- d > S_p + margin:  full speed
    YELLOW -- S_p < d <= S_p + margin:  scaled speed  V_max = (d - S_p) / T_r
    RED    -- d <= S_p:  protective stop (V_max = 0)

The controller outputs a velocity_scale in [0.0, 1.0] that the robot
controller applies as a multiplier to its maximum joint velocity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np

from med_sentinel.robot_model import PandaModel
from med_sentinel.safety.distance_monitor import DistanceMonitor, DistanceResult
from med_sentinel.safety.human_tracker import HumanTracker


class SafetyZone(Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


@dataclass
class SSMState:
    """Full SSM state snapshot for logging and telemetry."""
    d_min: float = float("inf")
    S_p: float = 0.0
    S_h: float = 0.0
    S_r: float = 0.0
    S_s: float = 0.0
    velocity_scale: float = 1.0
    zone: SafetyZone = SafetyZone.GREEN
    closest_link: str = ""
    closest_human_joint: str = ""
    v_robot: float = 0.0
    v_human: float = 0.0
    protective_stop: bool = False


class SSMController:
    """ISO 15066 compliant Speed and Separation Monitoring.

    Parameters are loaded from the ``safety.ssm`` section of the config.
    The controller is called once per control cycle with the current joint
    state and outputs a velocity scaling factor.
    """

    def __init__(
        self,
        robot_model: PandaModel,
        distance_monitor: DistanceMonitor,
        human_tracker: HumanTracker,
        config: Dict[str, Any],
    ):
        self._robot_model = robot_model
        self._distance_monitor = distance_monitor
        self._human_tracker = human_tracker

        ssm_cfg = config.get("safety", {}).get("ssm", {})
        self._T_r = ssm_cfg.get("T_reaction", 0.1)
        self._T_s = ssm_cfg.get("T_stopping", 0.2)
        self._C = ssm_cfg.get("C_intrusion", 0.05)
        self._Z_d = ssm_cfg.get("Z_d", 0.04)
        self._Z_r = ssm_cfg.get("Z_r", 0.02)
        self._v_human_max = ssm_cfg.get("v_human_max", 1.6)
        self._margin = ssm_cfg.get("margin", 0.3)

        self._state = SSMState()

    @property
    def state(self) -> SSMState:
        return self._state

    def update(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        dt: float = 0.01,
    ) -> SSMState:
        """Run one SSM cycle.

        Args:
            q: Current joint positions.
            dq: Current joint velocities.
            dt: Control cycle period (seconds).

        Returns:
            Updated SSMState with zone, velocity_scale, etc.
        """
        dist_result = self._distance_monitor.compute(q)

        v_human = self._compute_human_speed(dist_result.closest_human_joint)
        v_robot = self._compute_robot_link_speed(q, dq, dist_result.closest_link)

        S_h = self._compute_S_h(v_human)
        S_r = self._compute_S_r(v_robot)
        S_s = self._compute_S_s(q, dq, v_robot)
        S_p = S_h + S_r + S_s + self._C + self._Z_d + self._Z_r

        zone, velocity_scale = self._classify_zone(dist_result.d_min, S_p)

        self._state = SSMState(
            d_min=dist_result.d_min,
            S_p=S_p,
            S_h=S_h,
            S_r=S_r,
            S_s=S_s,
            velocity_scale=velocity_scale,
            zone=zone,
            closest_link=dist_result.closest_link,
            closest_human_joint=dist_result.closest_human_joint,
            v_robot=v_robot,
            v_human=v_human,
            protective_stop=(zone == SafetyZone.RED),
        )
        return self._state

    def _compute_S_h(self, v_human: float) -> float:
        """Human contribution: distance human can cover during reaction + stopping."""
        v = max(v_human, 0.0)
        return v * (self._T_r + self._T_s)

    def _compute_S_r(self, v_robot: float) -> float:
        """Robot reaction distance: distance robot travels during reaction time."""
        return abs(v_robot) * self._T_r

    def _compute_S_s(
        self, q: np.ndarray, dq: np.ndarray, v_robot: float
    ) -> float:
        """Robot stopping distance from current speed.

        Uses v^2 / (2 * a_max) where a_max is derived from the robot's
        effort limits and mass matrix diagonal (approximate worst-case).
        """
        v = abs(v_robot)
        if v < 1e-6:
            return 0.0

        try:
            M = self._robot_model.mass_matrix(q)
            effort_lim = self._robot_model.effort_limits
            M_diag = np.diag(M)
            valid = M_diag > 1e-6
            a_max_per_joint = np.where(
                valid, effort_lim[:len(M_diag)] / M_diag, 0.0
            )
            a_max = float(np.max(a_max_per_joint)) if np.any(a_max_per_joint > 0) else 10.0
        except Exception:
            a_max = 10.0

        return (v ** 2) / (2.0 * a_max)

    def _classify_zone(
        self, d_current: float, S_p: float
    ) -> tuple[SafetyZone, float]:
        """Classify the current state and compute velocity scaling."""
        if d_current <= S_p:
            return SafetyZone.RED, 0.0

        if d_current <= S_p + self._margin:
            scale = (d_current - S_p) / max(self._margin, 1e-6)
            scale = np.clip(scale, 0.0, 1.0)
            return SafetyZone.YELLOW, float(scale)

        return SafetyZone.GREEN, 1.0

    def _compute_human_speed(self, joint_name: str) -> float:
        """Get human speed at the closest joint, capped by v_human_max."""
        speed = self._human_tracker.get_closest_joint_speed(joint_name)
        return min(speed, self._v_human_max)

    def _compute_robot_link_speed(
        self, q: np.ndarray, dq: np.ndarray, link_name: str
    ) -> float:
        """Compute Cartesian speed of the closest robot link.

        Uses J(q) * dq to get the linear velocity at the link.
        """
        frame_mapping = {
            "panda_link0": "panda_joint1",
            "panda_link1": "panda_joint1",
            "panda_link2": "panda_joint2",
            "panda_link3": "panda_joint3",
            "panda_link4": "panda_joint4",
            "panda_link5": "panda_joint5",
            "panda_link6": "panda_joint6",
            "panda_link7": "panda_joint7",
            "panda_hand":  "panda_hand_joint",
        }
        frame_name = frame_mapping.get(link_name, link_name)

        try:
            J = self._robot_model.jacobian(q, frame=frame_name)
            twist = J @ dq[:J.shape[1]]
            linear_vel = twist[:3]
            return float(np.linalg.norm(linear_vel))
        except Exception:
            return 0.0

    def get_max_allowed_velocity(self) -> float:
        """Convenience: return velocity_scale from the most recent update."""
        return self._state.velocity_scale
