"""Quaternion and pose utilities for Med-Sentinel 360.

All rotations use quaternions in (w, x, y, z) convention, consistent
with Isaac Sim's internal representation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass
class Pose:
    """6-DOF pose with position (x, y, z) and quaternion (w, x, y, z)."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))

    @classmethod
    def from_xyzq(cls, x: float, y: float, z: float,
                  qw: float = 1.0, qx: float = 0.0,
                  qy: float = 0.0, qz: float = 0.0) -> "Pose":
        return cls(
            position=np.array([x, y, z], dtype=np.float64),
            orientation=np.array([qw, qx, qy, qz], dtype=np.float64),
        )

    @property
    def pos_tuple(self) -> Tuple[float, float, float]:
        return tuple(self.position)

    @property
    def quat_tuple(self) -> Tuple[float, float, float, float]:
        return tuple(self.orientation)


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """Normalize a quaternion to unit length."""
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions in (w, x, y, z) format."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles (radians) to quaternion (w, x, y, z)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)

    return normalize_quaternion(np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]))


def euler_from_quaternion(q: np.ndarray) -> Tuple[float, float, float]:
    """Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw)."""
    w, x, y, z = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def random_yaw_quaternion(rng: random.Random | None = None) -> np.ndarray:
    """Generate a random rotation about the Z-axis (upright objects)."""
    r = rng or random
    yaw = r.uniform(0, 2 * math.pi)
    return quaternion_from_euler(0.0, 0.0, yaw)


def random_position_in_radius(
    center: np.ndarray,
    radius: float,
    z: float = 0.0,
    rng: random.Random | None = None,
) -> np.ndarray:
    """Sample a random (x, y, z) position within a circle of given radius."""
    r = rng or random
    angle = r.uniform(0, 2 * math.pi)
    dist = radius * math.sqrt(r.uniform(0, 1))
    return np.array([
        center[0] + dist * math.cos(angle),
        center[1] + dist * math.sin(angle),
        z,
    ])
