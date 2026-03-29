"""Spatial math utilities for Med-Sentinel 360, backed by Pinocchio.

Wraps pinocchio.SE3 and pinocchio.Quaternion for rigid body transforms.
All quaternions follow the (w, x, y, z) convention used by Isaac Sim.
Pinocchio internally stores coeffs as (x, y, z, w) but its constructor
takes (w, x, y, z) -- we handle the conversion transparently.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pinocchio as pin


# ---------------------------------------------------------------------------
# Pose dataclass wrapping pin.SE3
# ---------------------------------------------------------------------------

@dataclass
class Pose:
    """6-DOF pose backed by a Pinocchio SE3 transform.

    External interface uses (w, x, y, z) quaternion convention.
    """

    _se3: pin.SE3 = field(default_factory=lambda: pin.SE3.Identity())

    @classmethod
    def from_xyzq(cls, x: float, y: float, z: float,
                  qw: float = 1.0, qx: float = 0.0,
                  qy: float = 0.0, qz: float = 0.0) -> "Pose":
        """Construct from position + quaternion (w, x, y, z)."""
        quat = pin.Quaternion(qw, qx, qy, qz)
        quat.normalize()
        se3 = pin.SE3(quat.toRotationMatrix(), np.array([x, y, z]))
        return cls(_se3=se3)

    @classmethod
    def from_se3(cls, se3: pin.SE3) -> "Pose":
        return cls(_se3=se3)

    @classmethod
    def from_rotation_translation(cls, R: np.ndarray, t: np.ndarray) -> "Pose":
        return cls(_se3=pin.SE3(R, t))

    @classmethod
    def identity(cls) -> "Pose":
        return cls(_se3=pin.SE3.Identity())

    @property
    def se3(self) -> pin.SE3:
        return self._se3

    @property
    def position(self) -> np.ndarray:
        return self._se3.translation.copy()

    @property
    def rotation(self) -> np.ndarray:
        return self._se3.rotation.copy()

    @property
    def orientation(self) -> np.ndarray:
        """Quaternion as (w, x, y, z)."""
        return rotation_to_quat_wxyz(self._se3.rotation)

    @property
    def pos_tuple(self) -> Tuple[float, float, float]:
        t = self._se3.translation
        return (float(t[0]), float(t[1]), float(t[2]))

    @property
    def quat_tuple(self) -> Tuple[float, float, float, float]:
        q = self.orientation
        return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    def compose(self, other: "Pose") -> "Pose":
        """Compose two transforms: self * other."""
        return Pose(_se3=self._se3 * other._se3)

    def inverse(self) -> "Pose":
        return Pose(_se3=self._se3.inverse())

    def act_on_point(self, point: np.ndarray) -> np.ndarray:
        """Transform a 3D point from local frame to this frame."""
        return self._se3.act(point)


# ---------------------------------------------------------------------------
# Quaternion helpers  (w, x, y, z) <-> Pinocchio/Eigen (x, y, z, w)
# ---------------------------------------------------------------------------

def quat_wxyz_to_pin(q: np.ndarray) -> pin.Quaternion:
    """Convert (w, x, y, z) array to a Pinocchio Quaternion."""
    pq = pin.Quaternion(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    pq.normalize()
    return pq


def quat_pin_to_wxyz(pq: pin.Quaternion) -> np.ndarray:
    """Convert a Pinocchio Quaternion to (w, x, y, z) array."""
    return np.array([pq.w, pq.x, pq.y, pq.z])


def rotation_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to quaternion (w, x, y, z)."""
    pq = pin.Quaternion(R)
    pq.normalize()
    return quat_pin_to_wxyz(pq)


def quat_wxyz_to_rotation(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    return quat_wxyz_to_pin(q).toRotationMatrix()


# ---------------------------------------------------------------------------
# Quaternion arithmetic
# ---------------------------------------------------------------------------

def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions in (w, x, y, z) format."""
    pq1 = quat_wxyz_to_pin(q1)
    pq2 = quat_wxyz_to_pin(q2)
    result = pq1 * pq2
    return quat_pin_to_wxyz(result)


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """Inverse of a unit quaternion (w, x, y, z)."""
    pq = quat_wxyz_to_pin(q)
    return quat_pin_to_wxyz(pq.conjugate())


def quaternion_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions."""
    pq1 = quat_wxyz_to_pin(q1)
    pq2 = quat_wxyz_to_pin(q2)
    result = pq1.slerp(t, pq2)
    return quat_pin_to_wxyz(result)


# ---------------------------------------------------------------------------
# Euler <-> Quaternion conversions via Pinocchio RPY
# ---------------------------------------------------------------------------

def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles (radians, RPY) to quaternion (w, x, y, z)."""
    R = pin.rpy.rpyToMatrix(roll, pitch, yaw)
    return rotation_to_quat_wxyz(R)


def euler_from_quaternion(q: np.ndarray) -> Tuple[float, float, float]:
    """Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw)."""
    R = quat_wxyz_to_rotation(q)
    rpy = pin.rpy.matrixToRpy(R)
    return (float(rpy[0]), float(rpy[1]), float(rpy[2]))


# ---------------------------------------------------------------------------
# Random sampling
# ---------------------------------------------------------------------------

def random_yaw_quaternion(rng: random.Random | None = None) -> np.ndarray:
    """Random rotation about the Z-axis (keeps objects upright)."""
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
