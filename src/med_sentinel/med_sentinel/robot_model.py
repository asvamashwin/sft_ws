"""Pinocchio-based robot model for Med-Sentinel 360.

Loads the Franka Panda URDF and provides:
  - Forward kinematics (FK)
  - Inverse kinematics (IK) via CLIK
  - Jacobian computation
  - Joint limit queries
  - Dynamics (mass matrix, Coriolis, gravity)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pinocchio as pin

from med_sentinel.utils.transforms import Pose, rotation_to_quat_wxyz


def _default_urdf_path() -> str:
    """Resolve the vendored Panda URDF path."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "description", "urdf", "panda.urdf",
    )


def _default_srdf_path() -> str:
    """Resolve the vendored Panda SRDF path."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "description", "srdf", "panda.srdf",
    )


def _package_dirs() -> List[str]:
    """Mesh package lookup directories."""
    return [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."),
    ]


class PandaModel:
    """Pinocchio model of the Franka Panda loaded from URDF.

    Provides FK, IK, Jacobians, and dynamics computations.
    """

    EE_FRAME_NAME = "panda_hand_tcp"
    ARM_JOINT_NAMES = [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7",
    ]
    FINGER_JOINT_NAMES = [
        "panda_finger_joint1", "panda_finger_joint2",
    ]

    def __init__(
        self,
        urdf_path: Optional[str] = None,
        srdf_path: Optional[str] = None,
    ):
        urdf_path = urdf_path or _default_urdf_path()
        srdf_path = srdf_path or _default_srdf_path()
        pkg_dirs = _package_dirs()

        self._model = pin.buildModelFromUrdf(urdf_path)
        self._data = self._model.createData()

        self._collision_model = pin.buildGeomFromUrdf(
            self._model, urdf_path, pin.GeometryType.COLLISION, package_dirs=pkg_dirs,
        )
        self._visual_model = pin.buildGeomFromUrdf(
            self._model, urdf_path, pin.GeometryType.VISUAL, package_dirs=pkg_dirs,
        )
        self._collision_data = pin.GeometryData(self._collision_model)

        if os.path.isfile(srdf_path):
            pin.loadReferenceConfigurations(self._model, srdf_path, False)
            self._srdf_loaded = True
        else:
            self._srdf_loaded = False

        self._ee_frame_id = self._model.getFrameId(self.EE_FRAME_NAME)
        self._arm_joint_ids = [
            self._model.getJointId(name) for name in self.ARM_JOINT_NAMES
        ]

    @property
    def model(self) -> pin.Model:
        return self._model

    @property
    def data(self) -> pin.Data:
        return self._data

    @property
    def nq(self) -> int:
        """Configuration space dimension."""
        return self._model.nq

    @property
    def nv(self) -> int:
        """Velocity space dimension."""
        return self._model.nv

    @property
    def joint_names(self) -> List[str]:
        return list(self._model.names)[1:]  # skip 'universe'

    @property
    def lower_limits(self) -> np.ndarray:
        return self._model.lowerPositionLimit.copy()

    @property
    def upper_limits(self) -> np.ndarray:
        return self._model.upperPositionLimit.copy()

    @property
    def velocity_limits(self) -> np.ndarray:
        return self._model.velocityLimit.copy()

    @property
    def effort_limits(self) -> np.ndarray:
        return self._model.effortLimit.copy()

    def default_configuration(self) -> np.ndarray:
        """Return the SRDF 'default' configuration, or neutral if unavailable."""
        if self._srdf_loaded and "default" in self._model.referenceConfigurations:
            return self._model.referenceConfigurations["default"].copy()
        return pin.neutral(self._model)

    # ------------------------------------------------------------------
    # Forward Kinematics
    # ------------------------------------------------------------------

    def forward_kinematics(self, q: np.ndarray) -> Pose:
        """Compute end-effector pose for joint configuration q."""
        pin.forwardKinematics(self._model, self._data, q)
        pin.updateFramePlacements(self._model, self._data)
        se3 = self._data.oMf[self._ee_frame_id]
        return Pose.from_se3(pin.SE3(se3))

    def frame_placement(self, q: np.ndarray, frame_name: str) -> Pose:
        """Compute the pose of any named frame."""
        frame_id = self._model.getFrameId(frame_name)
        pin.forwardKinematics(self._model, self._data, q)
        pin.updateFramePlacements(self._model, self._data)
        se3 = self._data.oMf[frame_id]
        return Pose.from_se3(pin.SE3(se3))

    def all_link_poses(self, q: np.ndarray) -> Dict[str, Pose]:
        """Compute poses of all links (joint frames)."""
        pin.forwardKinematics(self._model, self._data, q)
        poses = {}
        for i in range(1, self._model.njoints):
            name = self._model.names[i]
            poses[name] = Pose.from_se3(pin.SE3(self._data.oMi[i]))
        return poses

    # ------------------------------------------------------------------
    # Jacobians
    # ------------------------------------------------------------------

    def jacobian(self, q: np.ndarray, frame: str = "ee") -> np.ndarray:
        """Compute the 6xN frame Jacobian (local world-aligned).

        Args:
            q: Joint configuration.
            frame: "ee" for end-effector, or a specific frame name.

        Returns:
            6xN Jacobian matrix [linear_vel; angular_vel].
        """
        if frame == "ee":
            frame_id = self._ee_frame_id
        else:
            frame_id = self._model.getFrameId(frame)

        pin.computeJointJacobians(self._model, self._data, q)
        pin.updateFramePlacements(self._model, self._data)
        return pin.getFrameJacobian(
            self._model, self._data, frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

    # ------------------------------------------------------------------
    # Inverse Kinematics (CLIK -- Closed-Loop IK)
    # ------------------------------------------------------------------

    def inverse_kinematics(
        self,
        target_pose: Pose,
        q_init: Optional[np.ndarray] = None,
        max_iters: int = 1000,
        dt: float = 0.1,
        tol: float = 1e-4,
        damp: float = 1e-12,
    ) -> Tuple[np.ndarray, bool]:
        """Solve IK for a target end-effector pose using damped least-squares.

        Args:
            target_pose: Desired end-effector Pose.
            q_init: Initial guess (defaults to default config).
            max_iters: Maximum CLIK iterations.
            dt: Step size.
            tol: Convergence tolerance on SE3 error norm.
            damp: Damping factor for pseudo-inverse.

        Returns:
            (q_solution, converged) tuple.
        """
        oMdes = target_pose.se3
        q = q_init.copy() if q_init is not None else self.default_configuration()

        for _ in range(max_iters):
            pin.forwardKinematics(self._model, self._data, q)
            pin.updateFramePlacements(self._model, self._data)

            oMcurrent = self._data.oMf[self._ee_frame_id]
            error = pin.log6(oMcurrent.inverse() * oMdes).vector

            if np.linalg.norm(error) < tol:
                return self._clamp_to_limits(q), True

            J = pin.computeFrameJacobian(
                self._model, self._data, q,
                self._ee_frame_id, pin.ReferenceFrame.LOCAL,
            )
            JtJ = J.T @ J + damp * np.eye(self.nv)
            dq = np.linalg.solve(JtJ, J.T @ error)
            q = pin.integrate(self._model, q, dq * dt)

        return self._clamp_to_limits(q), False

    def _clamp_to_limits(self, q: np.ndarray) -> np.ndarray:
        """Clamp joint values to within model limits."""
        return np.clip(q, self._model.lowerPositionLimit, self._model.upperPositionLimit)

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """Joint-space mass (inertia) matrix M(q)."""
        return pin.crba(self._model, self._data, q).copy()

    def coriolis_matrix(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """Coriolis matrix C(q, dq)."""
        pin.computeCoriolisMatrix(self._model, self._data, q, dq)
        return self._data.C.copy()

    def gravity_torques(self, q: np.ndarray) -> np.ndarray:
        """Gravity compensation torques g(q)."""
        return pin.computeGeneralizedGravity(self._model, self._data, q).copy()

    def inverse_dynamics(
        self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray
    ) -> np.ndarray:
        """Inverse dynamics via RNEA: tau = M*ddq + C*dq + g."""
        return pin.rnea(self._model, self._data, q, dq, ddq).copy()

    # ------------------------------------------------------------------
    # Collision checking
    # ------------------------------------------------------------------

    def check_self_collision(self, q: np.ndarray) -> bool:
        """Return True if any self-collision is detected."""
        pin.forwardKinematics(self._model, self._data, q)
        pin.updateGeometryPlacements(
            self._model, self._data, self._collision_model, self._collision_data, q,
        )
        return pin.computeCollisions(
            self._model, self._data,
            self._collision_model, self._collision_data,
            q, True,
        )

    def is_within_limits(self, q: np.ndarray) -> bool:
        """Check if all joints are within their position limits."""
        lower = self._model.lowerPositionLimit
        upper = self._model.upperPositionLimit
        return bool(np.all(q >= lower) and np.all(q <= upper))
