"""Human avatar tracking for Med-Sentinel 360.

Spawns a human character on the Isaac Sim stage and tracks skeletal
joint positions each simulation step.  Computes per-joint velocities
via finite differencing for use in the SSM controller.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from med_sentinel.utils.transforms import Pose

TRACKED_JOINTS = [
    "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot",
    "RightUpLeg", "RightLeg", "RightFoot",
]

ISO_BODY_REGION_MAP: Dict[str, str] = {
    "Head": "head", "Neck": "head",
    "Spine": "chest", "Spine1": "chest", "Spine2": "chest",
    "LeftShoulder": "arm", "LeftArm": "arm", "LeftForeArm": "arm",
    "RightShoulder": "arm", "RightArm": "arm", "RightForeArm": "arm",
    "LeftHand": "hand", "RightHand": "hand",
    "Hips": "leg",
    "LeftUpLeg": "leg", "LeftLeg": "leg", "LeftFoot": "leg",
    "RightUpLeg": "leg", "RightLeg": "leg", "RightFoot": "leg",
}


class HumanTracker:
    """Tracks a human avatar's skeletal joints in the Isaac Sim stage.

    The tracker works in two modes:
      1. Isaac Sim mode: reads UsdSkel joint positions from a character prim.
      2. Standalone mode: positions are set programmatically via waypoints
         (for testing without Isaac Sim).
    """

    def __init__(self, config: Dict[str, Any]):
        human_cfg = config.get("safety", {}).get("human", {})
        self._prim_path = human_cfg.get("prim_path", "/World/Human")
        self._usd_path = human_cfg.get("usd_path", "")
        self._start_pos = np.array(
            human_cfg.get("start_position", [2.0, 0.0, 0.0]), dtype=np.float64
        )
        self._orientation = np.array(
            human_cfg.get("orientation", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64
        )

        self._joint_positions: Dict[str, np.ndarray] = {}
        self._prev_joint_positions: Dict[str, np.ndarray] = {}
        self._joint_velocities: Dict[str, np.ndarray] = {}
        self._last_update_time: float = 0.0
        self._spawned = False
        self._stage = None

        self._base_position = self._start_pos.copy()
        self._skeleton_offsets = self._build_default_skeleton()

    def _build_default_skeleton(self) -> Dict[str, np.ndarray]:
        """Default T-pose offsets relative to Hips (approximate adult)."""
        return {
            "Hips":           np.array([0.0,  0.0,  0.95]),
            "Spine":          np.array([0.0,  0.0,  1.05]),
            "Spine1":         np.array([0.0,  0.0,  1.15]),
            "Spine2":         np.array([0.0,  0.0,  1.30]),
            "Neck":           np.array([0.0,  0.0,  1.50]),
            "Head":           np.array([0.0,  0.0,  1.70]),
            "LeftShoulder":   np.array([0.0,  0.15, 1.40]),
            "LeftArm":        np.array([0.0,  0.30, 1.40]),
            "LeftForeArm":    np.array([0.0,  0.50, 1.40]),
            "LeftHand":       np.array([0.0,  0.65, 1.40]),
            "RightShoulder":  np.array([0.0, -0.15, 1.40]),
            "RightArm":       np.array([0.0, -0.30, 1.40]),
            "RightForeArm":   np.array([0.0, -0.50, 1.40]),
            "RightHand":      np.array([0.0, -0.65, 1.40]),
            "LeftUpLeg":      np.array([0.0,  0.10, 0.85]),
            "LeftLeg":        np.array([0.0,  0.10, 0.45]),
            "LeftFoot":       np.array([0.0,  0.10, 0.05]),
            "RightUpLeg":     np.array([0.0, -0.10, 0.85]),
            "RightLeg":       np.array([0.0, -0.10, 0.45]),
            "RightFoot":      np.array([0.0, -0.10, 0.05]),
        }

    def spawn(self, stage=None, nucleus_root: str = ""):
        """Spawn the human avatar onto the Isaac Sim stage."""
        self._stage = stage
        if stage is not None and self._usd_path:
            try:
                from omni.isaac.core.utils.stage import add_reference_to_stage
                from pxr import UsdGeom, Gf

                usd_full = nucleus_root + self._usd_path
                add_reference_to_stage(usd_path=usd_full, prim_path=self._prim_path)

                prim = stage.GetPrimAtPath(self._prim_path)
                if prim.IsValid():
                    xformable = UsdGeom.Xformable(prim)
                    xformable.ClearXformOpOrder()
                    xformable.AddTranslateOp().Set(Gf.Vec3d(*self._start_pos.tolist()))
                    q = self._orientation
                    xformable.AddOrientOp().Set(Gf.Quatd(q[0], q[1], q[2], q[3]))
                    self._spawned = True
                    print(f"[HumanTracker] Spawned human at {self._prim_path}")
            except ImportError:
                print("[HumanTracker] Isaac Sim not available, using standalone mode")

        self._update_joint_positions_from_base()
        self._last_update_time = time.perf_counter()

    def update(self, dt: Optional[float] = None):
        """Read current joint positions and compute velocities.

        Args:
            dt: Time delta since last call. If None, computed from wall clock.
        """
        now = time.perf_counter()
        if dt is None:
            dt = now - self._last_update_time if self._last_update_time > 0 else 0.01
        self._last_update_time = now

        self._prev_joint_positions = dict(self._joint_positions)

        if self._spawned and self._stage is not None:
            self._read_skeleton_from_stage()
        else:
            self._update_joint_positions_from_base()

        if dt > 1e-6:
            for name in self._joint_positions:
                prev = self._prev_joint_positions.get(name)
                if prev is not None:
                    self._joint_velocities[name] = (
                        self._joint_positions[name] - prev
                    ) / dt
                else:
                    self._joint_velocities[name] = np.zeros(3)

    def _read_skeleton_from_stage(self):
        """Read joint positions from the UsdSkel binding on the prim."""
        try:
            from pxr import UsdSkel, Gf, Vt

            prim = self._stage.GetPrimAtPath(self._prim_path)
            if not prim.IsValid():
                self._update_joint_positions_from_base()
                return

            skel_root = UsdSkel.Root.Find(prim)
            if not skel_root:
                self._update_joint_positions_from_base()
                return

            binding = UsdSkel.BindingAPI(prim)
            skel_query = binding.GetInheritedSkeletonQuery()
            if not skel_query:
                self._update_joint_positions_from_base()
                return

            xforms = skel_query.ComputeJointWorldTransforms(Gf.Interval.All)
            joint_order = skel_query.GetJointOrder()

            for i, joint_path in enumerate(joint_order):
                joint_name = str(joint_path).split("/")[-1]
                if joint_name in TRACKED_JOINTS and i < len(xforms):
                    t = xforms[i].ExtractTranslation()
                    self._joint_positions[joint_name] = np.array([t[0], t[1], t[2]])

        except Exception:
            self._update_joint_positions_from_base()

    def _update_joint_positions_from_base(self):
        """Compute world positions from base position + skeleton offsets."""
        for name, offset in self._skeleton_offsets.items():
            self._joint_positions[name] = self._base_position + offset

    # ------------------------------------------------------------------
    # Motion control (for testing without Isaac Sim animation)
    # ------------------------------------------------------------------

    def set_base_position(self, position: np.ndarray):
        """Teleport the human base to a new position."""
        self._base_position = np.array(position, dtype=np.float64)

    def move_toward(self, target: np.ndarray, speed: float, dt: float):
        """Move the human base toward a target at given speed.

        Returns True if the target has been reached.
        """
        direction = target - self._base_position
        dist = np.linalg.norm(direction)
        if dist < 0.01:
            return True
        direction_norm = direction / dist
        step = min(speed * dt, dist)
        self._base_position += direction_norm * step
        return False

    def move_along_waypoints(
        self, waypoints: List[np.ndarray], speed: float, dt: float
    ) -> bool:
        """Move through a sequence of waypoints. Returns True when all done."""
        if not waypoints:
            return True
        if self.move_toward(waypoints[0], speed, dt):
            waypoints.pop(0)
        return len(waypoints) == 0

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_joint_positions(self) -> Dict[str, np.ndarray]:
        return dict(self._joint_positions)

    def get_joint_velocities(self) -> Dict[str, np.ndarray]:
        return dict(self._joint_velocities)

    def get_max_speed(self) -> float:
        """Maximum speed across all tracked joints."""
        if not self._joint_velocities:
            return 0.0
        return max(np.linalg.norm(v) for v in self._joint_velocities.values())

    def get_closest_joint_speed(self, joint_name: str) -> float:
        """Speed of a specific joint."""
        vel = self._joint_velocities.get(joint_name)
        return float(np.linalg.norm(vel)) if vel is not None else 0.0

    @property
    def base_position(self) -> np.ndarray:
        return self._base_position.copy()

    @staticmethod
    def body_region(joint_name: str) -> str:
        """Map a joint name to an ISO 15066 body region."""
        return ISO_BODY_REGION_MAP.get(joint_name, "chest")
