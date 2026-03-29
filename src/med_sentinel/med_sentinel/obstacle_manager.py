"""Obstacle manager for Med-Sentinel 360 ("Medical Chaos").

Spawns, randomizes, and clears medical obstacles (carts, IV poles, trays)
within a defined radius of the robot using USD APIs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from med_sentinel.utils.transforms import (
    Pose,
    random_position_in_radius,
    random_yaw_quaternion,
)


@dataclass
class SpawnedObstacle:
    """Record of a spawned obstacle prim."""

    prim_path: str
    asset_type: str
    pose: Pose


class ObstacleManager:
    """Spawns and manages medical obstacles around the robot.

    Uses the USD stage API to add reference prims for each obstacle,
    applying random poses within a configurable radius.
    """

    def __init__(
        self,
        world,
        stage,
        nucleus_root: str,
        robot_prim_path: str,
        config: Dict[str, Any],
    ):
        self._world = world
        self._stage = stage
        self._nucleus_root = nucleus_root
        self._robot_prim_path = robot_prim_path

        obs_cfg = config["obstacles"]
        self._spawn_radius = obs_cfg["spawn_radius"]
        self._min_distance = obs_cfg["min_distance"]
        self._ground_z = obs_cfg["ground_z"]
        self._asset_catalog = obs_cfg["assets"]

        self._spawned: List[SpawnedObstacle] = []
        self._spawn_counter = 0

    @property
    def spawned_obstacles(self) -> List[SpawnedObstacle]:
        return list(self._spawned)

    @property
    def count(self) -> int:
        return len(self._spawned)

    def _get_robot_position(self) -> np.ndarray:
        """Read the robot base position from the stage."""
        from pxr import UsdGeom

        prim = self._stage.GetPrimAtPath(self._robot_prim_path)
        if not prim.IsValid():
            return np.zeros(3)
        xformable = UsdGeom.Xformable(prim)
        xform = xformable.ComputeLocalToWorldTransform(0)
        translation = xform.ExtractTranslation()
        return np.array([translation[0], translation[1], translation[2]])

    def _is_valid_placement(self, position: np.ndarray) -> bool:
        """Check that the new position doesn't collide with existing obstacles."""
        for obs in self._spawned:
            dist = np.linalg.norm(position[:2] - obs.pose.position[:2])
            if dist < self._min_distance:
                return False

        robot_pos = self._get_robot_position()
        if np.linalg.norm(position[:2] - robot_pos[:2]) < self._min_distance:
            return False

        return True

    def _find_valid_position(
        self,
        center: np.ndarray,
        rng: random.Random,
        max_attempts: int = 50,
    ) -> Optional[np.ndarray]:
        """Try to find a collision-free position within the spawn radius."""
        for _ in range(max_attempts):
            pos = random_position_in_radius(
                center, self._spawn_radius, self._ground_z, rng
            )
            if self._is_valid_placement(pos):
                return pos
        return None

    def spawn_obstacle(
        self,
        asset_type: str,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> Optional[SpawnedObstacle]:
        """Spawn a single obstacle of the given type.

        Args:
            asset_type: Key into the asset catalog (e.g. "medical_cart").
            position: Optional explicit position. If None, uses a random valid placement.
            orientation: Optional quaternion (w,x,y,z). If None, uses random yaw.

        Returns:
            The SpawnedObstacle record, or None if placement failed.
        """
        if asset_type not in self._asset_catalog:
            available = list(self._asset_catalog.keys())
            raise ValueError(
                f"Unknown asset type '{asset_type}'. Available: {available}"
            )

        from omni.isaac.core.utils.stage import add_reference_to_stage
        from pxr import UsdGeom, Gf

        asset_info = self._asset_catalog[asset_type]
        usd_path = self._nucleus_root + asset_info["usd_path"]

        self._spawn_counter += 1
        prim_path = f"/World/Obstacles/{asset_type}_{self._spawn_counter:04d}"

        if position is None:
            robot_pos = self._get_robot_position()
            rng = random.Random()
            position = self._find_valid_position(robot_pos, rng)
            if position is None:
                print(f"[ObstacleManager] Could not find valid placement for {asset_type}")
                return None

        if orientation is None:
            orientation = random_yaw_quaternion()

        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"[ObstacleManager] Failed to create prim at {prim_path}")
            return None

        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()

        xformable.AddTranslateOp().Set(Gf.Vec3d(*position.tolist()))
        xformable.AddOrientOp().Set(
            Gf.Quatd(float(orientation[0]), float(orientation[1]),
                     float(orientation[2]), float(orientation[3]))
        )

        scale = asset_info.get("scale", [1.0, 1.0, 1.0])
        xformable.AddScaleOp().Set(Gf.Vec3d(*scale))

        pose = Pose(position=position, orientation=orientation)
        record = SpawnedObstacle(prim_path=prim_path, asset_type=asset_type, pose=pose)
        self._spawned.append(record)

        print(f"[ObstacleManager] Spawned {asset_type} at {prim_path} "
              f"pos=({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
        return record

    def randomize(self, count: int, seed: Optional[int] = None) -> List[SpawnedObstacle]:
        """Randomly spawn N obstacles from the asset catalog.

        Args:
            count: Number of obstacles to spawn.
            seed: Optional random seed for reproducibility.

        Returns:
            List of successfully spawned obstacles.
        """
        rng = random.Random(seed)
        asset_types = list(self._asset_catalog.keys())
        results = []

        for _ in range(count):
            asset_type = rng.choice(asset_types)
            robot_pos = self._get_robot_position()
            position = self._find_valid_position(robot_pos, rng)
            if position is None:
                print("[ObstacleManager] No more valid placements available")
                break

            orientation = random_yaw_quaternion(rng)
            record = self.spawn_obstacle(asset_type, position, orientation)
            if record is not None:
                results.append(record)

        print(f"[ObstacleManager] Spawned {len(results)}/{count} obstacles")
        return results

    def clear(self):
        """Remove all spawned obstacles from the stage."""
        from pxr import Sdf

        for obs in self._spawned:
            prim = self._stage.GetPrimAtPath(obs.prim_path)
            if prim.IsValid():
                self._stage.RemovePrim(obs.prim_path)

        container = self._stage.GetPrimAtPath("/World/Obstacles")
        if container and container.IsValid():
            self._stage.RemovePrim("/World/Obstacles")

        cleared_count = len(self._spawned)
        self._spawned.clear()
        print(f"[ObstacleManager] Cleared {cleared_count} obstacles")

    def get_obstacles(self) -> List[Dict[str, Any]]:
        """Return info about all spawned obstacles."""
        return [
            {
                "prim_path": obs.prim_path,
                "asset_type": obs.asset_type,
                "position": obs.pose.pos_tuple,
                "orientation": obs.pose.quat_tuple,
            }
            for obs in self._spawned
        ]
