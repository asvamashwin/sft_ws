"""Scene builder for Med-Sentinel 360.

Initializes the Isaac Sim world, imports the hospital environment,
and spawns the Franka Panda robot at a configurable pose.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

import yaml
import numpy as np

from isaacsim import SimulationApp

HEADLESS = "--headless" in sys.argv

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "scene_params.yaml",
)


def load_config(path: str = _CONFIG_PATH) -> Dict[str, Any]:
    """Load scene parameters from YAML config."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


class MedSentinelScene:
    """Builds and manages the Isaac Sim scene for Med-Sentinel 360."""

    NUCLEUS_BASE = "omniverse://localhost/NVIDIA/Assets"

    def __init__(self, config: Optional[Dict[str, Any]] = None, headless: bool = False):
        self._config = config or load_config()
        self._headless = headless

        self._sim_app = SimulationApp({
            "headless": self._headless,
            "width": 1920,
            "height": 1080,
        })

        # Isaac Sim imports must happen after SimulationApp is created
        import omni.usd
        from omni.isaac.core import World
        from omni.isaac.core.utils.nucleus import get_assets_root_path

        self._nucleus_root = get_assets_root_path()
        if self._nucleus_root is None:
            self._nucleus_root = self.NUCLEUS_BASE

        scene_cfg = self._config["scene"]
        self._world = World(
            physics_dt=scene_cfg["physics_dt"],
            rendering_dt=scene_cfg["rendering_dt"],
            stage_units_in_meters=1.0,
        )
        self._stage = omni.usd.get_context().get_stage()
        self._robot = None

    @property
    def world(self):
        return self._world

    @property
    def stage(self):
        return self._stage

    @property
    def robot(self):
        return self._robot

    @property
    def sim_app(self) -> SimulationApp:
        return self._sim_app

    def build(self) -> "MedSentinelScene":
        """Build the full scene: ground plane, hospital, and robot."""
        self._add_ground_plane()
        self._import_hospital()
        self._spawn_robot()
        return self

    def _add_ground_plane(self):
        self._world.scene.add_default_ground_plane()

    def _import_hospital(self):
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from pxr import Gf

        hospital_cfg = self._config["hospital"]
        usd_path = self._nucleus_root + hospital_cfg["usd_path"]
        prim_path = hospital_cfg["prim_path"]

        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

        prim = self._stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            from pxr import UsdGeom
            xformable = UsdGeom.Xformable(prim)
            xformable.ClearXformOpOrder()

            pos = hospital_cfg["position"]
            xformable.AddTranslateOp().Set(Gf.Vec3d(*pos))

            q = hospital_cfg["orientation"]
            xformable.AddOrientOp().Set(Gf.Quatd(q[0], q[1], q[2], q[3]))

    def _spawn_robot(self):
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.core.robots import Robot

        robot_cfg = self._config["robot"]
        usd_path = self._nucleus_root + robot_cfg["usd_path"]
        prim_path = robot_cfg["prim_path"]

        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

        pos = np.array(robot_cfg["position"], dtype=np.float64)
        orient = np.array(robot_cfg["orientation"], dtype=np.float64)

        self._robot = self._world.scene.add(
            Robot(
                prim_path=prim_path,
                name="franka",
                position=pos,
                orientation=orient,
            )
        )

    def reset(self):
        """Reset the world and apply default joint positions."""
        self._world.reset()
        if self._robot is not None:
            default_joints = self._config["robot"].get("default_joint_positions")
            if default_joints:
                self._robot.set_joint_positions(np.array(default_joints))

    def step(self):
        """Advance the simulation by one step."""
        self._world.step(render=not self._headless)

    def close(self):
        """Shut down the simulation."""
        self._sim_app.close()


def main():
    config = load_config()
    scene = MedSentinelScene(config=config, headless=HEADLESS)
    scene.build()
    scene.reset()

    print("[MedSentinel] Scene built successfully. Running simulation...")
    try:
        while scene.sim_app.is_running():
            scene.step()
    except KeyboardInterrupt:
        print("[MedSentinel] Shutting down...")
    finally:
        scene.close()


if __name__ == "__main__":
    main()
