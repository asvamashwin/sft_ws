"""Scene builder for Med-Sentinel 360.

Initializes the Isaac Sim world, imports the hospital environment,
spawns the Franka Panda robot, and loads the Pinocchio robot model
from the vendored URDF for FK/IK/dynamics alongside the simulation.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

import yaml
import numpy as np

from isaacsim import SimulationApp

from med_sentinel.robot_model import PandaModel

HEADLESS = "--headless" in sys.argv

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CONFIG_PATH = os.path.join(_PKG_ROOT, "config", "scene_params.yaml")
_URDF_PATH = os.path.join(_PKG_ROOT, "description", "urdf", "panda.urdf")
_SRDF_PATH = os.path.join(_PKG_ROOT, "description", "srdf", "panda.srdf")


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
        self._panda_model: Optional[PandaModel] = None
        self._robot_description: Optional[str] = None

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
    def panda_model(self) -> Optional[PandaModel]:
        """Pinocchio model loaded from the vendored URDF."""
        return self._panda_model

    @property
    def robot_description(self) -> Optional[str]:
        """Raw URDF XML string for publishing as robot_description parameter."""
        return self._robot_description

    @property
    def sim_app(self) -> SimulationApp:
        return self._sim_app

    def build(self) -> "MedSentinelScene":
        """Build the full scene: ground plane, hospital, robot, and model."""
        self._add_ground_plane()
        self._import_hospital()
        self._spawn_robot()
        self._load_robot_model()
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

    def _load_robot_model(self):
        """Load the Pinocchio model from the vendored URDF/SRDF."""
        self._panda_model = PandaModel(urdf_path=_URDF_PATH, srdf_path=_SRDF_PATH)

        with open(_URDF_PATH, "r") as f:
            self._robot_description = f.read()

        print(f"[MedSentinel] Pinocchio model loaded: "
              f"nq={self._panda_model.nq}, nv={self._panda_model.nv}")

    def reset(self):
        """Reset the world and apply default joint positions from SRDF."""
        self._world.reset()
        if self._robot is not None:
            if self._panda_model is not None:
                default_q = self._panda_model.default_configuration()
                self._robot.set_joint_positions(default_q)
            else:
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
