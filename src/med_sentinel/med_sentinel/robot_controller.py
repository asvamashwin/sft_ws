"""Robot controller for Med-Sentinel 360.

Programmatically creates an OmniGraph Action Graph with:
  - ROS2 Context node
  - Articulation Controller
  - ROS2 Joint State publisher / subscriber

Exposes a set_joint_positions() method for commanding the Franka.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

import numpy as np

from med_sentinel.scene_builder import MedSentinelScene, load_config


class FrankaController:
    """Drives the Franka Panda joints via OmniGraph + ArticulationController."""

    def __init__(self, scene: MedSentinelScene, config: Dict[str, Any]):
        self._scene = scene
        self._config = config
        self._robot_cfg = config["robot"]
        self._graph_cfg = config["omnigraph"]

        self._articulation_controller = None
        self._og_graph = None
        self._joint_count = 9  # 7 arm + 2 finger joints

    def setup(self) -> "FrankaController":
        """Initialize articulation controller and build the action graph."""
        self._setup_articulation_controller()
        self._build_action_graph()
        return self

    def _setup_articulation_controller(self):
        """Get the ArticulationController from the spawned robot."""
        robot = self._scene.robot
        if robot is None:
            raise RuntimeError("Robot not found in scene. Call scene.build() first.")
        self._articulation_controller = robot.get_articulation_controller()

    def _build_action_graph(self):
        """Programmatically create the OmniGraph Action Graph for ROS2 control."""
        import omni.graph.core as og

        graph_path = self._graph_cfg["graph_path"]
        ros2_cfg = self._graph_cfg["ros2_bridge"]
        robot_prim_path = self._robot_cfg["prim_path"]

        keys = og.Controller.Keys
        (graph, nodes, _, _) = og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ROS2Context", "omni.isaac.ros2_bridge.ROS2Context"),
                    ("PublishJointState", "omni.isaac.ros2_bridge.ROS2PublishJointState"),
                    ("SubscribeJointState", "omni.isaac.ros2_bridge.ROS2SubscribeJointState"),
                    ("ArticulationController", "omni.isaac.core_nodes.IsaacArticulationController"),
                ],
                keys.SET_VALUES: [
                    ("ROS2Context.inputs:domain_id", 0),
                    ("PublishJointState.inputs:topicName", ros2_cfg["joint_state_topic"]),
                    ("PublishJointState.inputs:targetPrim", robot_prim_path),
                    ("SubscribeJointState.inputs:topicName", ros2_cfg["joint_command_topic"]),
                    ("ArticulationController.inputs:robotPath", robot_prim_path),
                    ("ArticulationController.inputs:usePath", True),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
                    ("ROS2Context.outputs:context", "PublishJointState.inputs:context"),
                    ("ROS2Context.outputs:context", "SubscribeJointState.inputs:context"),
                    ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                    ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                    ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                    ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
                ],
            },
        )

        self._og_graph = graph
        self._og_nodes = {
            "tick": nodes[0],
            "ros2_context": nodes[1],
            "publisher": nodes[2],
            "subscriber": nodes[3],
            "controller": nodes[4],
        }

    def set_joint_positions(self, positions: List[float]):
        """Command the robot to move to target joint positions.

        Args:
            positions: Target joint angles in radians. Length must match
                       the robot's DOF (9 for Franka: 7 arm + 2 finger).
        """
        target = np.array(positions, dtype=np.float64)
        if len(target) != self._joint_count:
            raise ValueError(
                f"Expected {self._joint_count} joint values, got {len(target)}"
            )
        self._articulation_controller.apply_action(
            joint_positions=target,
        )

    def set_arm_positions(self, positions: List[float]):
        """Command only the 7 arm joints (fingers stay as-is).

        Args:
            positions: 7 joint angles in radians for the arm.
        """
        if len(positions) != 7:
            raise ValueError(f"Expected 7 arm joint values, got {len(positions)}")

        current = self._scene.robot.get_joint_positions()
        target = np.array(current, dtype=np.float64)
        target[:7] = positions
        self.set_joint_positions(target.tolist())

    def get_joint_positions(self) -> np.ndarray:
        """Read the current joint positions."""
        return self._scene.robot.get_joint_positions()

    def get_joint_velocities(self) -> np.ndarray:
        """Read the current joint velocities."""
        return self._scene.robot.get_joint_velocities()

    def move_to_default(self):
        """Move robot to the default joint configuration from config."""
        default = self._robot_cfg.get("default_joint_positions")
        if default:
            self.set_joint_positions(default)


def main():
    headless = "--headless" in sys.argv
    config = load_config()

    scene = MedSentinelScene(config=config, headless=headless)
    scene.build()
    scene.reset()

    controller = FrankaController(scene, config)
    controller.setup()

    print("[MedSentinel] Robot controller ready. Moving to default position...")
    controller.move_to_default()

    step_count = 0
    try:
        while scene.sim_app.is_running():
            scene.step()
            step_count += 1

            if step_count == 300:
                print("[MedSentinel] Sending test position command...")
                controller.set_arm_positions(
                    [0.0, -1.0, 0.0, -2.0, 0.0, 1.2, 0.5]
                )

            if step_count % 600 == 0:
                joints = controller.get_joint_positions()
                print(f"[MedSentinel] Step {step_count} | Joints: {np.round(joints, 3)}")

    except KeyboardInterrupt:
        print("[MedSentinel] Shutting down...")
    finally:
        scene.close()


if __name__ == "__main__":
    main()
