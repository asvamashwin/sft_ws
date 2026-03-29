"""Robot-human distance computation for ISO 15066 SSM.

Approximates each robot link as a capsule (line segment + radius) and
computes the minimum distance from every capsule to every tracked
human joint (treated as a point).

Uses the ``PandaModel.all_link_poses(q)`` API to obtain link transforms,
then extracts the capsule axis from each link's local z-direction scaled
by the configured length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from med_sentinel.robot_model import PandaModel
from med_sentinel.safety.human_tracker import HumanTracker


@dataclass
class CapsuleGeometry:
    """Collision approximation for a single robot link."""
    radius: float
    length: float


@dataclass
class DistanceResult:
    """Result of a minimum-distance query."""
    d_min: float                     # global minimum distance (m)
    closest_link: str                # robot link name
    closest_human_joint: str         # human joint name
    closest_point_robot: np.ndarray  # nearest point on robot capsule
    closest_point_human: np.ndarray  # human joint position
    all_distances: Dict[Tuple[str, str], float]  # (link, joint) -> distance


def _point_to_segment_distance(
    point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Compute minimum distance from a point to a line segment.

    Returns (distance, closest_point_on_segment).
    """
    seg = seg_end - seg_start
    seg_len_sq = np.dot(seg, seg)

    if seg_len_sq < 1e-12:
        return float(np.linalg.norm(point - seg_start)), seg_start.copy()

    t = np.dot(point - seg_start, seg) / seg_len_sq
    t = np.clip(t, 0.0, 1.0)
    closest = seg_start + t * seg
    return float(np.linalg.norm(point - closest)), closest


class DistanceMonitor:
    """Computes minimum distance between robot capsules and human joints.

    The capsule for each link is oriented along the link's local z-axis,
    centred at the link origin.
    """

    def __init__(
        self,
        robot_model: PandaModel,
        human_tracker: HumanTracker,
        config: Dict[str, Any],
    ):
        self._robot_model = robot_model
        self._human_tracker = human_tracker

        capsule_cfg = config.get("safety", {}).get("robot_capsules", {})
        self._capsules: Dict[str, CapsuleGeometry] = {}
        for link_name, geom in capsule_cfg.items():
            self._capsules[link_name] = CapsuleGeometry(
                radius=geom.get("radius", 0.05),
                length=geom.get("length", 0.1),
            )

        if not self._capsules:
            self._capsules = self._default_capsules()

        self._last_result: Optional[DistanceResult] = None

    @staticmethod
    def _default_capsules() -> Dict[str, CapsuleGeometry]:
        return {
            "panda_link0": CapsuleGeometry(0.06, 0.333),
            "panda_link1": CapsuleGeometry(0.06, 0.0),
            "panda_link2": CapsuleGeometry(0.06, 0.316),
            "panda_link3": CapsuleGeometry(0.05, 0.0),
            "panda_link4": CapsuleGeometry(0.05, 0.384),
            "panda_link5": CapsuleGeometry(0.05, 0.0),
            "panda_link6": CapsuleGeometry(0.04, 0.088),
            "panda_link7": CapsuleGeometry(0.04, 0.107),
            "panda_hand":  CapsuleGeometry(0.04, 0.058),
        }

    def compute(self, q: np.ndarray) -> DistanceResult:
        """Compute all pairwise capsule-point distances.

        Args:
            q: Current robot joint configuration.

        Returns:
            DistanceResult with global minimum and per-pair data.
        """
        link_poses = self._robot_model.all_link_poses(q)
        human_joints = self._human_tracker.get_joint_positions()

        best_dist = float("inf")
        best_link = ""
        best_joint = ""
        best_pt_robot = np.zeros(3)
        best_pt_human = np.zeros(3)
        all_distances: Dict[Tuple[str, str], float] = {}

        for link_name, capsule in self._capsules.items():
            pose = link_poses.get(link_name)
            if pose is None:
                joint_name_mapping = self._link_to_joint_name(link_name)
                pose = link_poses.get(joint_name_mapping)
                if pose is None:
                    continue

            origin = pose.position
            rotation = pose.rotation_matrix()
            z_axis = rotation[:, 2]

            half = capsule.length / 2.0
            seg_start = origin - z_axis * half
            seg_end = origin + z_axis * half

            for joint_name, joint_pos in human_joints.items():
                raw_dist, closest_on_seg = _point_to_segment_distance(
                    joint_pos, seg_start, seg_end
                )
                dist = max(0.0, raw_dist - capsule.radius)
                all_distances[(link_name, joint_name)] = dist

                if dist < best_dist:
                    best_dist = dist
                    best_link = link_name
                    best_joint = joint_name
                    best_pt_robot = closest_on_seg
                    best_pt_human = joint_pos.copy()

        self._last_result = DistanceResult(
            d_min=best_dist,
            closest_link=best_link,
            closest_human_joint=best_joint,
            closest_point_robot=best_pt_robot,
            closest_point_human=best_pt_human,
            all_distances=all_distances,
        )
        return self._last_result

    @property
    def last_result(self) -> Optional[DistanceResult]:
        return self._last_result

    @staticmethod
    def _link_to_joint_name(link_name: str) -> str:
        """Map link prim names to Pinocchio joint names.

        Pinocchio names joints (panda_joint1 .. panda_joint8, panda_hand_joint)
        while URDF links are (panda_link0 .. panda_link8, panda_hand).
        all_link_poses() uses the joint names.
        """
        mapping = {
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
        return mapping.get(link_name, link_name)
