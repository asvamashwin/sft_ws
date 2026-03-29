"""Power and Force Limiting (PFL) monitor per ISO 15066.

Attaches contact sensors to each Franka link via Isaac Sim's
``omni.isaac.sensor`` API, reads per-link contact forces each sim step,
and compares against ISO 15066 Table A.2 body-region transient limits.

When Isaac Sim is unavailable (standalone testing), forces can be
injected programmatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from med_sentinel.safety.human_tracker import HumanTracker

CONTACT_SENSOR_LINKS = [
    "panda_link0", "panda_link1", "panda_link2", "panda_link3",
    "panda_link4", "panda_link5", "panda_link6", "panda_link7",
    "panda_hand",
]


@dataclass
class ContactEvent:
    """A single contact reading from one link."""
    link_name: str
    force_magnitude: float   # Newtons
    body_region: str         # ISO 15066 body region of closest human joint
    force_limit: float       # applicable limit for that body region
    exceeded: bool           # True if force > limit


@dataclass
class PFLState:
    """Aggregate PFL state for one control cycle."""
    max_force: float = 0.0
    max_force_link: str = ""
    protective_stop: bool = False
    contacts: List[ContactEvent] = field(default_factory=list)


class PFLMonitor:
    """ISO 15066 Power and Force Limiting monitor.

    Reads contact sensors on every Franka link each sim step and triggers
    a protective stop if any force exceeds the transient body-region limit.
    """

    def __init__(
        self,
        human_tracker: HumanTracker,
        config: Dict[str, Any],
    ):
        self._human_tracker = human_tracker

        pfl_cfg = config.get("safety", {}).get("pfl", {})
        self._force_limits: Dict[str, float] = pfl_cfg.get("force_limits", {
            "head": 130, "chest": 280, "hand": 210, "arm": 250, "leg": 300,
        })
        self._default_limit = pfl_cfg.get("default_limit", 250)

        self._sensors: Dict[str, Any] = {}
        self._sim_mode = False
        self._injected_forces: Dict[str, float] = {}
        self._state = PFLState()

    @property
    def state(self) -> PFLState:
        return self._state

    def setup_sensors(self, stage=None, robot_prim_path: str = "/World/Franka"):
        """Attach contact sensors to each robot link in Isaac Sim."""
        if stage is None:
            return

        try:
            from omni.isaac.sensor import ContactSensor

            for link_name in CONTACT_SENSOR_LINKS:
                sensor_path = f"{robot_prim_path}/{link_name}/contact_sensor"
                sensor = ContactSensor(
                    prim_path=sensor_path,
                    name=f"{link_name}_contact",
                    min_threshold=0.1,
                    max_threshold=10000.0,
                    radius=-1,
                )
                self._sensors[link_name] = sensor

            self._sim_mode = True
            print(f"[PFLMonitor] Attached contact sensors to {len(self._sensors)} links")
        except ImportError:
            print("[PFLMonitor] Isaac Sim sensor API not available, using standalone mode")

    def inject_force(self, link_name: str, force_magnitude: float):
        """Inject a contact force for standalone testing."""
        self._injected_forces[link_name] = force_magnitude

    def clear_injected_forces(self):
        self._injected_forces.clear()

    def update(
        self,
        closest_human_joint: str = "",
        closest_link: str = "",
    ) -> PFLState:
        """Read contact forces and check against ISO limits.

        Args:
            closest_human_joint: Name of the closest human joint (for body
                region lookup).
            closest_link: Name of the closest robot link.

        Returns:
            Updated PFLState.
        """
        forces = self._read_forces()
        body_region = HumanTracker.body_region(closest_human_joint)

        contacts: List[ContactEvent] = []
        max_force = 0.0
        max_force_link = ""
        protective_stop = False

        for link_name, force_mag in forces.items():
            region = body_region if link_name == closest_link else "chest"
            limit = self._force_limits.get(region, self._default_limit)

            exceeded = force_mag > limit
            contacts.append(ContactEvent(
                link_name=link_name,
                force_magnitude=force_mag,
                body_region=region,
                force_limit=limit,
                exceeded=exceeded,
            ))

            if force_mag > max_force:
                max_force = force_mag
                max_force_link = link_name

            if exceeded:
                protective_stop = True

        self._state = PFLState(
            max_force=max_force,
            max_force_link=max_force_link,
            protective_stop=protective_stop,
            contacts=contacts,
        )
        return self._state

    def _read_forces(self) -> Dict[str, float]:
        """Read contact force magnitudes from sensors or injected values."""
        forces: Dict[str, float] = {}

        if self._sim_mode:
            for link_name, sensor in self._sensors.items():
                try:
                    reading = sensor.get_current_frame()
                    if reading and "value" in reading:
                        forces[link_name] = abs(float(reading["value"]))
                    else:
                        forces[link_name] = 0.0
                except Exception:
                    forces[link_name] = 0.0
        else:
            for link_name in CONTACT_SENSOR_LINKS:
                forces[link_name] = self._injected_forces.get(link_name, 0.0)

        return forces

    def get_force_limit(self, body_region: str) -> float:
        """Lookup the ISO 15066 transient force limit for a body region."""
        return self._force_limits.get(body_region, self._default_limit)
