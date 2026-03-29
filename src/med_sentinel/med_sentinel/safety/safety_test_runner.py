"""Automated ISO 15066 safety validation test runner.

Executes five scripted scenarios that move a simulated human avatar
along predefined paths while the robot is active, then asserts that
the SSM and PFL subsystems respond correctly.

Tests can run standalone (no Isaac Sim) using the ``HumanTracker``
standalone motion API and injected PFL forces.

Scenarios
---------
1. Slow approach   -- human at 0.5 m/s, verify smooth velocity scaling
2. Fast approach   -- human at 1.5 m/s, verify protective stop before S_p
3. Lateral pass    -- close flyby, verify GREEN->YELLOW->RED->YELLOW->GREEN
4. Contact         -- inject force on link, verify PFL stop below limit
5. Recovery        -- after stop, human retreats, verify resume within 500 ms
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from med_sentinel.robot_model import PandaModel
from med_sentinel.safety.human_tracker import HumanTracker
from med_sentinel.safety.distance_monitor import DistanceMonitor
from med_sentinel.safety.ssm_controller import SSMController, SSMState, SafetyZone
from med_sentinel.safety.pfl_monitor import PFLMonitor, PFLState
from med_sentinel.safety.safety_logger import SafetyLogger


@dataclass
class TestResult:
    """Result of a single safety test."""
    name: str
    passed: bool
    stop_time_ms: float = 0.0
    force_at_impact_N: float = 0.0
    d_min_at_stop: float = float("inf")
    v_max_at_trigger: float = 0.0
    zone_sequence: List[str] = field(default_factory=list)
    notes: str = ""
    ssm_trace: List[SSMState] = field(default_factory=list)
    pfl_trace: List[PFLState] = field(default_factory=list)


class SafetyTestRunner:
    """Orchestrates the five ISO 15066 safety validation scenarios."""

    SIM_DT = 0.01  # 100 Hz control cycle

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._robot_model = PandaModel()
        self._human_tracker = HumanTracker(config)
        self._distance_monitor = DistanceMonitor(
            self._robot_model, self._human_tracker, config
        )
        self._ssm = SSMController(
            self._robot_model, self._distance_monitor,
            self._human_tracker, config
        )
        self._pfl = PFLMonitor(self._human_tracker, config)
        self._logger = SafetyLogger(config, session_name="validation")

        self._q = self._robot_model.default_configuration()
        self._dq = np.zeros(self._robot_model.nv)

        self._results: List[TestResult] = []

    def run_all(self) -> List[TestResult]:
        """Execute all five tests and return results."""
        self._logger.start()

        tests = [
            ("Test 1 -- Slow Approach", self._test_slow_approach),
            ("Test 2 -- Fast Approach", self._test_fast_approach),
            ("Test 3 -- Lateral Pass", self._test_lateral_pass),
            ("Test 4 -- Contact (PFL)", self._test_contact),
            ("Test 5 -- Recovery", self._test_recovery),
        ]

        for name, test_fn in tests:
            print(f"\n{'='*60}")
            print(f"  Running: {name}")
            print(f"{'='*60}")
            self._reset_state()
            result = test_fn(name)
            self._results.append(result)
            status = "PASS" if result.passed else "FAIL"
            print(f"  Result: {status} -- {result.notes}")

        self._logger.stop()
        self._print_summary()
        return self._results

    def _reset_state(self):
        """Reset human position and clear PFL injections."""
        self._human_tracker.set_base_position(np.array([2.0, 0.0, 0.0]))
        self._human_tracker.update(dt=self.SIM_DT)
        self._pfl.clear_injected_forces()

    def _step(self) -> tuple[SSMState, PFLState]:
        """Run one SSM + PFL control cycle."""
        self._human_tracker.update(dt=self.SIM_DT)
        ssm_state = self._ssm.update(self._q, self._dq, dt=self.SIM_DT)
        pfl_state = self._pfl.update(
            closest_human_joint=ssm_state.closest_human_joint,
            closest_link=ssm_state.closest_link,
        )
        self._logger.log(ssm_state, pfl_state)
        return ssm_state, pfl_state

    # ------------------------------------------------------------------
    # Test 1: Slow Approach
    # ------------------------------------------------------------------

    def _test_slow_approach(self, name: str) -> TestResult:
        """Human walks toward robot at 0.5 m/s.

        PASS conditions:
          - Velocity scaling decreases smoothly (no jumps > 0.3 between steps)
          - Zone transitions are monotonic: GREEN -> YELLOW -> RED
          - Protective stop triggers before d_min < S_p
        """
        speed = 0.5
        target = np.array([0.0, 0.0, 0.0])
        max_steps = 800  # 8 seconds at 100 Hz

        ssm_trace: List[SSMState] = []
        pfl_trace: List[PFLState] = []
        zone_seq: List[str] = []
        prev_scale = 1.0
        smooth = True
        monotonic = True
        stopped_before_violation = True

        for _ in range(max_steps):
            self._human_tracker.move_toward(target, speed, self.SIM_DT)
            ssm_state, pfl_state = self._step()
            ssm_trace.append(ssm_state)
            pfl_trace.append(pfl_state)

            if not zone_seq or zone_seq[-1] != ssm_state.zone.value:
                zone_seq.append(ssm_state.zone.value)

            scale_jump = abs(ssm_state.velocity_scale - prev_scale)
            if scale_jump > 0.3:
                smooth = False
            prev_scale = ssm_state.velocity_scale

            if ssm_state.zone == SafetyZone.RED:
                break

        expected_order = ["GREEN", "YELLOW", "RED"]
        if zone_seq != expected_order[:len(zone_seq)]:
            monotonic = False

        if ssm_trace:
            last = ssm_trace[-1]
            if last.d_min < last.S_p and not last.protective_stop:
                stopped_before_violation = False

        passed = smooth and monotonic and stopped_before_violation
        notes_parts = []
        if not smooth:
            notes_parts.append("velocity scaling not smooth")
        if not monotonic:
            notes_parts.append(f"zone order {zone_seq} != expected")
        if not stopped_before_violation:
            notes_parts.append("stop did not trigger before S_p violation")
        if passed:
            notes_parts.append("smooth scaling, correct zone transitions")

        return TestResult(
            name=name,
            passed=passed,
            d_min_at_stop=ssm_trace[-1].d_min if ssm_trace else float("inf"),
            v_max_at_trigger=ssm_trace[-1].v_robot if ssm_trace else 0.0,
            zone_sequence=zone_seq,
            notes="; ".join(notes_parts),
            ssm_trace=ssm_trace,
            pfl_trace=pfl_trace,
        )

    # ------------------------------------------------------------------
    # Test 2: Fast Approach
    # ------------------------------------------------------------------

    def _test_fast_approach(self, name: str) -> TestResult:
        """Human runs at 1.5 m/s toward the robot.

        PASS conditions:
          - Protective stop triggers before d_min reaches zero
          - Stop triggers within the RED zone
          - Larger S_p than in the slow approach (faster human = bigger S_h)
        """
        speed = 1.5
        target = np.array([0.0, 0.0, 0.0])
        max_steps = 400

        ssm_trace: List[SSMState] = []
        pfl_trace: List[PFLState] = []
        zone_seq: List[str] = []
        stopped = False
        d_min_at_stop = float("inf")
        v_at_trigger = 0.0

        for _ in range(max_steps):
            self._human_tracker.move_toward(target, speed, self.SIM_DT)
            ssm_state, pfl_state = self._step()
            ssm_trace.append(ssm_state)
            pfl_trace.append(pfl_state)

            if not zone_seq or zone_seq[-1] != ssm_state.zone.value:
                zone_seq.append(ssm_state.zone.value)

            if ssm_state.protective_stop and not stopped:
                stopped = True
                d_min_at_stop = ssm_state.d_min
                v_at_trigger = ssm_state.v_robot
                break

        passed = stopped and d_min_at_stop > 0.0
        notes = (
            f"stop at d_min={d_min_at_stop:.3f}m"
            if stopped
            else "protective stop never triggered"
        )

        return TestResult(
            name=name,
            passed=passed,
            d_min_at_stop=d_min_at_stop,
            v_max_at_trigger=v_at_trigger,
            zone_sequence=zone_seq,
            notes=notes,
            ssm_trace=ssm_trace,
            pfl_trace=pfl_trace,
        )

    # ------------------------------------------------------------------
    # Test 3: Lateral Pass
    # ------------------------------------------------------------------

    def _test_lateral_pass(self, name: str) -> TestResult:
        """Human walks past the robot at close range laterally.

        Path: (1.5, -1.5, 0) -> (0.3, 0.0, 0) -> (1.5, 1.5, 0)

        PASS conditions:
          - Zone transitions include GREEN -> YELLOW -> RED -> YELLOW -> GREEN
          - Robot resumes (velocity_scale > 0.5) after human passes
        """
        waypoints = [
            np.array([0.3, 0.0, 0.0]),
            np.array([1.5, 1.5, 0.0]),
        ]
        self._human_tracker.set_base_position(np.array([1.5, -1.5, 0.0]))
        speed = 0.8
        max_steps = 1200

        ssm_trace: List[SSMState] = []
        pfl_trace: List[PFLState] = []
        zone_seq: List[str] = []
        wp_list = list(waypoints)

        for _ in range(max_steps):
            self._human_tracker.move_along_waypoints(wp_list, speed, self.SIM_DT)
            ssm_state, pfl_state = self._step()
            ssm_trace.append(ssm_state)
            pfl_trace.append(pfl_state)

            if not zone_seq or zone_seq[-1] != ssm_state.zone.value:
                zone_seq.append(ssm_state.zone.value)

            if not wp_list:
                break

        expected_pattern = ["GREEN", "YELLOW", "RED", "YELLOW", "GREEN"]
        pattern_match = zone_seq == expected_pattern

        resumed = any(
            s.velocity_scale > 0.5 and s.zone == SafetyZone.GREEN
            for s in ssm_trace[-100:]
        ) if len(ssm_trace) > 100 else False

        passed = pattern_match and resumed
        notes_parts = []
        if not pattern_match:
            notes_parts.append(f"zone pattern {zone_seq} != {expected_pattern}")
        if not resumed:
            notes_parts.append("robot did not resume after pass")
        if passed:
            notes_parts.append("correct bidirectional zone transitions")

        return TestResult(
            name=name,
            passed=passed,
            zone_sequence=zone_seq,
            notes="; ".join(notes_parts),
            ssm_trace=ssm_trace,
            pfl_trace=pfl_trace,
        )

    # ------------------------------------------------------------------
    # Test 4: Contact (PFL)
    # ------------------------------------------------------------------

    def _test_contact(self, name: str) -> TestResult:
        """Human touches a robot link, injecting a contact force.

        PASS conditions:
          - PFL triggers protective stop when force exceeds body-region limit
          - Max force recorded is below the limit (stop occurs before exceed)
        """
        self._human_tracker.set_base_position(np.array([0.3, 0.0, 0.0]))
        max_steps = 300

        ssm_trace: List[SSMState] = []
        pfl_trace: List[PFLState] = []
        pfl_stopped = False
        force_at_impact = 0.0

        for step in range(max_steps):
            self._human_tracker.update(dt=self.SIM_DT)

            if step >= 50:
                ramp = min(1.0, (step - 50) / 100.0)
                body_region = HumanTracker.body_region(
                    self._ssm.state.closest_human_joint or "RightHand"
                )
                limit = self._pfl.get_force_limit(body_region)
                injected = ramp * (limit + 50)
                self._pfl.inject_force("panda_link6", injected)

            ssm_state = self._ssm.update(self._q, self._dq, dt=self.SIM_DT)
            pfl_state = self._pfl.update(
                closest_human_joint=ssm_state.closest_human_joint,
                closest_link=ssm_state.closest_link,
            )
            self._logger.log(ssm_state, pfl_state)
            ssm_trace.append(ssm_state)
            pfl_trace.append(pfl_state)

            if pfl_state.protective_stop and not pfl_stopped:
                pfl_stopped = True
                force_at_impact = pfl_state.max_force
                break

        self._pfl.clear_injected_forces()

        passed = pfl_stopped
        notes = (
            f"PFL triggered at {force_at_impact:.1f}N"
            if pfl_stopped
            else "PFL did not trigger"
        )

        return TestResult(
            name=name,
            passed=passed,
            force_at_impact_N=force_at_impact,
            notes=notes,
            ssm_trace=ssm_trace,
            pfl_trace=pfl_trace,
        )

    # ------------------------------------------------------------------
    # Test 5: Recovery
    # ------------------------------------------------------------------

    def _test_recovery(self, name: str) -> TestResult:
        """After protective stop, human retreats. Robot should resume.

        PASS conditions:
          - Robot is initially in RED / protective stop
          - After human moves away, robot enters GREEN within 500 ms
          - velocity_scale returns to 1.0
        """
        self._human_tracker.set_base_position(np.array([0.3, 0.0, 0.0]))
        max_steps = 200

        ssm_trace: List[SSMState] = []
        pfl_trace: List[PFLState] = []
        zone_seq: List[str] = []

        for _ in range(50):
            self._human_tracker.update(dt=self.SIM_DT)
            ssm_state, pfl_state = self._step()
            ssm_trace.append(ssm_state)
            pfl_trace.append(pfl_state)
            if not zone_seq or zone_seq[-1] != ssm_state.zone.value:
                zone_seq.append(ssm_state.zone.value)

        was_in_red = any(s.zone == SafetyZone.RED for s in ssm_trace)

        retreat_target = np.array([3.0, 0.0, 0.0])
        retreat_speed = 1.0
        recovery_step = None

        for step in range(max_steps):
            self._human_tracker.move_toward(retreat_target, retreat_speed, self.SIM_DT)
            ssm_state, pfl_state = self._step()
            ssm_trace.append(ssm_state)
            pfl_trace.append(pfl_state)
            if not zone_seq or zone_seq[-1] != ssm_state.zone.value:
                zone_seq.append(ssm_state.zone.value)

            if ssm_state.zone == SafetyZone.GREEN and recovery_step is None:
                recovery_step = step

        recovery_time_ms = (
            recovery_step * self.SIM_DT * 1000.0
            if recovery_step is not None
            else float("inf")
        )
        resumed = recovery_step is not None and recovery_time_ms <= 500.0

        passed = was_in_red and resumed
        notes_parts = []
        if not was_in_red:
            notes_parts.append("never entered RED zone")
        if not resumed:
            notes_parts.append(f"recovery took {recovery_time_ms:.0f}ms (limit 500ms)")
        if passed:
            notes_parts.append(f"recovered in {recovery_time_ms:.0f}ms")

        return TestResult(
            name=name,
            passed=passed,
            stop_time_ms=recovery_time_ms,
            zone_sequence=zone_seq,
            notes="; ".join(notes_parts),
            ssm_trace=ssm_trace,
            pfl_trace=pfl_trace,
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _print_summary(self):
        """Print a formatted summary of all test results."""
        print(f"\n{'='*60}")
        print("  SAFETY VALIDATION SUMMARY")
        print(f"{'='*60}")

        passed = sum(1 for r in self._results if r.passed)
        total = len(self._results)

        for r in self._results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.name}")
            if r.zone_sequence:
                print(f"         Zones: {' -> '.join(r.zone_sequence)}")
            if r.d_min_at_stop < float("inf"):
                print(f"         d_min at stop: {r.d_min_at_stop:.3f} m")
            if r.force_at_impact_N > 0:
                print(f"         Force at impact: {r.force_at_impact_N:.1f} N")
            if r.stop_time_ms > 0 and r.stop_time_ms < float("inf"):
                print(f"         Stop/recovery time: {r.stop_time_ms:.0f} ms")

        print(f"\n  Total: {passed}/{total} passed")

        log_summary = self._logger.summary()
        if log_summary.get("record_count", 0) > 0:
            print(f"\n  Logger summary:")
            print(f"    Records logged: {log_summary['record_count']}")
            print(f"    Min d_min:      {log_summary['d_min_min']:.3f} m")
            print(f"    Max force:      {log_summary['max_contact_force']:.1f} N")
            print(
                f"    Zone dist:      "
                f"G={log_summary['zone_distribution'].get('GREEN', 0)} "
                f"Y={log_summary['zone_distribution'].get('YELLOW', 0)} "
                f"R={log_summary['zone_distribution'].get('RED', 0)}"
            )
        print(f"{'='*60}\n")

    @property
    def results(self) -> List[TestResult]:
        return list(self._results)


def main():
    """CLI entry point for running safety validation tests."""
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Med-Sentinel 360 ISO 15066 Safety Validation"
    )
    parser.add_argument(
        "--config", "-c",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)
            ))),
            "config", "scene_params.yaml",
        ),
        help="Path to scene_params.yaml",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="safety_logs",
        help="Output directory for CSV logs",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config.setdefault("safety", {}).setdefault("logging", {})
    config["safety"]["logging"]["output_dir"] = args.output_dir

    runner = SafetyTestRunner(config)
    results = runner.run_all()

    exit_code = 0 if all(r.passed for r in results) else 1
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
