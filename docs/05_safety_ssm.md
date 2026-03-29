# 05 -- ISO 15066 Safety: Speed and Separation Monitoring (SSM)

Med-Sentinel 360 implements a real-time safety layer that enforces
**ISO 15066** collaborative robotics requirements.  This document covers
the mathematical formulation, module design, configuration, and how to
run the validation test suite.

---

## 1. ISO 15066 Overview

ISO 15066 defines safety requirements for collaborative robot systems.
Med-Sentinel 360 implements two of the four collaborative operation modes:

| Mode | Description | Med-Sentinel Module |
|------|-------------|---------------------|
| **Speed and Separation Monitoring (SSM)** | Robot slows/stops based on human proximity | `ssm_controller.py` |
| **Power and Force Limiting (PFL)** | Robot stops if contact force exceeds limits | `pfl_monitor.py` |

---

## 2. Protective Separation Distance Formula

The core of SSM is the **protective separation distance** \( S_p \):

\[
S_p = S_h + S_r + S_s + C + Z_d + Z_r
\]

### 2.1 Term Definitions

| Symbol | Name | Formula | Source in Code |
|--------|------|---------|----------------|
| \( S_h \) | Human contribution | \( v_h \cdot (T_r + T_s) \) | `HumanTracker` joint velocities |
| \( S_r \) | Robot reaction distance | \( v_r \cdot T_r \) | `PandaModel.jacobian(q) @ dq` |
| \( S_s \) | Robot stopping distance | \( \frac{v_r^2}{2 \cdot a_{max}} \) | `PandaModel.mass_matrix(q)` + `effort_limits` |
| \( C \) | Intrusion distance | Constant (sensor-dependent) | Config: `safety.ssm.C_intrusion` |
| \( Z_d \) | Human position uncertainty | Constant | Config: `safety.ssm.Z_d` |
| \( Z_r \) | Robot position uncertainty | Constant | Config: `safety.ssm.Z_r` |

### 2.2 Derived Parameters

| Parameter | Derivation |
|-----------|-----------|
| \( v_h \) | Finite-differenced human joint velocity from `HumanTracker`, capped at `v_human_max` |
| \( v_r \) | Cartesian velocity of the closest robot link: \( \|J_{link}(q) \cdot \dot{q}\|_2 \) |
| \( a_{max} \) | \( \max_i \frac{\tau_{limit,i}}{M_{ii}(q)} \) where \( M \) is the joint-space mass matrix |
| \( T_r \) | System reaction time (sensing + controller latency) |
| \( T_s \) | Worst-case stopping time |

### 2.3 Safety Zones

```
                          d_current
  ───────────────────────────────────────────────────────►

  ┌───────────┐  ┌──────────────────┐  ┌─────────────────┐
  │    RED     │  │      YELLOW      │  │      GREEN      │
  │  v = 0    │  │  v = scaled      │  │   v = full      │
  │  STOP     │  │  V = (d-Sp)/Tr   │  │   V = V_max     │
  └───────────┘  └──────────────────┘  └─────────────────┘
  0            S_p               S_p + margin            ∞
```

| Zone | Condition | Velocity Scale | Action |
|------|-----------|---------------|--------|
| **RED** | \( d \leq S_p \) | 0.0 | Protective stop |
| **YELLOW** | \( S_p < d \leq S_p + m \) | \( \frac{d - S_p}{m} \) | Proportional scaling |
| **GREEN** | \( d > S_p + m \) | 1.0 | Full speed |

Where \( m \) = `safety.ssm.margin` (default 0.3 m).

---

## 3. Power and Force Limiting (PFL)

If the robot makes physical contact despite SSM, the PFL monitor
triggers a protective stop when force exceeds ISO 15066 Table A.2
transient limits.

### ISO 15066 Table A.2 (Transient Contact)

| Body Region | Force Limit (N) | Config Key |
|-------------|-----------------|------------|
| Head/Face   | 130             | `head`     |
| Chest       | 280             | `chest`    |
| Hand/Finger | 210             | `hand`     |
| Arm         | 250             | `arm`      |
| Leg         | 300             | `leg`      |

The body region is determined by the closest human joint to the
contact point, using the mapping in `HumanTracker.body_region()`.

---

## 4. Module Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Safety Pipeline                       │
│                                                         │
│  HumanTracker          DistanceMonitor                  │
│  ┌──────────┐          ┌──────────────┐                 │
│  │ skeleton │─joints──►│ capsule-point│──d_min          │
│  │ tracking │          │   distance   │──link/joint──┐  │
│  └──────────┘          └──────────────┘              │  │
│       │                      ▲                       │  │
│       │v_human        q (from│robot)                 ▼  │
│       │                      │              ┌───────────┐│
│       └──────────────────────┼─────────────►│    SSM    ││
│                              │              │Controller ││
│                              │              └─────┬─────┘│
│                              │                    │      │
│  PFLMonitor                  │            velocity_scale │
│  ┌──────────┐                │                    │      │
│  │ contact  │──force─────────┼──────────┐         │      │
│  │ sensors  │                │          ▼         ▼      │
│  └──────────┘                │   ┌────────────────┐      │
│                              │   │ SafetyLogger   │      │
│                              │   │ CSV + Protobuf │      │
│                              │   └────────────────┘      │
└─────────────────────────────────────────────────────────┘
```

### File Map

| File | Responsibility |
|------|---------------|
| `safety/human_tracker.py` | Spawn human USD, read skeleton joints, compute joint velocities |
| `safety/distance_monitor.py` | Capsule-point minimum distance per link-joint pair |
| `safety/ssm_controller.py` | ISO 15066 S_p formula, zone classification, velocity scaling |
| `safety/pfl_monitor.py` | Contact sensor readings, ISO force limit enforcement |
| `safety/safety_logger.py` | CSV logging + protobuf `SafetyState` telemetry |
| `safety/safety_test_runner.py` | Five automated validation scenarios |

---

## 5. Distance Computation

Each robot link is approximated as a **capsule** (line segment + radius).

```
     Capsule for panda_link4
     ─────────────────────

        ┌─────────────┐
        │  radius=0.05│ ←─ collision cylinder
        │             │
  start ├─────────────┤ end
        │  length=0.384│
        │             │
        └─────────────┘
```

The minimum distance from a human joint (point) to a capsule is:

\[
d = \max\left(0, \; \min_t \| p - (A + t \cdot (B - A)) \| - r \right)
\quad t \in [0, 1]
\]

Where \( A, B \) are the capsule endpoints and \( r \) is the radius.

Capsule parameters per link are configured in
`safety.robot_capsules` in `scene_params.yaml`.

---

## 6. Configuration

All safety parameters live under the `safety:` key in
`config/scene_params.yaml`:

```yaml
safety:
  ssm:
    T_reaction: 0.1          # seconds
    T_stopping: 0.2          # seconds
    C_intrusion: 0.05        # meters
    Z_d: 0.04                # meters
    Z_r: 0.02                # meters
    v_human_max: 1.6          # m/s
    margin: 0.3               # meters

  pfl:
    force_limits:
      head: 130
      chest: 280
      hand: 210
      arm: 250
      leg: 300
    default_limit: 250

  human:
    usd_path: "/Isaac/People/Characters/Male/male.usd"
    prim_path: "/World/Human"
    start_position: [2.0, 0.0, 0.0]

  robot_capsules:
    panda_link4: { radius: 0.05, length: 0.384 }
    # ... (see scene_params.yaml for full list)

  logging:
    output_dir: "safety_logs"
    csv_enabled: true
    proto_enabled: true
```

---

## 7. Protobuf Telemetry

Safety state is streamed at 100 Hz via the existing WebSocket bridge
using the `SafetyState` protobuf message:

```protobuf
message SafetyState {
  Timestamp  stamp             = 1;
  uint64     sequence          = 2;
  double     d_min             = 3;   // min human-robot distance
  double     S_p               = 4;   // protective separation distance
  double     velocity_scale    = 5;   // [0.0 - 1.0]
  string     closest_link      = 6;
  string     closest_human     = 7;
  SafetyZone zone              = 8;   // GREEN / YELLOW / RED
  double     max_contact_force = 9;
  bool       protective_stop   = 10;
  double     v_robot           = 11;
  double     v_human           = 12;
}
```

---

## 8. Running the Validation Tests

### 8.1 Standalone (no Isaac Sim)

The test runner works without Isaac Sim by using programmatic
human motion and injected forces:

```bash
cd ~/sft_ws
source install/setup.bash

# Run all 5 tests
ros2 run med_sentinel safety_test

# Or directly:
python -m med_sentinel.safety.safety_test_runner \
    --config src/med_sentinel/config/scene_params.yaml \
    --output-dir safety_logs
```

### 8.2 With Isaac Sim

When Isaac Sim is available, the test runner will:
- Spawn the human avatar USD on the stage
- Read real skeletal joint positions from the `UsdSkel` API
- Use actual contact sensors instead of injected forces

### 8.3 Test Scenarios

| # | Name | Human Motion | Pass Criteria |
|---|------|-------------|---------------|
| 1 | Slow approach | 0.5 m/s toward robot | Smooth velocity scaling; monotonic GREEN -> YELLOW -> RED |
| 2 | Fast approach | 1.5 m/s toward robot | Protective stop before d_min reaches zero |
| 3 | Lateral pass | 0.8 m/s flyby at 0.3 m | Zone pattern: GREEN -> YELLOW -> RED -> YELLOW -> GREEN |
| 4 | Contact (PFL) | Stationary + ramping force | PFL triggers below ISO force limit |
| 5 | Recovery | Retreat from RED zone | Robot resumes GREEN within 500 ms |

### 8.4 Reading the Output

Test results are printed to the terminal:

```
============================================================
  SAFETY VALIDATION SUMMARY
============================================================
  [PASS] Test 1 -- Slow Approach
         Zones: GREEN -> YELLOW -> RED
         d_min at stop: 0.412 m
  [PASS] Test 2 -- Fast Approach
         d_min at stop: 0.687 m
  ...
  Total: 5/5 passed
============================================================
```

CSV logs are written to `safety_logs/safety_<timestamp>.csv` with
per-cycle data suitable for plotting in Python or Excel.

---

## 9. Tuning Guide

| Symptom | Parameter to Adjust | Direction |
|---------|-------------------|-----------|
| Robot stops too far from human | `margin` | Decrease |
| Robot doesn't stop fast enough | `T_reaction`, `T_stopping` | Increase |
| False PFL triggers | `force_limits.*` | Increase (but stay within ISO) |
| Jittery velocity scaling | `C_intrusion`, `Z_d`, `Z_r` | Increase (adds buffer) |
| Human detected too late | `v_human_max` | Increase |
