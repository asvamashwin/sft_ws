# Med-Sentinel 360 -- Module Reference

## Package Structure

```
src/med_sentinel/
├── description/
│   ├── urdf/
│   │   ├── panda.urdf              # Franka Panda URDF (mesh paths updated)
│   │   └── panda_collision.urdf    # Collision-only URDF
│   ├── srdf/
│   │   └── panda.srdf              # Planning groups, named states, collision pairs
│   └── meshes/
│       ├── collision/*.stl         # Collision meshes
│       └── visual/*.dae            # Visual meshes
├── proto/
│   └── med_sentinel.proto          # Protobuf schema definitions
├── config/
│   └── scene_params.yaml           # All scene configuration
├── launch/
│   └── med_sentinel.launch.py      # ROS2 launch file
├── med_sentinel/
│   ├── scene_builder.py            # Isaac Sim World + Stage setup
│   ├── robot_model.py              # Pinocchio FK/IK/Jacobians/Dynamics
│   ├── robot_controller.py         # OmniGraph + joint control
│   ├── obstacle_manager.py         # "Medical Chaos" obstacle spawner
│   ├── utils/
│   │   └── transforms.py           # Pinocchio SE3/Quaternion utilities
│   └── bridge/
│       ├── med_sentinel_pb2.py     # Generated protobuf bindings
│       ├── proto_handler.py        # Protobuf serialization helpers
│       ├── server.py               # FastAPI WebSocket server
│       ├── sim_bridge.py           # Isaac Sim <-> server bridge
│       ├── benchmark.py            # Latency benchmark client
│       └── stress_test.py          # Multi-client stress test
├── setup.py
├── setup.cfg
└── package.xml
```

---

## scene_builder.py

**Purpose**: Initializes Isaac Sim, imports the hospital USD environment,
and spawns the Franka Panda robot.

### Class: `MedSentinelScene`

| Method | What it does |
|--------|-------------|
| `__init__(config, headless)` | Creates `SimulationApp`, initializes `World`, connects to Nucleus |
| `build()` | Adds ground plane, imports hospital USD, spawns Franka robot |
| `reset()` | Resets the world and applies default joint positions |
| `step()` | Advances simulation by one physics/render step |
| `close()` | Shuts down the SimulationApp |

### Key Properties

- `world` -- the `omni.isaac.core.World` instance
- `stage` -- the USD stage (`pxr.Usd.Stage`)
- `robot` -- the spawned `Robot` prim wrapper
- `panda_model` -- `PandaModel` instance (Pinocchio, loaded from URDF)
- `robot_description` -- raw URDF XML string (for ROS2 `robot_description` param)

### How the hospital is imported

The hospital USD is referenced from NVIDIA's Nucleus asset server using
`add_reference_to_stage()`. The prim is placed at the position/orientation
defined in `scene_params.yaml`. This is a USD reference (not a copy), so
the original asset stays on the Nucleus server and is loaded at runtime.

### How the robot is spawned

The Franka USD model is similarly referenced from Nucleus. An
`omni.isaac.core.robots.Robot` wrapper is created around the prim, which
gives access to joint control APIs (`get_joint_positions`,
`set_joint_positions`, `get_articulation_controller`).

---

## robot_model.py

**Purpose**: Pinocchio-based model of the Franka Panda, loaded from the
vendored URDF/SRDF. Provides FK, IK, Jacobians, dynamics, and collision checking.

### Class: `PandaModel`

| Method | What it does |
|--------|-------------|
| `forward_kinematics(q)` | Compute EE pose for joint config q |
| `frame_placement(q, frame_name)` | Pose of any named frame |
| `all_link_poses(q)` | Poses of all joint frames |
| `jacobian(q, frame)` | 6xN frame Jacobian (local world-aligned) |
| `inverse_kinematics(target, q_init)` | Damped least-squares CLIK solver |
| `mass_matrix(q)` | Joint-space inertia matrix M(q) |
| `coriolis_matrix(q, dq)` | Coriolis matrix C(q, dq) |
| `gravity_torques(q)` | Gravity compensation torques g(q) |
| `inverse_dynamics(q, dq, ddq)` | Full RNEA: tau = M*ddq + C*dq + g |
| `check_self_collision(q)` | True if self-collision detected |
| `is_within_limits(q)` | True if all joints within limits |
| `default_configuration()` | SRDF "default" named state |

### Properties

- `nq`, `nv` -- configuration / velocity space dimensions (both 9)
- `joint_names` -- ordered list of joint names
- `lower_limits`, `upper_limits` -- position limits from URDF
- `velocity_limits`, `effort_limits` -- rate and torque limits

### IK Algorithm

Uses CLIK (Closed-Loop Inverse Kinematics) with damped least-squares:
1. Compute SE3 error between current and target via `pin.log6()`
2. Compute local frame Jacobian
3. Solve `(J^T J + lambda I) dq = J^T error` for velocity step
4. Integrate with `pin.integrate()` (respects joint topology)
5. Repeat until error norm < tolerance

---

## robot_controller.py

**Purpose**: Builds the OmniGraph Action Graph programmatically and
provides a clean API for commanding joint positions.

### Class: `FrankaController`

| Method | What it does |
|--------|-------------|
| `setup()` | Gets ArticulationController from robot, builds OmniGraph |
| `set_joint_positions(positions)` | Command all 9 joints (7 arm + 2 finger) |
| `set_arm_positions(positions)` | Command only the 7 arm joints |
| `get_joint_positions()` | Read current joint angles |
| `get_joint_velocities()` | Read current joint velocities |
| `move_to_default()` | Move to the config-defined home position |

### OmniGraph nodes created

The Action Graph is built using `og.Controller.edit()` with these nodes:

1. **OnPlaybackTick** -- fires every simulation step
2. **ROS2Context** -- initializes the ROS2 bridge
3. **ROS2PublishJointState** -- publishes joint state to `/joint_states`
4. **ROS2SubscribeJointState** -- receives commands from `/joint_commands`
5. **IsaacArticulationController** -- drives the robot joints

All nodes are wired together in code -- no GUI interaction needed.

---

## obstacle_manager.py

**Purpose**: The "Medical Chaos" class. Spawns, randomizes, and clears
medical obstacles (carts, IV poles, trays) around the robot.

### Class: `ObstacleManager`

| Method | What it does |
|--------|-------------|
| `spawn_obstacle(asset_type, position, orientation)` | Spawn one obstacle at a specific or random pose |
| `randomize(count, seed)` | Spawn N random obstacles from the asset catalog |
| `clear()` | Remove all spawned obstacles from the stage |
| `get_obstacles()` | List all spawned obstacles with their poses |

### Collision-free placement

`randomize()` uses `_find_valid_position()` which:
1. Samples a random (x, y) within `spawn_radius` of the robot
2. Checks that it's at least `min_distance` from all existing obstacles
3. Checks that it's at least `min_distance` from the robot base
4. Retries up to 50 times before giving up

### Asset catalog

Defined in `scene_params.yaml` under `obstacles.assets`. Each entry has:
- `usd_path` -- relative to Nucleus root
- `scale` -- XYZ scale factors

---

## utils/transforms.py

**Purpose**: Spatial math backed by Pinocchio. All quaternions use (w, x, y, z)
convention (Isaac Sim standard). Pinocchio uses (x, y, z, w) internally --
the conversion is handled transparently.

### `Pose` dataclass (wraps `pin.SE3`)

| Method/Property | What it does |
|----------------|-------------|
| `from_xyzq(x, y, z, qw, qx, qy, qz)` | Construct from position + quaternion |
| `from_se3(se3)` | Construct from a Pinocchio SE3 |
| `compose(other)` | Transform composition: `self * other` |
| `inverse()` | Inverse transform |
| `act_on_point(point)` | Transform a 3D point |
| `position`, `orientation` | Numpy arrays |
| `se3` | Underlying `pin.SE3` object |

### Quaternion Functions

| Function | What it does |
|----------|-------------|
| `quaternion_multiply(q1, q2)` | Product via `pin.Quaternion` |
| `quaternion_inverse(q)` | Conjugate of unit quaternion |
| `quaternion_slerp(q1, q2, t)` | Spherical linear interpolation |
| `quaternion_from_euler(r, p, y)` | RPY -> quaternion via `pin.rpy` |
| `euler_from_quaternion(q)` | Quaternion -> RPY via `pin.rpy` |
| `quat_wxyz_to_pin(q)` | (w,x,y,z) -> `pin.Quaternion` |
| `rotation_to_quat_wxyz(R)` | Rotation matrix -> (w,x,y,z) |

### Random Sampling

| Function | What it does |
|----------|-------------|
| `random_yaw_quaternion()` | Random rotation about Z-axis |
| `random_position_in_radius(center, radius, z)` | Random XY position in a circle |

---

## bridge/proto_handler.py

**Purpose**: Converts between numpy/Python types and Protobuf messages.

| Function | What it does |
|----------|-------------|
| `build_robot_state(...)` | Numpy arrays -> RobotState protobuf |
| `parse_control_command(data)` | Wire bytes -> Python dict |
| `build_control_command(...)` | Python args -> ControlCommand protobuf |
| `build_ping(sequence)` | Create a PingPong message with client timestamp |
| `stamp_ping_server(data)` | Add server timestamps to a received PingPong |

---

## bridge/server.py

**Purpose**: FastAPI application with a binary WebSocket endpoint.

### Endpoints

| Endpoint | Method | What it does |
|----------|--------|-------------|
| `/ws` | WebSocket | Bidirectional binary channel |
| `/health` | GET | Server health + client count |
| `/stats` | GET | Detailed telemetry statistics |

### How telemetry flows

1. `SimBridge.sample_and_push()` calls `update_robot_state(bytes)` to push
   serialized RobotState into `bridge.latest_robot_state`
2. `_telemetry_sender()` runs per-client, reading `latest_robot_state`
   every 10ms (100 Hz) and sending it as a WebSocket binary frame
3. `_command_receiver()` runs per-client, parsing incoming ControlCommand
   frames and storing them in `bridge.latest_command`

---

## bridge/sim_bridge.py

**Purpose**: Glues Isaac Sim to the WebSocket server.

### Class: `SimBridge`

| Method | What it does |
|--------|-------------|
| `sample_and_push()` | Read robot state, serialize, push to server (rate-limited to 100 Hz) |
| `pull_command()` | Get latest ControlCommand from a web client |
| `apply_command(controller)` | Pull + apply command to FrankaController |

### `start_server_thread()`

Launches the FastAPI server in a daemon thread so it doesn't block
the Isaac Sim main loop.

---

## bridge/benchmark.py

**Purpose**: Measures end-to-end latency and throughput.

### Tests

1. **Ping/Pong test**: Sends N PingPong messages, measures wall-clock RTT
2. **Telemetry test**: Receives frames for N seconds, measures actual Hz

### Usage

```bash
python -m med_sentinel.bridge.benchmark --url ws://localhost:8765/ws --count 1000
```

---

## bridge/stress_test.py

**Purpose**: Simulates multiple concurrent clients under load.

Each client simultaneously receives telemetry and sends ControlCommands
at a configurable rate. Reports per-client and aggregate statistics.

### Usage

```bash
python -m med_sentinel.bridge.stress_test --clients 5 --duration 30 --cmd-hz 100
```
