# Med-Sentinel 360 -- Module Reference

## Package Structure

```
src/med_sentinel/
├── proto/
│   └── med_sentinel.proto          # Protobuf schema definitions
├── config/
│   └── scene_params.yaml           # All scene configuration
├── launch/
│   └── med_sentinel.launch.py      # ROS2 launch file
├── med_sentinel/
│   ├── scene_builder.py            # Isaac Sim World + Stage setup
│   ├── robot_controller.py         # OmniGraph + joint control
│   ├── obstacle_manager.py         # "Medical Chaos" obstacle spawner
│   ├── utils/
│   │   └── transforms.py           # Quaternion math utilities
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

**Purpose**: Quaternion and pose math. All rotations use (w, x, y, z) convention.

| Function | What it does |
|----------|-------------|
| `normalize_quaternion(q)` | Normalize to unit length |
| `quaternion_multiply(q1, q2)` | Hamilton product |
| `quaternion_from_euler(r, p, y)` | Euler angles (rad) -> quaternion |
| `euler_from_quaternion(q)` | Quaternion -> Euler angles |
| `random_yaw_quaternion()` | Random rotation about Z-axis |
| `random_position_in_radius(center, radius, z)` | Random XY position in a circle |

### `Pose` dataclass

Holds `position` (3D numpy) and `orientation` (quaternion numpy).
Factory method `Pose.from_xyzq()` for readable construction.

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
