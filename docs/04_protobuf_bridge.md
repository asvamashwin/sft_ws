# Med-Sentinel 360 -- Protobuf Bridge Deep Dive

## Why Protobuf over JSON?

For a 100Hz telemetry stream, serialization cost matters:

| Format | RobotState size | Serialize time | Deserialize time |
|--------|----------------|----------------|------------------|
| JSON | ~2 KB | ~150 μs | ~200 μs |
| Protobuf | ~200 bytes | ~15 μs | ~20 μs |

At 100 Hz, that's the difference between 15ms/s and 1.5ms/s of CPU time
just for serialization. With multiple clients, this compounds.

## Schema Design

### RobotState (Server -> Client)

Published at 100 Hz from Isaac Sim. Contains everything a client needs
to render the robot's current state:

```protobuf
message RobotState {
  Timestamp        stamp           = 1;   // When the sample was taken
  uint64           sequence        = 2;   // Monotonic counter for drop detection
  string           robot_name      = 3;   // "franka"
  JointState       joint_state     = 4;   // 9 joints: positions, velocities, efforts
  EndEffectorState end_effector    = 5;   // EE pose + twist
  Pose             base_pose       = 6;   // Robot base in world frame
  RobotStatus      status          = 7;   // IDLE, MOVING, ERROR, ESTOPPED
  double           sim_time        = 8;   // Isaac Sim world clock
}
```

The `sequence` field lets clients detect dropped frames. If you receive
sequence 100 then 103, you know 2 frames were dropped.

### ControlCommand (Client -> Server)

Sent by web clients to command the robot:

```protobuf
message ControlCommand {
  Timestamp       stamp          = 1;   // Client-side timestamp
  uint64          sequence       = 2;   // Client-side sequence counter
  ControlMode     mode           = 3;   // POSITION, VELOCITY, or EFFORT
  repeated double joint_targets  = 4;   // 9 values matching the DOF
  double          max_velocity   = 5;   // Velocity scaling [0-1]
  double          max_accel      = 6;   // Acceleration scaling [0-1]
  bool            stop           = 7;   // Emergency stop
}
```

### PingPong (Bidirectional)

For latency measurement:

```protobuf
message PingPong {
  Timestamp client_send  = 1;   // Client stamps before sending
  Timestamp server_recv  = 2;   // Server stamps on receipt
  Timestamp server_send  = 3;   // Server stamps before reply
  uint64    sequence     = 4;   // Match request to response
}
```

This gives you three measurements:
- **Upstream**: `server_recv - client_send`
- **Server processing**: `server_send - server_recv`
- **Round-trip**: `client_recv - client_send` (measured by client)

## Wire Protocol

Every WebSocket binary frame starts with a 1-byte type tag:

```
┌──────────┬───────────────────────────┐
│ Type (1B)│ Protobuf payload (N bytes)│
└──────────┴───────────────────────────┘
```

| Type byte | Message | Direction |
|-----------|---------|-----------|
| `0x01` | RobotState | Server -> Client |
| `0x02` | ControlCommand | Client -> Server |
| `0x03` | PingPong | Bidirectional |

This avoids the overhead of a separate message type field in the Protobuf
schema and makes dispatching trivial (single byte check).

## Server Architecture

```
FastAPI app
├── GET /health          → {"status": "ok", "clients": 2, ...}
├── GET /stats           → Detailed telemetry statistics
└── WS  /ws              → Binary WebSocket
        ├── _telemetry_sender()   → Push RobotState at 100 Hz (per client)
        └── _command_receiver()   → Parse ControlCommand / PingPong
```

### Shared State: `BridgeState`

The `BridgeState` singleton holds:
- `clients` -- set of connected WebSocket instances
- `latest_robot_state` -- most recent serialized RobotState bytes
- `latest_command` -- most recent parsed ControlCommand dict
- `latencies_us` -- rolling window of latency measurements

The SimBridge pushes state in via `update_robot_state()` and pulls
commands via `get_latest_command()`.

## Latency Budget

Target: <20ms end-to-end (sim sample -> client receives frame)

| Component | Budget |
|-----------|--------|
| Sim sampling | ~1 ms |
| Protobuf serialization | ~0.02 ms |
| WebSocket send | ~0.1 ms (localhost) |
| Network transit | ~0.5 ms (LAN) / ~5ms (WAN) |
| Client deserialize | ~0.02 ms |
| **Total (localhost)** | **~1.6 ms** |
| **Total (LAN)** | **~2 ms** |

The <20ms target is easily achievable on localhost. Over WAN, it depends
on network conditions but the Protobuf encoding keeps payloads small
enough to fit in a single TCP segment.
