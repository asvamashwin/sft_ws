# Med-Sentinel 360 -- Architecture

## Overview

Med-Sentinel 360 is a medical-grade digital twin built on NVIDIA Isaac Sim.
It simulates a Franka Panda 7-DOF robot arm in a hospital operating room
environment, with real-time telemetry streaming to a web backend.

## System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Isaac Sim                            │
│                                                             │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────────┐    │
│  │ Hospital  │   │ Franka Panda │   │ ObstacleManager  │    │
│  │ USD Stage │   │  (7-DOF)     │   │ (Medical Chaos)  │    │
│  └──────────┘   └──────┬───────┘   └──────────────────┘    │
│                        │                                    │
│              ┌─────────▼──────────┐                         │
│              │  OmniGraph Action  │                         │
│              │  Graph (ROS2 +     │                         │
│              │  Articulation Ctrl)│                         │
│              └─────────┬──────────┘                         │
│                        │                                    │
│              ┌─────────▼──────────┐                         │
│              │    SimBridge       │  100 Hz sampling         │
│              │ (Protobuf encode)  │                         │
│              └─────────┬──────────┘                         │
└────────────────────────┼────────────────────────────────────┘
                         │ Binary Protobuf
                         ▼
              ┌─────────────────────┐
              │  FastAPI WebSocket  │  ws://0.0.0.0:8765/ws
              │  Server             │
              │  ┌───────────────┐  │
              │  │ /ws  endpoint │──┼──► Bidirectional binary frames
              │  │ /health       │  │    (RobotState ↓, ControlCommand ↑)
              │  │ /stats        │  │
              │  └───────────────┘  │
              └──────────┬──────────┘
                         │
                    ┌────▼────┐
                    │ Web     │  Browser / Dashboard / CLI
                    │ Clients │
                    └─────────┘
```

## Data Flow

1. **Isaac Sim** runs the physics at ~60 Hz (configurable in `scene_params.yaml`)
2. **SimBridge** samples joint state at 100 Hz (independent of physics rate)
3. Each sample is serialized into a **Protobuf** `RobotState` message (~200 bytes)
4. The **FastAPI server** pushes serialized frames over **WebSocket** to all connected clients
5. Clients send **ControlCommand** messages back, which SimBridge applies to the robot via ArticulationController
6. **PingPong** messages measure round-trip latency

## Wire Protocol

Every WebSocket frame is binary with a 1-byte type header:

| Byte 0 | Payload        | Direction        |
|--------|----------------|------------------|
| `0x01` | RobotState     | Server -> Client |
| `0x02` | ControlCommand | Client -> Server |
| `0x03` | PingPong       | Bidirectional    |

## Key Design Decisions

- **Protobuf over JSON**: ~10x smaller payloads, ~5x faster serialization.
  Critical for hitting <20ms at 100Hz.
- **Single WebSocket channel**: Both telemetry and commands share one connection
  to avoid connection overhead and simplify client implementations.
- **Rate-limiting in SimBridge**: Decouples simulation physics rate from
  network publishing rate, preventing bandwidth spikes.
- **Quaternions everywhere**: No Euler angles. Avoids gimbal lock and matches
  Isaac Sim's internal representation.
