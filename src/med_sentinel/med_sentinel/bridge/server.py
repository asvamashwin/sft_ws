"""FastAPI WebSocket server for Med-Sentinel 360.

Provides a bidirectional binary WebSocket channel:
  - Server -> Client: RobotState protobuf at up to 100 Hz
  - Client -> Server: ControlCommand protobuf
  - Ping/Pong for latency measurement

Run standalone:
    uvicorn med_sentinel.bridge.server:app --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from enum import IntEnum
from typing import Deque, Dict, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from med_sentinel.bridge import med_sentinel_pb2 as pb
from med_sentinel.bridge.proto_handler import (
    make_timestamp,
    stamp_ping_server,
    timestamp_to_float,
)

logger = logging.getLogger("med_sentinel.bridge")

TELEMETRY_HZ = 100
LATENCY_WINDOW = 200


class MessageType(IntEnum):
    """First byte of every WebSocket binary frame identifies the payload."""
    ROBOT_STATE = 0x01
    CONTROL_CMD = 0x02
    PING_PONG   = 0x03


app = FastAPI(title="Med-Sentinel 360 Bridge", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class BridgeState:
    """Shared mutable state for the bridge server."""

    def __init__(self):
        self.clients: Set[WebSocket] = set()
        self.latest_robot_state: Optional[bytes] = None
        self.latest_command: Optional[Dict] = None
        self.command_event = asyncio.Event()
        self.latencies_us: Deque[float] = deque(maxlen=LATENCY_WINDOW)
        self.tx_count: int = 0
        self.rx_count: int = 0

    @property
    def avg_latency_ms(self) -> float:
        if not self.latencies_us:
            return 0.0
        return sum(self.latencies_us) / len(self.latencies_us) / 1000.0

    @property
    def client_count(self) -> int:
        return len(self.clients)


bridge = BridgeState()


def update_robot_state(state_bytes: bytes):
    """Called by the sim bridge node to push new telemetry into the server."""
    bridge.latest_robot_state = state_bytes
    bridge.tx_count += 1


def get_latest_command() -> Optional[Dict]:
    """Called by the sim bridge node to pull the latest control command."""
    cmd = bridge.latest_command
    bridge.latest_command = None
    return cmd


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "clients": bridge.client_count,
        "tx_count": bridge.tx_count,
        "rx_count": bridge.rx_count,
        "avg_latency_ms": round(bridge.avg_latency_ms, 3),
    }


@app.get("/stats")
async def stats():
    return {
        "telemetry_hz": TELEMETRY_HZ,
        "connected_clients": bridge.client_count,
        "total_frames_sent": bridge.tx_count,
        "total_commands_received": bridge.rx_count,
        "avg_latency_ms": round(bridge.avg_latency_ms, 3),
        "latency_samples": len(bridge.latencies_us),
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    bridge.clients.add(ws)
    logger.info("Client connected. Total: %d", bridge.client_count)

    send_task = asyncio.create_task(_telemetry_sender(ws))

    try:
        await _command_receiver(ws)
    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)
    finally:
        send_task.cancel()
        bridge.clients.discard(ws)
        logger.info("Client removed. Total: %d", bridge.client_count)


async def _telemetry_sender(ws: WebSocket):
    """Push robot state to the client at TELEMETRY_HZ."""
    interval = 1.0 / TELEMETRY_HZ
    while True:
        start = time.perf_counter()
        if bridge.latest_robot_state is not None:
            frame = bytes([MessageType.ROBOT_STATE]) + bridge.latest_robot_state
            try:
                await ws.send_bytes(frame)
            except Exception:
                break
        elapsed = time.perf_counter() - start
        sleep_time = max(0.0, interval - elapsed)
        await asyncio.sleep(sleep_time)


async def _command_receiver(ws: WebSocket):
    """Receive ControlCommand or PingPong from the client."""
    while True:
        data = await ws.receive_bytes()
        if len(data) < 2:
            continue

        msg_type = data[0]
        payload = data[1:]

        if msg_type == MessageType.CONTROL_CMD:
            bridge.rx_count += 1
            cmd = pb.ControlCommand()
            cmd.ParseFromString(payload)
            bridge.latest_command = {
                "sequence": cmd.sequence,
                "mode": cmd.mode,
                "joint_targets": list(cmd.joint_targets),
                "max_velocity": cmd.max_velocity,
                "max_accel": cmd.max_accel,
                "stop": cmd.stop,
            }
            bridge.command_event.set()

        elif msg_type == MessageType.PING_PONG:
            pong = stamp_ping_server(payload)
            recv_time = timestamp_to_float(pong.server_recv)
            send_time = timestamp_to_float(pong.client_send)
            rtt_us = (recv_time - send_time) * 1e6
            if rtt_us > 0:
                bridge.latencies_us.append(rtt_us)

            reply = bytes([MessageType.PING_PONG]) + pong.SerializeToString()
            await ws.send_bytes(reply)


def run_server(host: str = "0.0.0.0", port: int = 8765):
    """Convenience entry point to start the server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_server()
