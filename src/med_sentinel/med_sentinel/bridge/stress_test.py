"""Stress test for Med-Sentinel 360 bridge.

Simulates multiple concurrent WebSocket clients sending high-frequency
ControlCommands while receiving RobotState telemetry, measuring how
the bridge performs under load.

Usage:
    python -m med_sentinel.bridge.stress_test \
        --url ws://localhost:8765/ws \
        --clients 5 \
        --duration 30 \
        --cmd-hz 100
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import List

import websockets

from med_sentinel.bridge import med_sentinel_pb2 as pb
from med_sentinel.bridge.proto_handler import build_control_command, make_timestamp
from med_sentinel.bridge.server import MessageType


@dataclass
class ClientStats:
    """Per-client metrics collected during stress test."""

    client_id: int
    frames_received: int = 0
    commands_sent: int = 0
    recv_latencies_ms: List[float] = field(default_factory=list)
    errors: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration_s(self) -> float:
        return self.end_time - self.start_time

    @property
    def recv_hz(self) -> float:
        if self.duration_s <= 0:
            return 0
        return self.frames_received / self.duration_s

    @property
    def avg_recv_latency_ms(self) -> float:
        if not self.recv_latencies_ms:
            return 0
        return statistics.mean(self.recv_latencies_ms)


async def stress_client(
    client_id: int,
    url: str,
    duration_s: float,
    cmd_hz: float,
) -> ClientStats:
    """Run a single stress client that sends commands and receives telemetry."""
    stats = ClientStats(client_id=client_id)
    cmd_interval = 1.0 / cmd_hz if cmd_hz > 0 else float("inf")

    try:
        async with websockets.connect(url) as ws:
            stats.start_time = time.perf_counter()
            deadline = stats.start_time + duration_s

            recv_task = asyncio.create_task(_recv_loop(ws, stats, deadline))
            send_task = asyncio.create_task(_send_loop(ws, stats, deadline, cmd_interval))

            await asyncio.gather(recv_task, send_task)
            stats.end_time = time.perf_counter()

    except Exception as exc:
        stats.errors += 1
        stats.end_time = time.perf_counter()
        print(f"  [Client {client_id}] Error: {exc}")

    return stats


async def _recv_loop(ws, stats: ClientStats, deadline: float):
    """Receive telemetry frames until deadline."""
    while time.perf_counter() < deadline:
        try:
            data = await asyncio.wait_for(ws.recv(), timeout=0.5)
            t_recv = time.perf_counter()
            stats.frames_received += 1

            if len(data) > 1 and data[0] == MessageType.ROBOT_STATE:
                state = pb.RobotState()
                state.ParseFromString(data[1:])
                send_time = state.stamp.seconds + state.stamp.nanos * 1e-9
                if send_time > 0:
                    latency_ms = (t_recv - send_time) * 1000.0
                    if 0 < latency_ms < 5000:
                        stats.recv_latencies_ms.append(latency_ms)

        except asyncio.TimeoutError:
            continue
        except Exception:
            stats.errors += 1
            break


async def _send_loop(ws, stats: ClientStats, deadline: float, interval: float):
    """Send ControlCommand messages at the specified rate."""
    import math
    seq = 0
    while time.perf_counter() < deadline:
        t = time.perf_counter()
        wave = math.sin(t * 0.5)
        targets = [wave * 0.3] * 7 + [0.04, 0.04]

        cmd = build_control_command(
            sequence=seq,
            joint_targets=targets,
            max_velocity=0.5,
            max_accel=0.5,
        )
        frame = bytes([MessageType.CONTROL_CMD]) + cmd.SerializeToString()

        try:
            await ws.send(frame)
            stats.commands_sent += 1
            seq += 1
        except Exception:
            stats.errors += 1
            break

        await asyncio.sleep(interval)


def print_stress_report(all_stats: List[ClientStats]):
    """Print aggregated stress test results."""
    total_frames = sum(s.frames_received for s in all_stats)
    total_cmds = sum(s.commands_sent for s in all_stats)
    total_errors = sum(s.errors for s in all_stats)

    all_latencies = []
    for s in all_stats:
        all_latencies.extend(s.recv_latencies_ms)

    print(f"\n{'=' * 60}")
    print(f"  STRESS TEST RESULTS ({len(all_stats)} clients)")
    print(f"{'=' * 60}")
    print(f"  {'Total frames received':.<40} {total_frames}")
    print(f"  {'Total commands sent':.<40} {total_cmds}")
    print(f"  {'Total errors':.<40} {total_errors}")

    if all_latencies:
        sorted_lat = sorted(all_latencies)
        p95_idx = min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)
        p99_idx = min(int(len(sorted_lat) * 0.99), len(sorted_lat) - 1)
        print(f"  {'Avg latency (ms)':.<40} {statistics.mean(all_latencies):.3f}")
        print(f"  {'Median latency (ms)':.<40} {statistics.median(all_latencies):.3f}")
        print(f"  {'P95 latency (ms)':.<40} {sorted_lat[p95_idx]:.3f}")
        print(f"  {'P99 latency (ms)':.<40} {sorted_lat[p99_idx]:.3f}")
        print(f"  {'Max latency (ms)':.<40} {max(all_latencies):.3f}")

    print()
    for s in all_stats:
        print(f"  Client {s.client_id}: "
              f"recv={s.frames_received} ({s.recv_hz:.1f} Hz), "
              f"sent={s.commands_sent}, "
              f"errors={s.errors}, "
              f"avg_lat={s.avg_recv_latency_ms:.3f}ms")

    print(f"{'=' * 60}\n")


async def main_async(args):
    print(f"[StressTest] Starting {args.clients} clients for {args.duration}s "
          f"at {args.cmd_hz} Hz command rate")
    print(f"[StressTest] Target: {args.url}")

    tasks = [
        stress_client(i, args.url, args.duration, args.cmd_hz)
        for i in range(args.clients)
    ]
    all_stats = await asyncio.gather(*tasks)
    print_stress_report(list(all_stats))


def main():
    parser = argparse.ArgumentParser(description="Med-Sentinel 360 bridge stress test")
    parser.add_argument("--url", default="ws://localhost:8765/ws")
    parser.add_argument("--clients", type=int, default=5, help="Number of concurrent clients")
    parser.add_argument("--duration", type=float, default=30.0, help="Test duration (seconds)")
    parser.add_argument("--cmd-hz", type=float, default=100.0, help="Command send rate per client")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
