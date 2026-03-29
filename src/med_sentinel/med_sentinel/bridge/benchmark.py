"""Latency benchmark client for Med-Sentinel 360 bridge.

Connects to the WebSocket server, sends PingPong messages, measures
round-trip time, and reports whether the <20ms latency target is met.

Usage:
    python -m med_sentinel.bridge.benchmark --url ws://localhost:8765/ws --count 1000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from typing import List

import websockets

from med_sentinel.bridge import med_sentinel_pb2 as pb
from med_sentinel.bridge.proto_handler import make_timestamp, timestamp_to_float
from med_sentinel.bridge.server import MessageType

LATENCY_TARGET_MS = 20.0


async def run_ping_benchmark(url: str, count: int) -> dict:
    """Send `count` PingPong messages and measure round-trip latencies."""
    latencies_ms: List[float] = []

    async with websockets.connect(url) as ws:
        for seq in range(count):
            ping = pb.PingPong()
            ping.client_send.CopyFrom(make_timestamp())
            ping.sequence = seq

            t_send = time.perf_counter()
            frame = bytes([MessageType.PING_PONG]) + ping.SerializeToString()
            await ws.send(frame)

            reply = await ws.recv()
            t_recv = time.perf_counter()

            rtt_ms = (t_recv - t_send) * 1000.0
            latencies_ms.append(rtt_ms)

    return _compute_stats(latencies_ms)


async def run_telemetry_benchmark(url: str, duration_s: float) -> dict:
    """Receive RobotState frames for `duration_s` seconds, measure throughput."""
    frame_times: List[float] = []
    start = time.perf_counter()

    async with websockets.connect(url) as ws:
        while (time.perf_counter() - start) < duration_s:
            try:
                data = await asyncio.wait_for(ws.recv(), timeout=1.0)
                frame_times.append(time.perf_counter())
            except asyncio.TimeoutError:
                continue

    if len(frame_times) < 2:
        return {"error": "Not enough frames received", "frame_count": len(frame_times)}

    intervals_ms = [
        (frame_times[i] - frame_times[i - 1]) * 1000.0
        for i in range(1, len(frame_times))
    ]

    elapsed = frame_times[-1] - frame_times[0]
    actual_hz = len(frame_times) / elapsed if elapsed > 0 else 0

    return {
        "duration_s": round(duration_s, 2),
        "frames_received": len(frame_times),
        "actual_hz": round(actual_hz, 1),
        "target_hz": 100,
        "hz_met": actual_hz >= 95.0,
        "avg_interval_ms": round(statistics.mean(intervals_ms), 3),
        "p99_interval_ms": round(_percentile(intervals_ms, 99), 3),
        "max_interval_ms": round(max(intervals_ms), 3),
    }


def _compute_stats(latencies_ms: List[float]) -> dict:
    """Compute latency statistics and check against target."""
    return {
        "count": len(latencies_ms),
        "target_ms": LATENCY_TARGET_MS,
        "avg_ms": round(statistics.mean(latencies_ms), 3),
        "median_ms": round(statistics.median(latencies_ms), 3),
        "p95_ms": round(_percentile(latencies_ms, 95), 3),
        "p99_ms": round(_percentile(latencies_ms, 99), 3),
        "min_ms": round(min(latencies_ms), 3),
        "max_ms": round(max(latencies_ms), 3),
        "stddev_ms": round(statistics.stdev(latencies_ms), 3) if len(latencies_ms) > 1 else 0,
        "target_met": statistics.mean(latencies_ms) < LATENCY_TARGET_MS,
        "p99_under_target": _percentile(latencies_ms, 99) < LATENCY_TARGET_MS,
    }


def _percentile(data: List[float], pct: float) -> float:
    """Compute the p-th percentile of a sorted list."""
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * pct / 100.0)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


def print_report(title: str, results: dict):
    """Pretty-print benchmark results."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    for key, value in results.items():
        label = key.replace("_", " ").title()
        if isinstance(value, bool):
            mark = "PASS" if value else "FAIL"
            print(f"  {label:.<40} {mark}")
        else:
            print(f"  {label:.<40} {value}")
    print(f"{'=' * 60}\n")


async def main_async(args):
    url = args.url

    print(f"[Benchmark] Connecting to {url} ...")

    print(f"\n[Benchmark] Running ping test ({args.count} pings) ...")
    ping_results = await run_ping_benchmark(url, args.count)
    print_report("Ping/Pong Latency", ping_results)

    print(f"[Benchmark] Running telemetry test ({args.duration}s) ...")
    telem_results = await run_telemetry_benchmark(url, args.duration)
    print_report("Telemetry Throughput", telem_results)

    passed = ping_results.get("target_met", False) and telem_results.get("hz_met", False)
    status = "ALL TARGETS MET" if passed else "TARGETS NOT MET"
    print(f"[Benchmark] Overall: {status}")
    return passed


def main():
    parser = argparse.ArgumentParser(description="Med-Sentinel 360 bridge benchmark")
    parser.add_argument("--url", default="ws://localhost:8765/ws", help="WebSocket URL")
    parser.add_argument("--count", type=int, default=1000, help="Number of ping messages")
    parser.add_argument("--duration", type=float, default=10.0, help="Telemetry test duration (seconds)")
    args = parser.parse_args()

    passed = asyncio.run(main_async(args))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
