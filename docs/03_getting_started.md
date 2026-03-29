# Med-Sentinel 360 -- Getting Started

## Prerequisites

| Component | Required | Your System |
|-----------|----------|-------------|
| Ubuntu | 22.04 LTS | 22.04.5 LTS |
| GPU | RTX 3060+ | RTX 4060 Laptop (8GB) |
| NVIDIA Driver | >= 535 | 580.126.09 |
| CUDA Toolkit | 12.x | 12.8 |
| ROS2 | Humble | Installed |
| Isaac Sim | 4.2.0.2 | via conda `isaacsim` env |
| Python | 3.10 | 3.10.12 (system) / 3.10.20 (conda) |

## Environment Setup

### 1. Source ROS2 (already in your .bashrc)

```bash
source /opt/ros/humble/setup.bash
```

### 2. Activate Isaac Sim conda environment

```bash
conda activate isaacsim
```

### 3. Build the workspace

```bash
cd ~/sft_ws
colcon build --packages-select med_sentinel --symlink-install
source install/setup.bash
```

## Running the System

### Step 1: Launch the scene (Isaac Sim)

This opens Isaac Sim, loads the hospital, and spawns the Franka Panda:

```bash
conda activate isaacsim
cd ~/sft_ws
python src/med_sentinel/med_sentinel/scene_builder.py
```

For headless mode (no GUI):

```bash
python src/med_sentinel/med_sentinel/scene_builder.py --headless
```

### Step 2: Launch with robot control + bridge

Full pipeline: scene + OmniGraph control + WebSocket bridge:

```bash
python src/med_sentinel/med_sentinel/robot_controller.py
```

### Step 3: Start the WebSocket server (standalone)

If you want to run the bridge server separately:

```bash
uvicorn med_sentinel.bridge.server:app --host 0.0.0.0 --port 8765
```

Check health:

```bash
curl http://localhost:8765/health
```

### Step 4: Run the benchmark

Verify <20ms latency target:

```bash
python -m med_sentinel.bridge.benchmark --url ws://localhost:8765/ws --count 1000
```

### Step 5: Run the stress test

Test with multiple concurrent clients:

```bash
python -m med_sentinel.bridge.stress_test --clients 5 --duration 30 --cmd-hz 100
```

## Configuration

All scene parameters live in `src/med_sentinel/config/scene_params.yaml`:

| Section | What it controls |
|---------|-----------------|
| `scene` | Physics/render rate, gravity |
| `hospital` | Hospital USD path, position, orientation |
| `robot` | Robot type, USD path, spawn pose, default joint angles |
| `obstacles` | Spawn radius, min spacing, asset catalog |
| `omnigraph` | Action graph path, ROS2 topic names |

## Spawning Obstacles

From a Python script or REPL inside Isaac Sim:

```python
from med_sentinel.obstacle_manager import ObstacleManager

manager = ObstacleManager(
    world=scene.world,
    stage=scene.stage,
    nucleus_root=scene._nucleus_root,
    robot_prim_path="/World/Franka",
    config=config,
)

# Spawn 5 random obstacles
manager.randomize(count=5, seed=42)

# Spawn a specific obstacle
manager.spawn_obstacle("medical_cart")

# Clear all
manager.clear()
```

## Protobuf Schema

The `.proto` file is at `src/med_sentinel/proto/med_sentinel.proto`.

To regenerate Python bindings after editing the schema:

```bash
cd src/med_sentinel
protoc --python_out=med_sentinel/bridge --proto_path=proto proto/med_sentinel.proto
```

## Troubleshooting

### Isaac Sim won't start
- Ensure the `isaacsim` conda env is active
- Check GPU: `nvidia-smi` should show your RTX 4060
- Try headless mode first to rule out display issues

### Nucleus assets not loading
- Hospital/robot USD paths in `scene_params.yaml` may need adjusting
  based on your Nucleus server setup
- Check available assets: `omniverse://localhost/NVIDIA/Assets/Isaac/`

### WebSocket connection refused
- Ensure the server is running on port 8765
- Check firewall: `sudo ufw allow 8765`

### High latency in benchmark
- Close other GPU-intensive applications
- Reduce physics rate in `scene_params.yaml`
- Run the server and sim on the same machine (avoids network latency)
