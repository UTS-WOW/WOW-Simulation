# Project Structure

This document provides a short overview of the main folders and files in the project.

## Root

```text
WOW-Simulation/
├── Assets/
├── NavalTrainerServer/
├── Packages/
├── ProjectSettings/
├── Training/
├── ENVIRONMENT.md
├── README.md
├── RL_README.md
└── SHIPS.md
```

### `Assets/`
Contains the Unity naval simulation source code and game assets.

- `Editor/` — Unity editor scripts, including the Linux training server build script.
- `Scenes/` — Unity game scenes.
- `Screenshots/` — screenshots generated from the simulation.
- `Scripts/` — main C# simulation and RL integration code.
- `Settings/` — Unity project/rendering settings.
- `Shaders/` — graphical shaders used by the simulator.
- `StreamingAssets/` — runtime files, including the exported RL policy.

### `NavalTrainerServer/`
Compiled Linux Dedicated Server version of the Unity simulator used for headless RL training on SageMaker.

### `Packages/`
Unity package configuration and dependencies.

### `ProjectSettings/`
Unity project configuration.

---

## Training

```text
Training/
├── configs/
├── naval_rl/
├── runs/
├── scenarios/
├── tests/
├── train.py
├── evaluate.py
├── bc.py
└── requirements.txt
```

### `naval_rl/`
Core Python reinforcement-learning implementation, including MAPPO, environment communication, neural-network models, curriculum and self-play logic.

### `runs/`
Generated training results. Each run can contain:

- `metrics.jsonl` — update-level training metrics.
- `episodes.jsonl` — individual battle results.
- `checkpoints/` — saved model checkpoints.
- `unity_logs/` — logs from Unity training environments.
- `tb/` — TensorBoard logs.
- `config.json` — training configuration.
- `spec.json` — observation and action specification.

### `scenarios/`
Battle scenarios used by the curriculum and evaluation.

### `configs/`
Additional experiment and curriculum configurations.

### `tests/`
Tests for the RL implementation.

### `train.py`
Main MAPPO training entry point.

### `evaluate.py`
Evaluates trained policies.

### `bc.py`
Behaviour Cloning training script.

### `requirements.txt`
Python dependencies required for training.

---

## Documentation

### `README.md`
General documentation for the naval simulator.

### `RL_README.md`
Detailed documentation of the MARL system, including observations, actions, rewards, MAPPO, CTDE, curriculum and self-play.

### `ENVIRONMENT.md`
Documentation of the naval simulation environment, maps and gameplay mechanics.

### `SHIPS.md`
Ship statistics, simulation-unit conversions and ship-specific assumptions.

### To Start training
cd ~/WOW-Simulation/Training

python3 train.py \
  --unity-binary ../NavalTrainerServer/NavalTrainerServer.x86_64
