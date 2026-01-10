# Mujoco MJX Lab - Architecture Documentation

## Overview

This project implements high-performance Reinforcement Learning (RL) training pipelines using **Google JAX** and **Mujoco MJX**. The architecture follows modern deep learning engineering practices, emphasizing type safety, modularity, and scalability.

## Core Design Principles

1.  **Type Safety**: All configurations are defined using Python `dataclasses` to prevent runtime errors and enable IDE autocompletion.
2.  **Modularity**: Neural networks are implemented using **Pure JAX** functional classes (inspired by Equinox/Haiku), ensuring compatibility across JAX versions (no hard Flax dependency).
3.  **Encapsulation**: Environment logic, rendering, and training utilities are separated into distinct modules.
4.  **Reproducibility**: Configuration and metrics are automatically logged with timestamps.

## Directory Structure

```
mujoco-mjx-lab/
├── src/
│   ├── config.py           # ✅ Type-safe configuration classes (APGConfig, PPOConfig)
│   ├── networks.py         # ✅ Pure JAX Neural Networks (MLP, APGPolicy, GaussianPolicy)
│   ├── envs.py             # ✅ JIT-compiled MJX environment logic
│   ├── rendering.py        # ✅ Rendering utilities
│   ├── training_utils.py   # ✅ Shared training loops & loggers
│   └── checkpoint_utils.py # ✅ Checkpoint IO
├── train_apg.py            # Main entry point for APG (Analytic Policy Gradient)
├── train_ppo.py            # Main entry point for PPO (Proximal Policy Optimization)
└── results/                # Output directory (auto-generated)
```

## Key Components

### 1. Configuration (`src/config.py`)
Instead of dictionaries, we use inheritance-based dataclasses:
```python
@dataclass
class APGConfig(BaseConfig):
    lr: float = 3e-4
    hidden_size: int = 64
```

### 2. Networks (`src/networks.py`)
We use **Pure JAX classes** to manage parameters and forward passes. This avoids complex dependencies.
```python
class APGPolicy:
    def init(self, rng, input_shape): ...
    def apply(self, params, x): ...
```

### 3. Environments (`src/envs.py`)
Pure functional JAX transformations. `create_env_functions` returns standard `(reset, step)` pairs compatible with `vmap` and `jit`.

### 4. Training Scripts
- **`train_apg.py`**: Deterministic policy gradient using `jax.grad`.
- **`train_ppo.py`**: Stochastic policy gradient using `ActorCritic` architecture and GAE.

## How to Extend

- **New Algorithm**: Create a new config in `config.py` and a new script `train_new_algo.py`. Reuse `training_utils`.
- **New Model**: Add XML to `models/` and update `xml_path` in config.
- **New Network Arch**: Define a new Class in `networks.py`.

## Maintainer Notes
- **No Flax/Haiku required**: Runs on minimal JAX installation.
- Checkpoints are saved using `pickle`.
- All JIT compilation happens at the start; first step might be slow.
