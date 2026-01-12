
Jax doens't support Windows
Powershell
```bash
wsl --install -d Ubuntu-22.04
```


connect to Ubuntu in vscode or cursor

check your system
```shell
nvidia-smi
```
->
```
NVIDIA-SMI 570.133.07             Driver Version: 572.83         CUDA Version: 12.8  
```

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip
```

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
```

```bash
pip install mujoco-mjx
pip install matplotlib
pip install -U "jax[cuda12]"
pip install optax
pip install warp-lang
pip install imageio[ffmpeg]
```


## Tips

- GPU watch on Linux:
    ```
    watch -n 0.5 nvidia-smi
    ```

# Project Structure

```
mujoco-mjx-lab/
├── src/                          # Common utilities
│   ├── __init__.py
│   ├── mjx_common.py            # Environment, policy, rendering functions
│   └── checkpoint_utils.py      # Checkpoint and logging utilities
├── models/                       # MuJoCo XML models
│   ├── humanoid.xml
│   └── ...
├── results/                      # Training outputs (auto-generated)
│   └── YYYYMMDD_HHMMSS_{method}/
│       ├── checkpoints/         # Model checkpoints
│       ├── videos/              # Rendered videos per checkpoint
│       ├── logs/                # Training metrics
│       └── config.json          # Training configuration
├── train_apg.py                 # APG (FoPG) training script
├── train_ppo.py                 # PPO training script
└── ...
```

# Benchmark Results

Based on `mjx_humanoid_speed_test.py`:

| Model        | Device Steps/Sec | Speedup |
|--------------|------------------|---------|
| HUMANOID     | ~6,176           | 1.0x    |
| HUMANOID_MJX | ~72,618          | 11.7x   |
| SPHERE       | ~5,957,372       | 965x    |

**HUMANOID_MJX is 11.7x faster** than regular MuJoCo humanoid!

# Execute

## Setup Environment
```bash
wsl
source .venv/bin/activate
```

## Speed Benchmark
```bash
python mjx_humanoid_speed_test.py
```

## Training

### APG (Analytic Policy Gradient)
```bash
python train_apg.py
```
- Batch size: 2048 parallel envs
- Horizon: 128 steps
- Optimized for MJX speed (~72K steps/sec)
- Saves checkpoints every 50 iterations
- Renders videos every 100 iterations

### PPO (Proximal Policy Optimization)
To train the agent using PPO with default configuration:
```bash
python train_ppo.py
```

To run a fast test/debug session (using `src/config_test.json`):
```bash
python train_ppo.py --test
```

To use a specific config file:
```bash
python train_ppo.py --config path/to/config.json
```
- Num envs: 1024
- Rollout length: 256 steps
- 4 PPO epochs per iteration
- Saves checkpoints every 20 iterations
- Renders videos every 50 iterations

## View Results

Training results are automatically saved to `results/YYYYMMDD_HHMMSS_{method}/`:
- **checkpoints/**: Model parameters at regular intervals
- **videos/**: Rendered policy rollouts showing learning progress
- **logs/metrics.jsonl**: Training metrics (returns, throughput, etc.)
- **config.json**: Full training configuration