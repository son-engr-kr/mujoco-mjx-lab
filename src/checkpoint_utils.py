"""
Checkpoint utilities for saving/loading training progress.
"""
import os
import pickle
import json
from dataclasses import is_dataclass, asdict
from datetime import datetime
from typing import Any, Dict, Optional, Union
import jax.numpy as jnp


def create_training_dir(method_name: str, base_dir: str = "results") -> str:
    """
    Create a timestamped directory for training results.
    
    Args:
        method_name: Name of training method (e.g., 'apg', 'ppo')
        base_dir: Base directory for all results
    
    Returns:
        Path to the created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{timestamp}_{method_name}"
    full_path = os.path.join(base_dir, dir_name)
    os.makedirs(full_path, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(full_path, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(full_path, "videos"), exist_ok=True)
    os.makedirs(os.path.join(full_path, "logs"), exist_ok=True)
    
    print(f"Created training directory: {full_path}")
    return full_path


def save_checkpoint(
    save_dir: str,
    step: int,
    params: Any,
    opt_state: Optional[Any] = None,
    metrics: Optional[Dict] = None,
):
    """
    Save a training checkpoint.
    """
    checkpoint_path = os.path.join(save_dir, "checkpoints", f"checkpoint_{step:06d}.pkl")
    
    checkpoint_data = {
        "step": step,
        "params": params,
        "opt_state": opt_state,
        "metrics": metrics or {},
    }
    
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint_data, f)
    
    print(f"Saved checkpoint: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path: str) -> Dict:
    """Load a checkpoint from file."""
    with open(checkpoint_path, "rb") as f:
        return pickle.load(f)


def save_config(save_dir: str, config: Union[Dict, Any]):
    """Save training configuration as JSON."""
    config_path = os.path.join(save_dir, "config.json")
    
    if is_dataclass(config):
        config_dict = asdict(config)
    else:
        config_dict = config
    
    # Convert non-serializable types to strings
    serializable_config = {}
    for k, v in config_dict.items():
        if isinstance(v, (int, float, str, bool, list, dict, type(None))):
            serializable_config[k] = v
        else:
            serializable_config[k] = str(v)
    
    with open(config_path, "w") as f:
        json.dump(serializable_config, f, indent=2)
    
    print(f"Saved config: {config_path}")


def save_metrics_log(save_dir: str, step: int, metrics: Dict):
    """Append metrics to a log file."""
    log_path = os.path.join(save_dir, "logs", "metrics.jsonl")
    
    log_entry = {"step": step, **metrics}
    
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
