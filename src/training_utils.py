"""
Common training utilities.
"""
import os
import time
import jax
import jax.numpy as jnp
import numpy as np
import mujoco
import mujoco.mjx as mjx
from typing import Dict, Tuple, Callable, Any, NamedTuple

from .envs import create_env_functions
from .rendering import render_policy_rollout
from .checkpoint_utils import save_checkpoint, save_metrics_log


# ==================== Normalization Utilities ====================

class RMSState(NamedTuple):
    """Running Mean and Variance State."""
    mean: jax.Array
    var: jax.Array
    count: jax.Array

    @classmethod
    def create(cls, feature_dim: int):
        return cls(
            mean=jnp.zeros(feature_dim, dtype=jnp.float32),
            var=jnp.ones(feature_dim, dtype=jnp.float32),
            count=jnp.array(1e-4, dtype=jnp.float32)
        )

def update_rms(state: RMSState, x: jax.Array) -> RMSState:
    """Update running statistics with new batch of data."""
    batch_mean = jnp.mean(x, axis=0)
    batch_var = jnp.var(x, axis=0)
    batch_count = x.shape[0]

    delta = batch_mean - state.mean
    tot_count = state.count + batch_count

    new_mean = state.mean + delta * batch_count / tot_count
    m_a = state.var * state.count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
    new_var = M2 / tot_count
    
    # Clip variance to prevent numerical issues
    new_var = jnp.maximum(new_var, 1e-4)
    
    return RMSState(mean=new_mean, var=new_var, count=tot_count)

def normalize_obs(state: RMSState, x: jax.Array) -> jax.Array:
    """Normalize observations using running statistics."""
    return (x - state.mean) / jnp.sqrt(state.var + 1e-8)


# ==================== Setup Functions ====================

def setup_jax_environment():
    """Configure JAX environment variables for optimal GPU usage."""
    os.environ["JAX_PLATFORMS"] = os.environ.get("JAX_PLATFORMS", "cuda")
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    # Recommended settings for MJX
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"


def load_model_and_create_env(
    xml_path: str,
    env_config: Any,
    lighten_solver: bool = False,
    solver_options: Dict[str, Any] = None
) -> Tuple[mujoco.MjModel, mjx.Model, np.ndarray, int, int, int, Callable, Callable, Callable, Callable]:
    """
    Load MuJoCo model and create environment functions.
    """
    print(f"Loading model: {xml_path}")
    m = mujoco.MjModel.from_xml_path(xml_path)
    
    # Resolve Body IDs dynamically

    env_config.pelvis_body_id = m.body("pelvis").id
    env_config.head_body_id = m.body("head").id
    
    print(f"Resolved Body IDs: Pelvis={env_config.pelvis_body_id}, Head={env_config.head_body_id}")

    if lighten_solver:
        m.opt.iterations = 1
        m.opt.ls_iterations = 1
        print("Lightened solver for throughput")
        
    if solver_options:
        print(f"Applying custom solver options: {solver_options}")
        for k, v in solver_options.items():
            setattr(m.opt, k, v)
    
    sys = mjx.put_model(m)
    nq, nv, nu = int(m.nq), int(m.nv), int(m.nu)
    q0 = np.copy(m.qpos0).astype(np.float32)
    
    print(f"Model dims: nq={nq}, nv={nv}, nu={nu}")
    
    # Create environment functions using the new envs module
    single_reset, single_step, v_reset, v_step = create_env_functions(sys, env_config, q0, nq, nv)
    
    return m, sys, q0, nq, nv, nu, single_reset, single_step, v_reset, v_step


def print_training_header(method_name: str, config: Any):
    """Print training header with configuration."""
    print("=" * 60)
    print(f"MJX Humanoid {method_name} Training")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}")


def print_config_summary(config: Dict):
    """Print configuration summary."""
    print("\n" + "=" * 60)
    print("Training Configuration:")
    for key, value in config.items():
        if isinstance(value, (int, float, str)):
            print(f"  {key}: {value}")
    print("=" * 60)


# ==================== Training Loop Utilities ====================

class TrainingLogger:
    """Helper class for training logging and checkpointing."""
    
    def __init__(self, training_dir: str, config):
        self.training_dir = training_dir
        self.config = config
        self.total_env_steps = 0
        self.start_time = time.time()
    
    def log_step(self, step: int, metrics: Dict, print_msg: str):
        """Log metrics for a single training step."""
        metrics["total_env_steps"] = self.total_env_steps
        metrics["elapsed_time"] = time.time() - self.start_time
        
        print(print_msg)
        save_metrics_log(self.training_dir, step, metrics)
    
    def should_checkpoint(self, step: int) -> bool:
        """Check if should save checkpoint at this step."""
        # Handle both dataclass and dict config
        checkpoint_every = getattr(self.config, "checkpoint_every", 50)
        total_steps = getattr(self.config, "total_steps", 1000)
        
        return (step + 1) % checkpoint_every == 0 or step == total_steps - 1
    
    def should_render(self, step: int) -> bool:
        """Check if should render video at this step."""
        render_every = getattr(self.config, "render_every", 100)
        total_steps = getattr(self.config, "total_steps", 1000)
        
        return (step + 1) % render_every == 0 or step == total_steps - 1
    
    def save_checkpoint(self, step: int, params, opt_state, metrics: Dict):
        """Save a training checkpoint."""
        save_checkpoint(self.training_dir, step + 1, params, opt_state, metrics)
    
    def render_video(
        self,
        step: int,
        m: mujoco.MjModel,
        sys: mjx.Model,
        params,
        policy_fn: Callable,
        q0: np.ndarray,
        filename_prefix: str = "step",
        env_step_fn=None,
        env_init_fn=None
    ):
        """Render a video of the current policy."""
        # Handle both dataclass and dict config
        duration = getattr(self.config, "render_duration", 5.0)
        fps = getattr(self.config, "render_fps", 60)
        camera_name = getattr(self.config, "camera_name", "side_view") # Default to side_view (tracking)

        video_path = os.path.join(
            self.training_dir,
            "videos",
            f"{filename_prefix}_{step+1:06d}.mp4"
        )
        print(f"  Rendering video to {video_path}...")
        
        render_policy_rollout(
            m, sys, params, policy_fn,
            video_path,
            duration=duration,
            fps=fps,
            q0=q0,
            camera_name=camera_name,
            env_step_fn=env_step_fn,
            env_init_fn=env_init_fn
        )
    
    def print_final_summary(self):
        """Print final training summary."""
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Total time: {time.time() - self.start_time:.1f}s")
        print(f"Total env steps: {int(self.total_env_steps)}")
        print(f"Results saved to: {self.training_dir}")
        print("=" * 60)
