"""
Rendering utilities for visualizing MJX policies.
"""
import os
import mujoco
import mujoco.mjx as mjx
import jax
import jax.numpy as jnp
import numpy as np
import imageio
from typing import Callable, Any

def render_policy_rollout(
    mj_model: mujoco.MjModel,
    mjx_model: mjx.Model,
    params: Any,
    policy_fn: Callable[[Any, jax.Array], jax.Array],
    output_path: str,
    duration: float = 5.0,
    fps: int = 60,
    q0: np.ndarray = None,
    width: int = 640,
    height: int = 480,
    camera_name: str = "side_view" 
):
    """
    Render a policy rollout to video.
    
    Args:
        mj_model: MuJoCo CPU model for rendering.
        mjx_model: MJX GPU model for simulation.
        params: Policy parameters (Flax FrozenDict or other).
        policy_fn: Function mapping (params, obs) -> action.
        output_path: Path to save the video.
        duration: Duration of simulation in seconds.
        fps: Frames per second.
        q0: Initial joint positions.
    """
    if q0 is None:
        q0 = np.copy(mj_model.qpos0).astype(np.float32)
    
    # Create MJX data
    mjx_data = mjx.make_data(mjx_model)
    mjx_data = mjx_data.replace(qpos=jnp.array(q0))
    mjx_data = mjx.forward(mjx_model, mjx_data)
    
    reset_data = mjx_data # Capture initial state for reset
    
    # Simulation step with policy (JIT compiled)
    @jax.jit
    def step_fn(data):
        obs = jnp.concatenate([data.qpos, data.qvel], axis=0).astype(jnp.float32)
        action = policy_fn(params, obs)
        action = jnp.clip(action, -1.0, 1.0)
        data = data.replace(ctrl=action)
        next_data = mjx.step(mjx_model, data)
        
        # Auto-Reset for visualization (Humanoid specific)
        # If height < 1.0, reset to initial state
        fallen = next_data.qpos[2] < 1.0
        
        def merge_if_reset(x_current, x_reset):
            # fallen is scalar bool
            return jnp.where(fallen, x_reset, x_current)
            
        next_data = jax.tree_util.tree_map(merge_if_reset, next_data, reset_data)
        return next_data
    
    # Run simulation scan
    dt = mj_model.opt.timestep
    total_steps = int(duration / dt)
    
    def loop_body(data, _):
        next_data = step_fn(data)
        return next_data, next_data.qpos
    
    print(f"Simulating {total_steps} steps ({duration}s)...")
    _, qpos_history = jax.lax.scan(loop_body, mjx_data, None, length=total_steps)
    
    # Convert to numpy for rendering on CPU
    qpos_history = np.array(qpos_history)
    
    # Render frames
    print("Generating frames...")
    frames = []
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    mj_data = mujoco.MjData(mj_model)
    
    step_stride = max(1, int(1.0 / (fps * dt)))
    
    for i in range(0, total_steps, step_stride):
        mj_data.qpos[:] = qpos_history[i]
        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data, camera=camera_name)
        frames.append(renderer.render())
    
    # Save video
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Video saved to {output_path}")
