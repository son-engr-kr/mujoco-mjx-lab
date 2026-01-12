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
    
    # Create MJX data / Initial State
    mjx_data = mjx.make_data(mjx_model)
    mjx_data = mjx_data.replace(qpos=jnp.array(q0))
    mjx_data = mjx.forward(mjx_model, mjx_data)
    
    # Generic state handling
    # If custom env logic is needed, the user should have passed a wrapped loop or similar.
    # But to keep API simple, we'll allow an optional `init_state_fn` and `custom_step_fn`.
    # For now, let's just make `step_fn` configurable via argument, but standard rendering assumes (params, obs) interface.
    
    # If we are given a `step_fn` arg (not policy_fn), use it.
    # But we don't have that arg.
    # Let's add `env_step_fn` and `env_init_fn` to args.
    pass

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
    camera_name: str = "side_view",
    # New optional args for custom envs
    env_step_fn: Callable = None, # (state, action) -> (next_state, obs, r, d)
    env_init_fn: Callable = None, # (key) -> (state, obs)
    rng_key: jax.Array = None
):
    """
    Render a policy rollout to video.
    Supports both standard MJX step and custom Env step.
    """
    if q0 is None:
        q0 = np.copy(mj_model.qpos0).astype(np.float32)
    
    # Default State Initialization
    if env_init_fn is None:
        # Standard MJX
        mjx_data = mjx.make_data(mjx_model)
        mjx_data = mjx_data.replace(qpos=jnp.array(q0))
        mjx_data = mjx.forward(mjx_model, mjx_data)
        init_state = mjx_data
        
        # Initial Obs
        init_obs = jnp.concatenate([mjx_data.qpos, mjx_data.qvel], axis=0).astype(jnp.float32)
    else:
        # Custom Env
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        init_state, init_obs = env_init_fn(rng_key)
        
        # If State is tuple (d, aux), access d for resetting visual?
        # We need access to `qpos` for rendering.
        # We assume state[0] is mjx.Data if tuple.
        pass

    # Simulation Scan Loop
    @jax.jit
    def scan_step(state_and_obs, _):
        state, obs = state_and_obs
        
        # Policy
        action = policy_fn(params, obs)
        action = jnp.clip(action, -1.0, 1.0)
        
        if env_step_fn is None:
            # Standard MJX Step
            data = state
            data = data.replace(ctrl=action)
            next_state = mjx.step(mjx_model, data)
            next_obs = jnp.concatenate([next_state.qpos, next_state.qvel], axis=0).astype(jnp.float32)
            
            # Simple Auto-Reset (Height Check)
            fallen = next_state.qpos[2] < 1.0
            def merge(current, reset): return jnp.where(fallen, reset, current)
            next_state = jax.tree_util.tree_map(merge, next_state, init_state) # Reset to initial
            
            # Re-get obs after reset? Usually just take next_obs.
            # But for visual continuity, we might want to see the fall?
            # Standard: just return next_state.
        else:
            # Custom Env Step
            next_state, next_obs, _, done = env_step_fn(state, action)
            
            # Auto-Reset Logic for Custom Env
            # We need to switch next_state to init_state (resampled) if done.
            # However, init_state passed in might be static or needs resampling?
            # env_init_fn is available? Yes, but inside JIT scan we should probably
            # just reuse `init_state` captured at start, OR call env_init_fn with a new key?
            # Generating a new key inside scan is tricky unless we carry rng.
            # For visualization, resetting to the SAME initial state is usually fine and stable.
            # Let's use `init_state` (and `init_obs`) which we computed before scan.
            
            # done is likely a scalar or (1,) array.
            d_shape = (1,)
            is_done = done.reshape(d_shape)[0] > 0.5 # Convert to bool scalar if possible or use where
            
            # We need to tree_map merge
            def merge_reset(x_next, x_init):
                 # Expand is_done to match x_next shape?? 
                 # Usually jnp.where broadcasts the condition.
                 return jnp.where(is_done, x_init, x_next)
            
            next_state = jax.tree_util.tree_map(merge_reset, next_state, init_state)
            next_obs = jax.tree_util.tree_map(merge_reset, next_obs, init_obs)
            
        # Extract QPOS for rendering
        if isinstance(next_state, tuple):
             # Assume (mjx_data, ...)
             qpos = next_state[0].qpos
        else:
             qpos = next_state.qpos
             
        return (next_state, next_obs), qpos

    dt = mj_model.opt.timestep
    total_steps = int(duration / dt)
    
    print(f"Simulating {total_steps} steps ({duration}s)...")
    
    _, qpos_history = jax.lax.scan(scan_step, (init_state, init_obs), None, length=total_steps)
    
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
