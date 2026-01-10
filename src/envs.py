"""
Environment definitions and factories for Mujoco MJX.
"""
import jax
import jax.numpy as jnp
from jax import jit, random
import mujoco
import mujoco.mjx as mjx
import numpy as np
from typing import Tuple, Callable, Any

# Type alias for ease of use
EnvState = mjx.Data
Obs = jax.Array
Action = jax.Array
Reward = jax.Array
Done = jax.Array
EnvStepFn = Callable[[EnvState, Action], Tuple[EnvState, Obs, Reward, Done]]
EnvResetFn = Callable[[jax.Array], Tuple[EnvState, Obs]]

def create_env_functions(sys: mjx.Model, q0: jnp.ndarray, nq: int, nv: int) -> Tuple[EnvResetFn, EnvStepFn, EnvResetFn, EnvStepFn]:
    """
    Creates JIT-compiled environment functions (reset, step) for the given system.
    
    Args:
        sys: The MJX system model.
        q0: Initial qpos.
        nq: Number of generalize coordinates.
        nv: Number of generalized velocities.
        
    Returns:
        single_reset, single_step, v_reset, v_step
    """
    
    @jit
    def single_pipeline_init(q: jax.Array, qd: jax.Array) -> EnvState:
        d = mjx.make_data(sys)
        d = d.replace(qpos=q, qvel=qd)
        d = mjx.forward(sys, d)
        return d

    @jit
    def single_reset(key: jax.Array) -> Tuple[EnvState, Obs]:
        k1, k2 = random.split(key)
        qpos = jnp.array(q0)
        qvel = jnp.zeros(nv, dtype=jnp.float32)
        
        # Simple randomization (reduced noise)
        # 0.01 is enough variation without causing explosions
        qpos = qpos.at[0:3].add(0.01 * (random.uniform(k1, (3,)) - 0.5))
        qvel = qvel.at[:].add(0.01 * (random.uniform(k2, (nv,)) - 0.5))
        
        # Initialize pipeline without modifying qpos based on contacts
        d = single_pipeline_init(qpos, qvel)
        
        obs = jnp.concatenate([d.qpos.astype(jnp.float32), d.qvel.astype(jnp.float32)], axis=0)
        return d, obs

    @jit
    def single_step(d: EnvState, action: Action) -> Tuple[EnvState, Obs, Reward, Done]:
        act = jnp.clip(action, -1.0, 1.0)
        d = d.replace(ctrl=act)
        d = mjx.step(sys, d)
        
        x = d.qpos
        v = d.qvel
        obs = jnp.concatenate([x.astype(jnp.float32), v.astype(jnp.float32)], axis=0)

        # Reward calculation (Standard Gym Humanoid-v4 Style)
        # 1. Healthy Reward (Alive Bonus) - Encourages staying up
        healthy_reward = 5.0
        
        # 2. Forward Progress
        # Gym uses 1.25 * velocity usually
        forward_reward = 1.25 * v[0].astype(jnp.float32)
        
        # 3. Control Cost (Energy Efficiency)
        ctrl_cost = 0.1 * jnp.sum(jnp.square(act))
        
        # 4. Small penalty for large velocities to output stability
        # Contact cost is ignored for simplicity
        
        reward = healthy_reward + forward_reward - ctrl_cost
        
        # Termination condition: Pelvis height < 1.0 (Standard Gym Threshold)
        # If z < 1.0, consider fallen.
        fallen = x[2] < 1.0
        done = jnp.where(fallen, 1.0, 0.0)
        
        # If fallen, remove healthy reward for this step (equivalent to small penalty)
        reward = jnp.where(fallen, reward - healthy_reward, reward)
        
        return d, obs, reward, done

    v_reset = jit(jax.vmap(single_reset))
    v_step = jit(jax.vmap(single_step, in_axes=(0, 0)))
    
    return single_reset, single_step, v_reset, v_step
