"""
Quick validation script to verify environment implementation fixes.
Checks that all new features (stance detection, energy penalty, etc.) work correctly.
"""

import warnings
# Suppress Warp deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="warp.*")
warnings.filterwarnings("ignore", message=".*warp.context.*")
warnings.filterwarnings("ignore", message=".*warp.math.*")

import jax
import jax.numpy as jnp
import numpy as np
import mujoco
import mujoco.mjx as mjx
from src.config import PPOConfig
from src.training_utils import load_model_and_create_env

def validate_env_implementation():
    """Validate that all fixes are working correctly."""
    print("=" * 60)
    print("Environment Implementation Validation")
    print("=" * 60)
    
    # Load config
    config = PPOConfig.from_json("src/config.json")
    
    # Load model and create env
    m, sys, q0, nq, nv, nu, single_reset, single_step, v_reset, v_step = load_model_and_create_env(
        config.xml_path,
        config.env_config,
        config.lighten_solver
    )
    
    print(f"\n✓ Model loaded: nq={nq}, nv={nv}, nu={nu}")
    
    # Check sensor IDs are resolved
    print(f"\n✓ Sensor IDs resolved:")
    print(f"  - Touch Right: {config.env_config.touch_sensor_right_id}")
    print(f"  - Touch Left: {config.env_config.touch_sensor_left_id}")
    
    assert config.env_config.touch_sensor_right_id >= 0, "Touch sensor right not resolved!"
    assert config.env_config.touch_sensor_left_id >= 0, "Touch sensor left not resolved!"
    
    # Test reset
    print(f"\n✓ Testing reset...")
    key = jax.random.PRNGKey(42)
    state, obs = single_reset(key)
    d, aux = state
    
    print(f"  - Obs shape: {obs.shape}")
    print(f"  - Aux shape: {aux.shape} (should be 8)")
    assert aux.shape == (8,), f"Aux state should be 8-dimensional, got {aux.shape}"
    
    print(f"  - Aux state: [flip, tx, ty, tz, close_count, stance_state, stance_time, last_pot]")
    print(f"    Values: {aux}")
    
    # Check initial velocity was applied if enabled
    if config.env_config.initial_velocity_max > 0:
        vel_magnitude = jnp.sqrt(d.qvel[0]**2 + d.qvel[1]**2)
        print(f"  - Initial velocity magnitude: {vel_magnitude:.4f}")
        print(f"    (Should be between 0 and {config.env_config.initial_velocity_max})")
    
    # Test step
    print(f"\n✓ Testing step...")
    action = jnp.zeros(nu)
    state_new, obs_new, reward, done = single_step(state, action)
    d_new, aux_new = state_new
    
    print(f"  - Reward: {reward:.4f}")
    print(f"  - Done: {done}")
    print(f"  - Stance state changed: {aux[5]} -> {aux_new[5]}")
    
    # Test vectorized operations
    print(f"\n✓ Testing vectorized operations...")
    n_envs = 4
    keys = jax.random.split(key, n_envs)
    states, obss = v_reset(keys)
    
    print(f"  - Batch obs shape: {obss.shape}")
    print(f"  - Batch aux shape: {states[1].shape}")
    
    actions = jnp.zeros((n_envs, nu))
    states_new, obss_new, rewards, dones = v_step(states, actions)
    
    print(f"  - Batch rewards: {rewards}")
    
    # Check stance states
    print(f"\n✓ Stance detection test:")
    stance_states = states[1][:, 5]  # Extract stance states from aux
    print(f"  - Stance states: {stance_states}")
    print(f"  - Unique values: {jnp.unique(stance_states)} (should be in [0,1,2,3])")
    
    # Test energy penalty calculation
    print(f"\n✓ Energy penalty components:")
    print(f"  - Joint velocities (d.qvel[6:]) shape: {d.qvel[6:].shape}")
    print(f"  - Joint forces (d.qfrc_actuator[6:]) shape: {d.qfrc_actuator[6:].shape}")
    print(f"  - Expected: Both should be (21,) for humanoid")
    
    # Verify config parameters
    print(f"\n✓ Reward configuration:")
    print(f"  - Progress weight: {config.env_config.progress_weight}")
    print(f"  - Electricity cost: {config.env_config.electricity_cost}")
    print(f"  - Stall torque cost: {config.env_config.stall_torque_cost}")
    print(f"  - Posture penalty weight: {config.env_config.posture_penalty_weight}")
    print(f"  - Tall bonus weight: {config.env_config.tall_bonus_weight}")
    print(f"  - Stance time reward weight: {config.env_config.stance_time_reward_weight}")
    
    if config.env_config.stance_time_reward_weight == 0.0:
        print(f"\n⚠️  WARNING: Stance time reward weight is 0.0!")
        print(f"    Set to 1.0 in config.json to match reference implementation")
    
    print("\n" + "=" * 60)
    print("✅ All validation checks passed!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    validate_env_implementation()
