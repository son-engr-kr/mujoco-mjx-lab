"""
Quick test script to verify the refactored training setup works.
Tests new Pure JAX networks and config system.
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.training_utils import setup_jax_environment, load_model_and_create_env
setup_jax_environment()

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

from src.config import APGConfig
from src.networks import APGPolicy
from src.checkpoint_utils import create_training_dir, save_config, save_checkpoint
from src.rendering import render_policy_rollout

def test_new_architecture():
    """Test the refactored architecture."""
    print("=" * 60)
    print("Testing Refactored Architecture (Pure JAX + Config)")
    print("=" * 60)
    
    # 1. Config
    print("\n1. Testing Config...")
    cfg = APGConfig(batch_size=4, horizon=10, total_steps=2)
    print(f"   ✓ Config created: {cfg}")
    
    # 2. Environment
    print("\n2. Loading Model & Environment...")
    try:
        m, sys_mjx, q0, nq, nv, nu, s_reset, s_step, v_reset, v_step = load_model_and_create_env(
            cfg.xml_path, cfg.lighten_solver
        )
        print("   ✓ Environment loaded")
    except Exception as e:
        print(f"   ! Environment load failed (check if model exists): {e}")
        return

    # 3. Networks (Pure JAX)
    print("\n3. Testing Pure JAX Network...")
    obs_dim = nq + nv
    act_dim = nu
    
    rng = random.PRNGKey(0)
    model = APGPolicy(action_dim=act_dim, hidden_dim=32)
    
    # Init with shape, not dummy input
    params = model.init(rng, obs_dim)
    print("   ✓ Network initialized")
    
    # 4. Checkpoint
    print("\n4. Testing Checkpoint...")
    test_dir = create_training_dir("test_refactor")
    save_config(test_dir, cfg)
    save_checkpoint(test_dir, 0, params, None, {"test": 1})
    print("   ✓ Checkpoint saved")
    
    # 5. Rendering
    print("\n5. Testing Rendering...")
    video_path = os.path.join(test_dir, "videos", "test.mp4")
    
    def policy_fn(p, x):
        return model.apply(p, x)
        
    render_policy_rollout(
        m, sys_mjx, params, policy_fn,
        video_path,
        duration=0.5,
        fps=30,
        q0=q0
    )
    print("   ✓ Rendering complete")
    
    print("\n" + "=" * 60)
    print("Refactoring Verification Complete! ✓")
    print("=" * 60)

if __name__ == "__main__":
    test_new_architecture()
