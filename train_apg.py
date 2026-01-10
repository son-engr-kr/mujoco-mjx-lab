"""
MJX Humanoid APG (FoPG) Training - Ultra-Light for 4GB VRAM
Optimized with:
- Minimal batch/horizon  
- Observation normalization
- Enhanced randomization
- AGGRESSIVE gradient clipping for stability
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# NOTE: MJX only supports Float32, so we use aggressive stabilization instead
import jax
jax.config.update("jax_debug_nans", True)
print("Float32 with aggressive stabilization (MJX limitation)")

# Setup JAX environment
from src.training_utils import setup_jax_environment
setup_jax_environment()

import time
import jax
import jax.numpy as jnp
from jax import jit, lax, random, jacfwd
import optax
import mujoco  # Required for solver enum

from src.training_utils import (
    load_model_and_create_env,
    TrainingLogger,
    print_training_header,
    print_config_summary,
)
from src.checkpoint_utils import create_training_dir, save_config
from src.config import APGConfig
from src.networks import APGPolicy


# ==================== Observation Normalization ====================
class RunningMeanStd:
    """Tracks running mean and std for observation normalization."""
    def __init__(self, shape):
        self.mean = jnp.zeros(shape, dtype=jnp.float32)
        self.var = jnp.ones(shape, dtype=jnp.float32)
        self.count = 1e-4
    
    def update(self, x):
        """Update running statistics."""
        batch_mean = jnp.mean(x, axis=0)
        batch_var = jnp.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = M2 / total_count
        
        self.mean = new_mean
        self.var = jnp.maximum(new_var, 1e-4)  # Prevent zero variance
        self.count = total_count
    
    def normalize(self, x):
        """Normalize observation."""
        return jnp.clip((x - self.mean) / jnp.sqrt(self.var + 1e-8), -10.0, 10.0)


def train():
    """Main APG training loop with obs normalization."""
    print("="*60)
    print("Starting APG Training")
    print("="*60)
    
    # 1. Configuration
    cfg = APGConfig()
    
    print("\n[1/7] Loading configuration...")
    print_training_header("APG (Ultra-Light 4GB)", cfg)
    
    # Create training directory
    training_dir = create_training_dir("apg")
    save_config(training_dir, cfg)
    logger = TrainingLogger(training_dir, cfg)
    print(f"  Training directory: {training_dir}")
    
    # 2. Environment
    print("\n[2/7] Loading environment...")
    
    # CRITICAL: Use CG Solver for APG stability on Humanoid
    # Newton solver (default) is numerically unstable for APG gradients
    solver_options = {
        "solver": mujoco.mjtSolver.mjSOL_CG,
        "iterations": 4,
        "ls_iterations": 4
    }
    
    try:
        m, sys, q0, nq, nv, nu, _, _, v_reset, v_step = load_model_and_create_env(
            cfg.xml_path,
            cfg.lighten_solver,
            solver_options=solver_options
        )
        print(f"  Model: {cfg.xml_path}")
        print(f"  State dim: {nq + nv}, Action dim: {nu}")
    except Exception as e:
        print(f"ERROR: Failed to load model - {e}")
        return

    # 3. Policy Network (Pure JAX)
    print("\n[3/7] Initializing policy network...")
    obs_dim = nq + nv
    act_dim = nu
    rng = random.PRNGKey(cfg.seed)
    rng, init_rng = random.split(rng)
    
    policy_model = APGPolicy(action_dim=act_dim, hidden_dim=cfg.hidden_size, hidden_depth=cfg.hidden_depth)
    params = policy_model.init(init_rng, obs_dim)
    print(f"  Network: MLP(hidden={cfg.hidden_size})")
    
    # 4. Observation Normalization
    print("\n[4/7] Setting up observation normalization...")
    if cfg.normalize_observations:
        obs_normalizer = RunningMeanStd((obs_dim,))
        print("  Observation normalization enabled")
        print("  Warm-up period: 50 steps (no normalization)")
    else:
        obs_normalizer = None
        print("  Observation normalization disabled")
    
    # 5. Optimizer with AGGRESSIVE gradient clipping
    print("\n[5/7] Initializing optimizer...")
    opt = optax.chain(
        optax.clip_by_global_norm(0.3),  # AGGRESSIVE clipping for Float32 stability
        optax.adam(cfg.lr)
    )
    opt_state = opt.init(params)
    print(f"  Optimizer: Adam (lr={cfg.lr})")
    print("  Gradient clipping: max_norm=0.3 (AGGRESSIVE)")
    
    # 6. Loss & Update
    batch_size = cfg.batch_size
    horizon = cfg.horizon
    gamma = cfg.gamma
    
    print("\n[6/7] Compiling training functions...")
    
    @jit
    def policy_fn(params, x):
        return policy_model.apply(params, x)

    @jit
    def rollout_return(params, rngs, obs_mean, obs_std, use_norm):
        """Compute discounted return with optional obs normalization."""
        d0, obs0 = v_reset(rngs)
        
        def body(carry, _):
            d, disc, acc = carry
            obs = jnp.concatenate([d.qpos.astype(jnp.float32), d.qvel.astype(jnp.float32)], axis=1)
            
            # Conditionally normalize observation
            obs_norm = jnp.where(
                use_norm,
                jnp.clip((obs - obs_mean) / (jnp.sqrt(obs_std) + 1e-8), -10.0, 10.0),
                obs
            )
            
            act = policy_model.apply(params, obs_norm)
            d_next, _, r, done = v_step(d, act)
            disc_next = disc * gamma * (1.0 - done)
            acc_next = acc + (disc * r)
            return (d_next, disc_next, acc_next), (obs, r)
        
        init_disc = jnp.ones((batch_size,), dtype=jnp.float32)
        init_acc = jnp.zeros((batch_size,), dtype=jnp.float32)
        
        # Optimized checkpointing (from Brax tutorial)
        remat_body = jax.checkpoint(body,
            policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
        (__, __, acc), (obs_traj, rewards) = lax.scan(remat_body, (d0, init_disc, init_acc), None, length=horizon)
        return jnp.mean(acc), obs_traj, jnp.mean(rewards)
    
    # Separate function for gradient computation
    def loss_and_aux_fn(params, keys, obs_mean, obs_std, use_norm):
        ret, obs_traj, mean_reward = rollout_return(params, keys, obs_mean, obs_std, use_norm)
        return -ret, (obs_traj, mean_reward)
    
    @jit
    def update(params, opt_state, keys, obs_mean, obs_std, use_norm):
        """Perform one optimization step."""
        (loss_val, (obs_traj, mean_reward)), g = jax.value_and_grad(loss_and_aux_fn, has_aux=True)(
            params, keys, obs_mean, obs_std, use_norm
        )
        updates, opt_state = opt.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        # Compute gradient norm for monitoring
        grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(g)))
        
        return params, opt_state, loss_val, obs_traj, mean_reward, grad_norm
    
    print("  Compilation complete")
    
    # Print configuration
    print_config_summary(cfg.__dict__)
    
    # Calculate realistic memory estimate
    # 1. Network parameters (3 layers: input->hidden, hidden->hidden, hidden->output)
    bytes_per_float = 4  # float32 (MJX limitation)
    network_params_bytes = (
        obs_dim * cfg.hidden_size +  # Layer 1 weights
        cfg.hidden_size +  # Layer 1 bias
        cfg.hidden_size * cfg.hidden_size +  # Layer 2 weights
        cfg.hidden_size +  # Layer 2 bias
        cfg.hidden_size * act_dim +  # Layer 3 weights
        act_dim  # Layer 3 bias
    ) * bytes_per_float
    
    # 2. Optimizer state (Adam stores momentum + variance = 2x params)
    optimizer_state_bytes = network_params_bytes * 2
    
    # 3. Rollout trajectory storage
    rollout_bytes = batch_size * horizon * (obs_dim + act_dim) * bytes_per_float
    
    # 4. Gradient/Jacobian storage (APG's main memory cost)
    gradient_overhead = 15  # APG stores full differentiation chain
    gradient_bytes = rollout_bytes * gradient_overhead
    
    # 5. JIT compilation overhead
    jit_overhead_bytes = 300 * 1024 * 1024  # ~300 MB
    
    total_bytes = (network_params_bytes + optimizer_state_bytes + 
                   rollout_bytes + gradient_bytes + jit_overhead_bytes)
    estimated_vram_gb = total_bytes / (1024**3)
    
    print(f"\nMemory-Optimized Settings (Float32):")
    print(f"  Batch x Horizon = {batch_size} x {horizon} = {batch_size * horizon} steps/update")
    print(f"  Network params: {network_params_bytes / (1024**2):.1f} MB")
    print(f"  Gradient storage: {gradient_bytes / (1024**2):.1f} MB")
    print(f"  Est. VRAM usage: ~{estimated_vram_gb:.1f} GB")
    print(f"  Total training steps: {cfg.total_steps * batch_size * horizon:,}")
    
    # 7. Training Loop
    print("\n[7/7] Starting training loop...")
    print("="*60)
    total_steps = cfg.total_steps
    warmup_steps = 100  # Extended warmup for Float32 stability
    
    for step in range(total_steps):
        iter_t0 = time.time()
        
        # Get current obs normalizer stats
        use_normalization = (step >= warmup_steps) and (obs_normalizer is not None)
        if obs_normalizer:
            obs_mean = obs_normalizer.mean
            obs_std = obs_normalizer.var
        else:
            obs_mean = jnp.zeros(obs_dim)
            obs_std = jnp.ones(obs_dim)
        
        # Generate random keys and update
        rng, sub = random.split(rng)
        keys = random.split(sub, batch_size)
        params, opt_state, loss_value, obs_traj, mean_reward, grad_norm = update(
            params, opt_state, keys, obs_mean, obs_std, use_normalization
        )
        
        # Check for NaN
        if jnp.isnan(loss_value):
            print(f"\nERROR: NaN detected at step {step}")
            print(f"  Mean reward: {mean_reward}")
            print(f"  Gradient norm: {grad_norm}")
            print(f"  Obs mean: {jnp.mean(obs_mean)}, std: {jnp.mean(jnp.sqrt(obs_std))}")
            print("\nSuggestions:")
            print("  1. Lower learning rate in config.py")
            print("  2. Increase gradient clipping threshold")
            print("  3. Check reward function implementation")
            break
        
        # Update observation statistics
        if obs_normalizer and step % 10 == 0:  # Update every 10 steps
            obs_flat = obs_traj.reshape(-1, obs_dim)
            obs_normalizer.update(obs_flat)
        
        # Metrics
        iter_dt = max(time.time() - iter_t0, 1e-9)
        env_steps = float(batch_size * horizon)
        logger.total_env_steps += env_steps
        env_steps_per_sec = env_steps / iter_dt
        
        metrics = {
            "loss": float(loss_value),
            "return": float(-loss_value),
            "mean_reward": float(mean_reward),
            "grad_norm": float(grad_norm),
            "env_steps_per_sec": env_steps_per_sec,
        }
        
        # Enhanced progress output
        if step < 10 or step % 10 == 0:
            norm_status = "ON" if use_normalization else "OFF"
            print_msg = (
                f"Step {step:5d} | "
                f"Return: {float(-loss_value):7.2f} | "
                f"Reward: {float(mean_reward):6.3f} | "
                f"GradNorm: {float(grad_norm):6.2f} | "
                f"Norm: {norm_status} | "
                f"Steps/s: {env_steps_per_sec:6.0f}"
            )
            logger.log_step(step, metrics, print_msg)
        else:
            # Just log to file without printing
            save_metrics_log = __import__('src.checkpoint_utils', fromlist=['save_metrics_log']).save_metrics_log
            save_metrics_log(training_dir, step, metrics)
        
        # Checkpoint
        if logger.should_checkpoint(step):
            checkpoint_data = {
                "params": params,
                "obs_mean": obs_normalizer.mean if obs_normalizer else None,
                "obs_std": obs_normalizer.var if obs_normalizer else None,
            }
            logger.save_checkpoint(step, checkpoint_data, opt_state, metrics)
        
        # Render
        if logger.should_render(step):
            # Use normalized policy for rendering
            def render_policy_fn(params_dict, obs):
                if isinstance(params_dict, dict) and "params" in params_dict:
                    p = params_dict["params"]
                    obs_m = params_dict.get("obs_mean", jnp.zeros_like(obs))
                    obs_s = params_dict.get("obs_std", jnp.ones_like(obs))
                    obs_norm = jnp.clip((obs - obs_m) / (jnp.sqrt(obs_s) + 1e-8), -10.0, 10.0)
                    return policy_model.apply(p, obs_norm)
                else:
                    return policy_model.apply(params_dict, obs)
            
            logger.render_video(step, m, sys, checkpoint_data, render_policy_fn, q0)
    
    logger.print_final_summary()
    return params, training_dir


if __name__ == "__main__":
    train()
