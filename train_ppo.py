"""
MJX Humanoid PPO Training using Pure JAX (No Flax).
Optimized based on benchmark: ~72,618 device steps/sec
"""
import warnings
# Suppress Warp deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="warp.*")
warnings.filterwarnings("ignore", message=".*warp.context.*")
warnings.filterwarnings("ignore", message=".*warp.math.*")

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Setup JAX environment first
from src.training_utils import setup_jax_environment
setup_jax_environment()

import time
import jax
import jax.numpy as jnp
from jax import jit, lax, random
from functools import partial
import optax
# import flax.linen as nn <-- REMOVED

# Enable NaN debugging (TURN OFF FOR SPEED)
jax.config.update("jax_debug_nans", False)

from src.training_utils import (
    load_model_and_create_env,
    TrainingLogger,
    print_training_header,
    print_config_summary,
    RMSState, update_rms, normalize_obs
)
from src.checkpoint_utils import create_training_dir, save_config
from src.config import PPOConfig
from src.networks import GaussianPolicy, ValueNet

def train():
    """Main PPO training loop."""
    # 1. Configuration
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Use test config (fast debug)")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    args, unknown = parser.parse_known_args()
    
    if args.config:
        config_path = args.config
    elif args.test:
        config_path = os.path.join(os.path.dirname(__file__), "src", "config_test.json")
    else:
        config_path = os.path.join(os.path.dirname(__file__), "src", "config.json")
        
    if os.path.exists(config_path):
        print(f"Loading config from {config_path}")
        cfg = PPOConfig.from_json(config_path)
    else:
        print(f"Config file not found: {config_path}, using defaults")
        cfg = PPOConfig()
    
    print_training_header("PPO", cfg)
    
    # Create training directory
    training_dir = create_training_dir("ppo")
    save_config(training_dir, cfg)
    logger = TrainingLogger(training_dir, cfg)
    
    # 2. Environment
    try:
        m, sys, q0, nq, nv, nu, single_reset, single_step, v_reset, v_step = load_model_and_create_env(
            cfg.xml_path,
            cfg.env_config,
            cfg.lighten_solver
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        # raise e # Debug
        return
    
    # 3. Networks (Pure JAX)
    # Obs Dim matches src/envs.py logic: 1 + 3 + (nq-7) + nv + 2
    obs_dim = 1 + 3 + (nq - 7) + nv + 2
    act_dim = nu
    
    rng = random.PRNGKey(cfg.seed)
    
    # Policy
    policy_model = GaussianPolicy(
        action_dim=act_dim, 
        hidden_layer_specs=cfg.policy_hidden_layer_specs,
        log_std_init=cfg.log_std_init
    )
    rng, init_rng_p = random.split(rng)
    policy_params = policy_model.init(init_rng_p, obs_dim)
    
    # Value
    value_model = ValueNet(
        hidden_layer_specs=cfg.value_hidden_layer_specs
    )
    rng, init_rng_v = random.split(rng)
    value_params = value_model.init(init_rng_v, obs_dim)
    
    # 4. Optimizers
    opt_p = optax.adam(cfg.lr_policy)
    opt_v = optax.adam(cfg.lr_value)
    
    opt_state_p = opt_p.init(policy_params)
    opt_state_v = opt_v.init(value_params)
    
    # 5. Functions
    num_envs = cfg.num_envs
    rollout_length = cfg.rollout_length
    
    rng, key_reset = random.split(rng)
    keys = random.split(key_reset, num_envs)
    state_b, obs_b = v_reset(keys)

    @jit
    def gaussian_logprob(mean, log_std, action):
        std = jnp.exp(log_std)
        var = std * std
        log2pi = jnp.log(2.0 * jnp.pi)
        return -0.5 * jnp.sum(((action - mean) ** 2) / var + 2.0 * log_std + log2pi, axis=-1)
    
    @partial(jit, static_argnames=("v_reset", "num_envs"))
    def collect_rollout(policy_params, rms_state, rng, state_b, obs_b, v_reset, num_envs):
        def step_fn(carry, _):
            rng, state, obs = carry
            rng, key = random.split(rng)
            
            # Normalize observations for Policy
            obs_norm = normalize_obs(rms_state, obs)
            obs_norm = jnp.clip(obs_norm, -10.0, 10.0)
            
            mean, log_std = policy_model.apply(policy_params, obs_norm)
            eps = random.normal(key, mean.shape).astype(jnp.float32)
            act = mean + jnp.exp(log_std) * eps
            
            # Step with tuple state
            state_next, obs_next, r, done = v_step(state, act)
            logp = gaussian_logprob(mean, log_std, act)
            
            # --- Auto-Reset ---
            rng, key_reset = random.split(rng)
            keys = random.split(key_reset, num_envs)
            state_reset, obs_reset = v_reset(keys)
            
            def merge_if_done(x_step, x_reset):
                target_shape = x_step.shape
                d_shape = (target_shape[0],) + (1,) * (len(target_shape) - 1)
                d_broad = done.reshape(d_shape)
                return jnp.where(d_broad, x_reset, x_step)
            
            state_next = jax.tree_util.tree_map(merge_if_done, state_next, state_reset)
            obs_next = merge_if_done(obs_next, obs_reset)
            # ------------------
            
            return (rng, state_next, obs_next), (obs, act, logp, r, done)

        (rng_final, state_last, obs_last), (obs_traj, act_traj, logp_traj, r_traj, done_traj) = lax.scan(
            step_fn, (rng, state_b, obs_b), None, length=rollout_length
        )
        return state_last, obs_last, obs_traj, act_traj, logp_traj, r_traj, done_traj

    @jit
    def compute_gae(rewards, values, dones):
        def scan_fn(carry, inputs):
            next_value, reward, done, value = inputs
            delta = reward + cfg.gamma * next_value * (1.0 - done) - value
            advantage = delta + cfg.gamma * cfg.lam * (1.0 - done) * carry
            return advantage, advantage

        init_adv = jnp.zeros_like(rewards[0])
        inputs = (values[1:], rewards, dones, values[:-1])
        _, adv_rev = lax.scan(lambda c, x: scan_fn(c, x), init_adv, inputs, reverse=True)
        ret = adv_rev + values[:-1]
        return adv_rev, ret

    @jit
    def ppo_loss_fn(policy_params, obs, acts, old_logp, adv):
        mean, log_std = policy_model.apply(policy_params, obs)
        logp = gaussian_logprob(mean, log_std, acts)
        ratio = jnp.exp(logp - old_logp)
        adv_n = (adv - jnp.mean(adv)) / (jnp.std(adv) + 1e-8)
        unclipped = ratio * adv_n
        clipped = jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_n
        policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
        entropy = 0.5 * jnp.sum(1.0 + jnp.log(2.0 * jnp.pi) + 2.0 * log_std) / acts.shape[-1]
        return policy_loss - cfg.ent_coef * entropy

    @jit
    def value_loss_fn(value_params, obs, returns):
        v = value_model.apply(value_params, obs)
        return jnp.mean((v - returns) ** 2)

    # Note: We need to define index generation helper here
    def make_index_batches(rng, total_size, minibatch_size, epochs):
        steps_per_epoch = total_size // minibatch_size
        cut = steps_per_epoch * minibatch_size
        idx_batches = []
        keys = random.split(rng, epochs)
        for k in keys:
            perm = random.permutation(k, total_size)
            perm = perm[:cut]
            idx_batches.append(perm.reshape(steps_per_epoch, minibatch_size))
        return jnp.concatenate(idx_batches, axis=0)

    @jit
    def run_ppo_updates(policy_params, value_params, opt_state_p, opt_state_v,
                       obs_flat, act_flat, logp_flat, ret_flat, adv_flat, index_batches):
        def body(carry, idx):
            pp, vp, osp, osv = carry
            o, a, olp, r, ad = obs_flat[idx], act_flat[idx], logp_flat[idx], ret_flat[idx], adv_flat[idx]
            
            gp = jax.grad(ppo_loss_fn)(pp, o, a, olp, ad)
            up_p, osp = opt_p.update(gp, osp, pp)
            pp = optax.apply_updates(pp, up_p)
            
            gv = jax.grad(value_loss_fn)(vp, o, r)
            up_v, osv = opt_v.update(gv, osv, vp)
            vp = optax.apply_updates(vp, up_v)
            return (pp, vp, osp, osv), None

        (pp, vp, osp, osv), _ = lax.scan(
            body, (policy_params, value_params, opt_state_p, opt_state_v), index_batches
        )
        return pp, vp, osp, osv

    # Print config
    print_config_summary(cfg.__dict__)
    
    total_timesteps = cfg.total_iterations * cfg.num_envs * cfg.rollout_length
    print(f"Total Timesteps: {total_timesteps:,} ({total_timesteps/1e6:.1f}M)")

    # Evaluation Function (Deterministic)
    @jit
    def evaluate(policy_params, rms_state, rng, iteration, num_eval_steps=500):
        """Run deterministic evaluation with iteration-based seed variation."""
        eval_envs = 32  # Run 32 parallel envs for evaluation
        
        # Create deterministic but varied seed based on iteration
        # This ensures different initial conditions per evaluation while remaining reproducible
        eval_seed = cfg.seed + 10000 + iteration  # Offset to avoid training seed collision
        eval_rng = random.PRNGKey(eval_seed)
        
        rng, key_reset = random.split(eval_rng)
        keys = random.split(key_reset, eval_envs)
        
        # Reset eval envs
        state, obs = v_reset(keys)
        
        def step_fn(carry, _):
            rng, state, obs, acc_reward = carry
            
            # Normalize for Policy
            obs_norm = normalize_obs(rms_state, obs)
            obs_norm = jnp.clip(obs_norm, -10.0, 10.0)
            
            # Deterministic Action (Mean only)
            mean, _ = policy_model.apply(policy_params, obs_norm)
            act = mean # No noise
            
            state_next, obs_next, r, done = v_step(state, act)
            
            # Auto-Reset
            rng, key_reset = random.split(rng)
            keys = random.split(key_reset, eval_envs)
            state_reset, obs_reset = v_reset(keys)
            
            def merge_if_done(x_step, x_reset):
                target_shape = x_step.shape
                d_shape = (target_shape[0],) + (1,) * (len(target_shape) - 1)
                d_broad = done.reshape(d_shape)
                return jnp.where(d_broad, x_reset, x_step)
                
            state_next = jax.tree_util.tree_map(merge_if_done, state_next, state_reset)
            obs_next = merge_if_done(obs_next, obs_reset)
            
            new_acc_reward = acc_reward + r
            return (rng, state_next, obs_next, new_acc_reward), None

        init_acc = jnp.zeros(eval_envs)
        # Carry: rng, state, obs, acc_reward
        (rng, _, _, total_rewards), _ = lax.scan(
            step_fn, (rng, state, obs, init_acc), None, length=num_eval_steps
        )
        return jnp.mean(total_rewards)

    # 8. RMS State
    rms_state = RMSState.create(obs_dim)
    
    # 9. Training Loop
    total_iterations = cfg.total_iterations
    
    for it in range(total_iterations):
        iter_t0 = time.time()
        
        # Collect rollout
        rng, key_roll = random.split(rng)
        state_last, obs_last, obs_traj, act_traj, logp_traj, r_traj, done_traj = collect_rollout(
            policy_params, rms_state, key_roll, state_b, obs_b, v_reset, num_envs
        )
        
        # Update RMS with collected observations
        # Flatten batch and time dims
        obs_traj_flat = obs_traj.reshape(-1, obs_dim)
        rms_state = update_rms(rms_state, obs_traj_flat)
        
        # Normalize observations for Updates
        obs_traj_norm = normalize_obs(rms_state, obs_traj)
        obs_traj_norm = jnp.clip(obs_traj_norm, -10.0, 10.0)
        
        obs_last_norm = normalize_obs(rms_state, obs_last)
        obs_last_norm = jnp.clip(obs_last_norm, -10.0, 10.0)
        
        # Compute values and GAE
        obs_stack = jnp.concatenate([obs_traj_norm, obs_last_norm[jnp.newaxis, ...]], axis=0)
        # Flatten batch dim for value net apply
        obs_stack_flat = obs_stack.reshape((rollout_length + 1) * num_envs, -1)
        v_traj_plus_flat = value_model.apply(value_params, obs_stack_flat)
        v_traj_plus = v_traj_plus_flat.reshape(rollout_length + 1, num_envs)
        
        adv, ret = compute_gae(r_traj, v_traj_plus, done_traj)
        
        # Flatten (Use Normalized Obs)
        obs_flat = obs_traj_norm.reshape(rollout_length * num_envs, -1)
        act_flat = act_traj.reshape(rollout_length * num_envs, -1)
        logp_flat = logp_traj.reshape(rollout_length * num_envs)
        adv_flat = adv.reshape(rollout_length * num_envs)
        ret_flat = ret.reshape(rollout_length * num_envs)
        
        # PPO Updates
        rng, rng_idx = random.split(rng)
        total_size = rollout_length * num_envs
        index_batches = make_index_batches(rng_idx, total_size, cfg.minibatch_size, cfg.epochs)
        
        policy_params, value_params, opt_state_p, opt_state_v = run_ppo_updates(
            policy_params, value_params, opt_state_p, opt_state_v,
            obs_flat, act_flat, logp_flat, ret_flat, adv_flat, index_batches
        )
        
        # Update env state
        state_b, obs_b = state_last, obs_last
    
        # Metrics
        env_steps = float(num_envs * rollout_length)
        logger.total_env_steps += env_steps
        iter_dt = max(time.time() - iter_t0, 1e-9)
        env_steps_per_sec = env_steps / iter_dt
        
        # Training Metrics
        # r_traj is (Rollout, NumEnvs). Sum over time, then mean over envs.
        train_return_avg = float(jnp.mean(jnp.sum(r_traj, axis=0)))
        train_return_max = float(jnp.max(jnp.sum(r_traj, axis=0)))
        
        # Episode Length (Approximate)
        # Total steps / (Total dones). If no dones, it's rollout length (or infinity).
        total_dones = float(jnp.sum(done_traj))
        if total_dones > 0:
            train_eplen_avg = env_steps / total_dones
        else:
            train_eplen_avg = float(rollout_length) # Fallback if no episodes finished
        
        metrics = {
            "train_return_avg": train_return_avg,
            "train_return_max": train_return_max,
            "train_eplen_avg": train_eplen_avg,
            "env_steps_per_sec": env_steps_per_sec,
        }

        # --- Evaluation & Checkpointing & Rendering ---
        should_render = (it % cfg.eval_interval == 0) and cfg.save_video
        should_checkpoint = (it % cfg.checkpoint_every == 0)
        should_eval = (it % cfg.eval_interval == 0)
        should_log = (it % cfg.log_interval == 0) or should_render or should_checkpoint or (it == total_iterations - 1) or should_eval
        
        # Progress & Time
        elapsed_sec = time.time() - logger.start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_sec))
        
        current_step_count = int(logger.total_env_steps)
        total_step_count = int(total_iterations * env_steps)
        pct = (current_step_count / total_step_count) * 100.0
        progress = f"{current_step_count:.1e}/{total_step_count:.1e} ({pct:.1f}%)"

        if should_log:
             log_str = f"Iter {it:4d} | Time {elapsed_str} | Step {progress} | S/s: {env_steps_per_sec:8.0f}"
             log_str += f" | Train_Ret(Avg): {train_return_avg:7.2f}"
             log_str += f" | Train_Len(Avg): {train_eplen_avg:6.0f}"
             
             if should_eval: 
                 rng, key_eval = random.split(rng)
                 eval_return = evaluate(policy_params, rms_state, key_eval, it)
                 metrics["eval_return"] = float(eval_return)
                 log_str += f" | Eval_Ret(Avg): {float(eval_return):7.2f}"
             
             logger.log_step(it, metrics, log_str)
        
        # Checkpoint
        if should_checkpoint:
            checkpoint_data = {"policy": policy_params, "value": value_params, "rms": rms_state}
            checkpoint_opt = {"policy": opt_state_p, "value": opt_state_v}
            logger.save_checkpoint(it, checkpoint_data, checkpoint_opt, metrics)
        
        # Render
        if should_render:
            # TODO: Improve render to handle tuple state if possible, currently it might fail or rely on naive assumption
            # For now we use the same render_policy_fn. 
            # Note: render_policy_rollout must be updated to support the new step function signature if we want it to work perfectly.
            # But render_policy_rollout typically implementation uses mujoco step or mjx step directly.
            # If so, it won't have the RandomFlip logic.
            # However, for visualization, it usually shows the "True" physics.
            # But the policy expects Flipped/Feature Obs.
            # If the rendering loop constructs OBS naively, the policy will fail.
            pass
            # To fix rendering:
            # We use the new env_step_fn support in render_policy_rollout
            
            def render_policy_fn(params, obs):
                obs = normalize_obs(rms_state, obs)
                obs = jnp.clip(obs, -10.0, 10.0)
                mean, _ = policy_model.apply(params, obs)
                return mean 
            
            logger.render_video(
                it, m, sys, policy_params, render_policy_fn, q0, 
                filename_prefix="iter",
                env_step_fn=single_step,
                env_init_fn=single_reset,
            )
            
    logger.print_final_summary()
    return policy_params, value_params, training_dir

if __name__ == "__main__":
    train()
