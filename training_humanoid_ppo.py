# %% [markdown]
# MJX-only Humanoid PPO Training with Parallel Envs


# %%
# Setup: imports, JAX precision
import os
os.environ["JAX_PLATFORMS"] = "cuda"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

from typing import Tuple
import time

import jax
import jax.numpy as jnp
from jax import jit, lax, random
import optax
import numpy as np
import mujoco
import mujoco.mjx as mjx


# %%
# Model / System
XML_PATH = "models/humanoid.xml"
m = mujoco.MjModel.from_xml_path(XML_PATH)
# Lighten solver to improve throughput (trades off physical accuracy)
m.opt.iterations = 1
m.opt.ls_iterations = 1
sys = mjx.put_model(m)

nq, nv, nu = int(m.nq), int(m.nv), int(m.nu)
q0 = np.copy(m.qpos0).astype(np.float32)


# %%
# Env (single) functions
@jit
def single_pipeline_init(q, qd):
    d = mjx.make_data(sys)
    d = d.replace(qpos=q, qvel=qd)
    d = mjx.forward(sys, d)
    return d


@jit
def single_reset(key: jax.Array) -> Tuple[mjx.Data, jax.Array]:
    k1, k2 = random.split(key)
    qpos = jnp.array(q0)
    qvel = jnp.zeros(nv, dtype=jnp.float32)
    # small randomization
    qpos = qpos.at[0:3].add(0.01 * (random.uniform(k1, (3,)) - 0.5))
    qvel = qvel.at[:].add(0.01 * (random.uniform(k2, (nv,)) - 0.5))
    d = single_pipeline_init(qpos, qvel)
    # place on ground
    pen = jnp.min(d._impl.contact.dist)
    qpos = qpos.at[2].set(qpos[2] - pen)
    d = single_pipeline_init(qpos, qvel)
    obs = jnp.concatenate([d.qpos.astype(jnp.float32), d.qvel.astype(jnp.float32)], axis=0)
    return d, obs


@jit
def single_step(d: mjx.Data, action: jax.Array):
    act = jnp.clip(action, -1.0, 1.0)
    d = d.replace(ctrl=act)
    d = mjx.step(sys, d)
    x = d.qpos
    v = d.qvel
    obs = jnp.concatenate([x.astype(jnp.float32), v.astype(jnp.float32)], axis=0)

    # reward from training_humanoid.py
    height = x[2].astype(jnp.float32)
    upright = jnp.exp(-jnp.square(jnp.float32(1.4) - height))
    ang_vel_pen = jnp.sum(jnp.square(v[:3].astype(jnp.float32)))
    joint_vel_pen = jnp.sum(jnp.square(v[3:].astype(jnp.float32)))
    reward = upright - jnp.float32(1e-3) * ang_vel_pen - jnp.float32(1e-4) * joint_vel_pen

    done = jnp.where(x[2] < 0.8, 1.0, 0.0)
    return d, obs, reward, done


# Batched env via vmap
v_reset = jit(jax.vmap(single_reset))
v_step = jit(jax.vmap(single_step, in_axes=(0, 0)))


# %%
# Policy/Value networks
def mlp_init(rng, in_dim, hidden, out_dim):
    k1, k2, k3 = random.split(rng, 3)
    w1 = (random.normal(k1, (in_dim, hidden)) / jnp.sqrt(in_dim)).astype(jnp.float32)
    b1 = jnp.zeros((hidden,), dtype=jnp.float32)
    w2 = (random.normal(k2, (hidden, hidden)) / jnp.sqrt(hidden)).astype(jnp.float32)
    b2 = jnp.zeros((hidden,), dtype=jnp.float32)
    w3 = (random.normal(k3, (hidden, out_dim)) / jnp.sqrt(hidden)).astype(jnp.float32)
    b3 = jnp.zeros((out_dim,), dtype=jnp.float32)
    return (w1, b1, w2, b2, w3, b3)


@jit
def mlp_apply(params, x):
    w1, b1, w2, b2, w3, b3 = params
    x = jnp.tanh(x @ w1 + b1)
    x = jnp.tanh(x @ w2 + b2)
    x = x @ w3 + b3
    return x


def policy_init(rng, obs_dim, act_dim):
    params = mlp_init(rng, obs_dim, 128, act_dim)
    log_std = jnp.zeros((act_dim,), dtype=jnp.float32)  # learned log std
    return (params, log_std)


@jit
def policy_apply(policy_params, obs):
    params, log_std = policy_params
    mean = mlp_apply(params, obs)
    return mean, log_std


def value_init(rng, obs_dim):
    return mlp_init(rng, obs_dim, 128, 1)


@jit
def value_apply(params, obs):
    v = mlp_apply(params, obs)
    return v.squeeze(-1)


@jit
def gaussian_logprob(mean, log_std, action):
    std = jnp.exp(log_std)
    var = std * std
    log2pi = jnp.log(2.0 * jnp.pi)
    return -0.5 * (jnp.sum(((action - mean) ** 2) / var + 2.0 * log_std + log2pi, axis=-1))


# %%
# Rollout collector (T steps, N envs)
@jit
def collect_rollout(policy_params, rng, data_b, obs_b, rollout_length, num_envs):
    def step_fn(carry, _):
        rng, d, _ = carry
        rng, key = random.split(rng)
        # obs from data
        obs = jnp.concatenate([d.qpos.astype(jnp.float32), d.qvel.astype(jnp.float32)], axis=1)
        mean, log_std = policy_apply(policy_params, obs)
        eps = random.normal(key, mean.shape).astype(jnp.float32)
        act = mean + jnp.exp(log_std) * eps
        d2, obs2, r, done = v_step(d, act)
        logp = gaussian_logprob(mean, log_std, act)
        return (rng, d2, obs2), (obs, act, logp, r, done)

    (_, d_last, obs_last), (obs_traj, act_traj, logp_traj, r_traj, done_traj) = lax.scan(
        step_fn, (rng, data_b, obs_b), None, length=rollout_length
    )
    return d_last, obs_last, obs_traj, act_traj, logp_traj, r_traj, done_traj


# %%
# Advantage (GAE) and returns
@jit
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    # rewards, dones: [T, N]; values: [T+1, N]
    def scan_fn(carry, inputs):
        next_value, reward, done, value = inputs
        delta = reward + gamma * next_value * (1.0 - done) - value
        advantage = delta + gamma * lam * (1.0 - done) * carry
        return advantage, advantage

    # reverse scan over time
    init_adv = jnp.zeros_like(rewards[0])
    inputs = (values[1:], rewards, dones, values[:-1])
    _, adv_rev = lax.scan(lambda c, x: scan_fn(c, x), init_adv, inputs, reverse=True)
    adv = adv_rev
    ret = adv + values[:-1]
    return adv, ret


# %%
# PPO loss (jitted once, outside the update loop to avoid recompile)
@jit
def ppo_policy_loss(policy_params, obs, acts, old_logp, adv, clip_eps, ent_coef):
    mean, log_std = policy_apply(policy_params, obs)
    logp = gaussian_logprob(mean, log_std, acts)
    ratio = jnp.exp(logp - old_logp)
    adv_n = (adv - jnp.mean(adv)) / (jnp.std(adv) + 1e-8)
    unclipped = ratio * adv_n
    clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_n
    policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
    # entropy of diagonal Gaussian: 0.5 * sum(1 + log(2πσ^2))
    entropy = 0.5 * jnp.sum(1.0 + jnp.log(2.0 * jnp.pi) + 2.0 * policy_params[1]) / acts.shape[-1]
    return policy_loss - ent_coef * entropy


@jit
def ppo_value_loss(value_params, obs, returns):
    v = value_apply(value_params, obs)
    return jnp.mean((v - returns) ** 2)


def ppo_update(policy_params, value_params, opt_p, opt_v, opt_state_p, opt_state_v,
               batch, clip_eps=0.2, vf_coef=0.5, ent_coef=0.0):
    obs, acts, old_logp, returns, adv = batch

    grads_p = jax.grad(ppo_policy_loss)(policy_params, obs, acts, old_logp, adv, clip_eps, ent_coef)
    updates_p, opt_state_p = opt_p.update(grads_p, opt_state_p, policy_params)
    policy_params = optax.apply_updates(policy_params, updates_p)

    grads_v = jax.grad(ppo_value_loss)(value_params, obs, returns)
    updates_v, opt_state_v = opt_v.update(grads_v, opt_state_v, value_params)
    value_params = optax.apply_updates(value_params, updates_v)

    return policy_params, value_params, opt_state_p, opt_state_v


# (moved inside main to close over opt_p/opt_v)


# Build all minibatch indices for epochs, trimming remainder to keep static shape
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


# %%
# Train
def main():
    rng = random.PRNGKey(0)
    obs_dim = nq + nv
    act_dim = nu

    # init networks/opts
    rng, kp, kv = random.split(rng, 3)
    policy_params = policy_init(kp, obs_dim, act_dim)
    value_params = value_init(kv, obs_dim)
    opt_p = optax.adam(3e-4)
    opt_v = optax.adam(1e-3)
    opt_state_p = opt_p.init(policy_params)
    opt_state_v = opt_v.init(value_params)

    # env batch
    num_envs = 512            # number of parallel environments
    rollout_length = 512     # steps per environment per update
    rng, key_reset = random.split(rng)
    keys = random.split(key_reset, num_envs)
    data_b, obs_b = v_reset(keys)

    gamma = 0.99
    lam = 0.95
    epochs = 2
    minibatch_size = rollout_length * num_envs  # single batch per epoch (fastest)

    print(f"Parallel envs (num_envs) = {num_envs}, rollout_length = {rollout_length}")

    for it in range(3):  # small demo
        iter_t0 = time.time()
        rng, key_roll = random.split(rng)
        rollout_t0 = time.time()
        d_last, obs_last, obs_traj, act_traj, logp_traj, r_traj, done_traj = collect_rollout(
            policy_params, key_roll, data_b, obs_b, rollout_length, num_envs
        )
        rollout_dt = max(time.time() - rollout_t0, 1e-9)
        env_steps = float(rollout_length * num_envs)
        rollout_sps = env_steps / rollout_dt

        # values for GAEs (T+1)
        obs_stack = jnp.concatenate([obs_traj, obs_last[jnp.newaxis, ...]], axis=0)
        v_traj_plus = value_apply(value_params, obs_stack.reshape((rollout_length + 1) * num_envs, -1)).reshape(rollout_length + 1, num_envs)
        adv, ret = compute_gae(r_traj, v_traj_plus, done_traj, gamma, lam)

        # flatten
        obs_flat = obs_traj.reshape(rollout_length * num_envs, -1)
        act_flat = act_traj.reshape(rollout_length * num_envs, -1)
        logp_flat = logp_traj.reshape(rollout_length * num_envs)
        adv_flat = adv.reshape(rollout_length * num_envs)
        ret_flat = ret.reshape(rollout_length * num_envs)

        # Build index batches (epochs × minibatches) and run updates in one JIT
        rng, rng_idx = random.split(rng)
        total_size = rollout_length * num_envs
        index_batches = make_index_batches(rng_idx, total_size, minibatch_size, epochs)

        @jit
        def run_updates_scanned(policy_params, value_params, opt_state_p, opt_state_v,
                                obs_flat, act_flat, logp_flat, ret_flat, adv_flat,
                                index_batches, clip_eps, ent_coef):
            def body(carry, idx):
                pp, vp, osp, osv = carry
                o = obs_flat[idx]
                a = act_flat[idx]
                olp = logp_flat[idx]
                r = ret_flat[idx]
                ad = adv_flat[idx]

                gp = jax.grad(ppo_policy_loss)(pp, o, a, olp, ad, clip_eps, ent_coef)
                up_p, osp = opt_p.update(gp, osp, pp)
                pp = optax.apply_updates(pp, up_p)

                gv = jax.grad(ppo_value_loss)(vp, o, r)
                up_v, osv = opt_v.update(gv, osv, vp)
                vp = optax.apply_updates(vp, up_v)
                return (pp, vp, osp, osv), None

            (pp, vp, osp, osv), _ = lax.scan(
                body, (policy_params, value_params, opt_state_p, opt_state_v), index_batches
            )
            return pp, vp, osp, osv

        policy_params, value_params, opt_state_p, opt_state_v = run_updates_scanned(
            policy_params, value_params, opt_state_p, opt_state_v,
            obs_flat, act_flat, logp_flat, ret_flat, adv_flat,
            index_batches, 0.2, 0.0
        )

        # update env state for next rollout
        data_b, obs_b = d_last, obs_last

        # simple logging
        ep_ret = jnp.mean(jnp.sum(r_traj, axis=0))
        iter_dt = max(time.time() - iter_t0, 1e-9)
        total_sps = env_steps / iter_dt  # env-steps per second including update time
        print(
            it,
            "avg return:", float(ep_ret),
            "rollout env-steps/s:", f"{rollout_sps:.1f}",
            "total env-steps/s:", f"{total_sps:.1f}",
        )

    return policy_params, value_params


if __name__ == "__main__":
    print("JAX backend:", jax.default_backend())
    _ = main()


