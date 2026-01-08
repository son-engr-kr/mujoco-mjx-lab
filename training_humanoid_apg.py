# %% [markdown]
# MJX-only Humanoid APG (FoPG) Training with Parallel Envs


# %%
# Setup
import os
os.environ["JAX_PLATFORMS"] = "cuda"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# Reduce autotuning memory usage
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_autotune_level=3").strip()

from typing import Tuple
import time

import jax
import jax.numpy as jnp
from jax import jit, lax, random, jacfwd
import optax
import numpy as np
import mujoco
import mujoco.mjx as mjx


# %%
# Model / System
XML_PATH = "models/humanoid.xml"
m = mujoco.MjModel.from_xml_path(XML_PATH)
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


# %%
# Batched env via vmap
v_reset = jit(jax.vmap(single_reset))
v_step = jit(jax.vmap(single_step, in_axes=(0, 0)))


# %%
# Policy (deterministic MLP)
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
    x = jnp.tanh(x @ w3 + b3)
    return x


# %%
# Rollout return (parallel envs) for APG


# %%
# Train
def main():
    obs_dim = nq + nv
    act_dim = nu
    rng = random.PRNGKey(0)
    params = mlp_init(rng, obs_dim, 16, act_dim)

    batch_size = 8  # number of parallel environments
    horizon = 64    # rollout length (steps per environment per update)
    gamma = 0.99
    lr = 3e-4
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    # Build loss and gradient fns that close over static hyper-parameters
    def make_losses(batch_size: int, horizon: int, gamma: float):
        @jit
        def rollout_return(params, rngs):
            d0, _ = v_reset(rngs)

            def body(carry, _):
                d, disc, acc = carry
                x = jnp.concatenate([d.qpos.astype(jnp.float32), d.qvel.astype(jnp.float32)], axis=1)
                act = mlp_apply(params, x)
                d_next, _, r, done = v_step(d, act)
                disc_next = disc * gamma * (1.0 - done)
                acc_next = acc + (disc * r)
                return (d_next, disc_next, acc_next), None

            init_disc = jnp.ones((batch_size,), dtype=jnp.float32)
            init_acc = jnp.zeros((batch_size,), dtype=jnp.float32)
            remat_body = jax.checkpoint(body)
            (__, __, acc), _ = lax.scan(remat_body, (d0, init_disc, init_acc), None, length=horizon)
            return jnp.mean(acc)

        loss_fn = jit(lambda p, keys: -rollout_return(p, keys))
        fwd_grad = jit(jacfwd(loss_fn))
        return loss_fn, fwd_grad

    loss_fn, fwd_grad = make_losses(batch_size, horizon, gamma)

    @jit
    def update(params, opt_state, keys):
        v = loss_fn(params, keys)
        g = fwd_grad(params, keys)
        updates, opt_state = opt.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, v

    print(f"Parallel envs (batch_size) = {batch_size}, rollout_length (horizon) = {horizon}")

    for step in range(5):  # small demo; increase for training
        iter_t0 = time.time()
        rng, sub = random.split(rng)
        keys = random.split(sub, batch_size)
        params, opt_state, v = update(params, opt_state, keys)
        iter_dt = max(time.time() - iter_t0, 1e-9)
        env_steps = float(batch_size * horizon)
        env_steps_per_sec = env_steps / iter_dt
        print(step, "loss:", float(v), "env-steps/s:", f"{env_steps_per_sec:.1f}")

    return params


if __name__ == "__main__":
    print("JAX backend:", jax.default_backend())
    _ = main()


