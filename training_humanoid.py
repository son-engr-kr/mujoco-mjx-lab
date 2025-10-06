# %% [markdown]
# # MJX Humanoid Training (Minimal)
# 
# - 목적: `models/humanoid.xml`로 MJX FoPG(APG) 최소 예제 학습
# - 원칙: mjx만 사용, 핵심만 간결히 구현 (Brax 미사용)
# 

# %%
# Setup: imports, JAX precision
import os
os.environ["JAX_PLATFORMS"] = "cuda"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad, lax, jacfwd
from jax import config as jax_config
jax_config.update("jax_enable_x64", False)
jax_config.update("jax_default_matmul_precision", "high")

import numpy as np
import mujoco
import mujoco.mjx as mjx




# %%
# JAX device check
print("JAX default backend:", jax.default_backend())
print("JAX devices:", jax.devices())
x = jnp.ones((1,))
print("Array device:", x.device)
print("Primary device:", jax.devices()[0])

# %%
import time
import jax, jax.numpy as jnp
from jax import jit

print("jax:", jax.__version__)
print("devices:", jax.devices())

@jit
def bigmm():
    x = jnp.ones((8192*2, 8192*2), dtype=jnp.float32)  # FP32 
    return x @ x.T

t0 = time.time()
y = bigmm().block_until_ready()
t1 = time.time()
print("first run:", t1 - t0, "sec", "device:", y.device, "dtype:", y.dtype)

t0 = time.time()
y = bigmm().block_until_ready()
t1 = time.time()
print("second run:", t1 - t0, "sec")

# %%
x = jnp.ones((8192, 8192))
y = x @ x.T
_ = y.block_until_ready()  # wait until operation is done

# %%
XML_PATH = "models/humanoid.xml"

# %%
# Load model and build MJX system
m = mujoco.MjModel.from_xml_path(XML_PATH)
sys = mjx.put_model(m)

# State sizes
q_size = m.nq
qd_size = m.nv
act_size = m.nu

# Initial state from keyframe or default
q0 = np.copy(m.qpos0).astype(np.float32)
qd0 = np.zeros(qd_size, dtype=np.float32)

@jit
def pipeline_init(q, qd):
    data = mjx.make_data(sys)
    data = data.replace(qpos=q, qvel=qd)
    data = mjx.forward(sys, data)
    return data

@jit
def pipeline_step(data, ctrl):
    data = data.replace(ctrl=ctrl)
    data = mjx.step(sys, data)
    return data


# %%
# Minimal MLP policy (stateless)

def mlp_init(rng, in_dim, hidden, out_dim):
    k1, k2, k3 = random.split(rng, 3)
    w1 = (random.normal(k1, (in_dim, hidden)) * (1.0 / jnp.sqrt(in_dim))).astype(jnp.float32)
    b1 = jnp.zeros((hidden,), dtype=jnp.float32)
    w2 = (random.normal(k2, (hidden, hidden)) * (1.0 / jnp.sqrt(hidden))).astype(jnp.float32)
    b2 = jnp.zeros((hidden,), dtype=jnp.float32)
    w3 = (random.normal(k3, (hidden, out_dim)) * (1.0 / jnp.sqrt(hidden))).astype(jnp.float32)
    b3 = jnp.zeros((out_dim,), dtype=jnp.float32)
    return (w1, b1, w2, b2, w3, b3)

@jit
def mlp_apply(params, x):
    w1, b1, w2, b2, w3, b3 = params
    x = x.astype(jnp.float32)
    x = jnp.tanh(x @ w1 + b1)
    x = jnp.tanh(x @ w2 + b2)
    x = jnp.tanh(x @ w3 + b3)
    return x

obs_dim = q_size + qd_size
act_dim = act_size
rng = random.PRNGKey(0)
params = mlp_init(rng, obs_dim, 128, act_dim)


# %%
# Rollout and loss (FoPG)

@jit
def get_obs(d):
    return jnp.concatenate([d.qpos, d.qvel], axis=0).astype(jnp.float32)

@jit
def reward_fn(d):
    height = d.qpos[2].astype(jnp.float32)
    up_reward = jnp.exp(-jnp.square(jnp.float32(1.4) - height))
    ang_vel_pen = jnp.sum(jnp.square(d.qvel[:3].astype(jnp.float32)))
    joint_vel_pen = jnp.sum(jnp.square(d.qvel[3:].astype(jnp.float32)))
    return up_reward - jnp.float32(1e-3) * ang_vel_pen - jnp.float32(1e-4) * joint_vel_pen

@jit
def rollout_return(params, key, horizon=256):
    d0 = pipeline_init(q0, qd0)
    gamma = 0.99

    def body(carry, _):
        d, disc = carry
        obs = get_obs(d)
        act = mlp_apply(params, obs)
        d_next = pipeline_step(d, act)
        r = reward_fn(d_next)
        return (d_next, disc * gamma), disc * r

    (_, _), rets = lax.scan(body, (d0, 1.0), None, length=horizon)
    return jnp.sum(rets)

loss = jit(lambda p, k: -rollout_return(p, k))


# %%
# SGD training (very short demo)

@jit
def sgd_update(params, grads, lr):
    w1, b1, w2, b2, w3, b3 = params
    g1, gb1, g2, gb2, g3, gb3 = grads
    return (
        w1 - lr * g1,
        b1 - lr * gb1,
        w2 - lr * g2,
        b2 - lr * gb2,
        w3 - lr * g3,
        b3 - lr * gb3,
    )

key = random.PRNGKey(42)
lr = 3e-4
num_updates = 1

# forward-mode grad to avoid reverse-mode through mjx internal while_loops
loss_fn = lambda p, k: loss(p, k)
# Jacobian-vector product (forward) of loss w.r.t params; we aggregate per-parameter grads
fwd_grad = jit(jacfwd(loss_fn))

@jit
def train_scan(params, key):
    def body(carry, _):
        params, key = carry
        key, sub = random.split(key)
        v = loss_fn(params, sub)
        g = fwd_grad(params, sub)
        params = sgd_update(params, g, lr)
        return (params, key), v
    (params_f, _), losses = lax.scan(body, (params, key), None, length=num_updates)
    return params_f, losses

params, losses = train_scan(params, key)
for i, v in enumerate(losses):
    print(i+1, "loss:", float(v))


# %%



