import os
import time

# Configure JAX/XLA before importing jax
os.environ["JAX_PLATFORMS"] = os.environ.get("JAX_PLATFORMS", "cuda")
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx



def build_models_from_source(kind: str):
    if kind == "humanoid":
        model = mujoco.MjModel.from_xml_path("models/humanoid.xml")
    elif kind == "humanoid_mjx":
        model = mujoco.MjModel.from_xml_path("models/humanoid_mjx.xml")
    elif kind == "sphere":
        XML = r"""
        <mujoco>
          <worldbody>
            <body>
              <freejoint/>
              <geom size=".15" mass="1" type="sphere"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(XML)
    else:
        raise ValueError(f"unknown kind: {kind}")

    print("model.opt.iterations:", model.opt.iterations)
    print("model.opt.ls_iterations:", model.opt.ls_iterations)
    mjx_model = mjx.put_model(model)
    return model, mjx_model


def make_batched_step(mjx_model):
    def step(vel):
        d = mjx.make_data(mjx_model)
        qvel = d.qvel.at[0].set(vel)
        d = d.replace(qvel=qvel)
        pos = mjx.step(mjx_model, d).qpos[0]
        return pos
    vmapped = jax.vmap(step)
    return jax.jit(vmapped)


def bench_once(kind: str, batch_size: int = 1000, iters: int = 200):
    _, mjx_model = build_models_from_source(kind)

    print("JAX backend:", jax.default_backend())

    # Input batch
    vel = jnp.linspace(0.0, 1.0, batch_size)

    # build per-model compiled kernel (no global capture)
    fn = make_batched_step(mjx_model)

    # warmup
    t0 = time.time()
    pos = fn(vel)
    pos.block_until_ready()
    warmup_dt = max(time.time() - t0, 1e-12)
    print("sample pos[0..4]:", jnp.asarray(pos[:5]))
    print("warmup time (s):", f"{warmup_dt:.6f}")

    # python-loop timing
    t1 = time.time()
    for _ in range(iters):
        out = fn(vel)
        out.block_until_ready()
    t2 = time.time()
    dt_py = max(t2 - t1, 1e-12)

    # device-loop timing via fori_loop
    @jax.jit
    def repeat_on_device(vel, iters):
        def body(i, acc):
            p = fn(vel)
            return acc + jnp.sum(p)  # DCE guard
        return jax.lax.fori_loop(0, iters, body, 0.0)

    t3 = time.time()
    _ = repeat_on_device(vel, iters).block_until_ready()
    t4 = time.time()
    dt_dev = max(t4 - t3, 1e-12)

    steps = int(batch_size) * iters
    sps_py = steps / dt_py
    sps_dev = steps / dt_dev

    print(f"[{kind}] batch size:", batch_size)
    print(f"[{kind}] iterations:", iters)
    print(f"[{kind}] elapsed python-loop (s):", f"{dt_py:.6f}")
    print(f"[{kind}] env-steps/s python-loop:", f"{sps_py:.1f}")
    print(f"[{kind}] elapsed device-loop (s):", f"{dt_dev:.6f}")
    print(f"[{kind}] env-steps/s device-loop:", f"{sps_dev:.1f}")


if __name__ == "__main__":
    bench_once("humanoid", batch_size=10000, iters=50)
    bench_once("humanoid_mjx", batch_size=10000, iters=50)
    bench_once("sphere", batch_size=10000, iters=50)


