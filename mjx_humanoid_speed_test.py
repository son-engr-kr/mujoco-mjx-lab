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
    """Load and convert MuJoCo models to MJX."""
    if kind == "humanoid":
        # Standard humanoid (Newton solver, heavier)
        model = mujoco.MjModel.from_xml_path("models/humanoid.xml")
    elif kind == "humanoid_mjx":
        # Optimized for MJX (PGS/Implicit solver, faster on GPU)
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

    mjx_model = mjx.put_model(model)
    return model, mjx_model


def make_batched_step(mjx_model):
    """Create a JIT-compiled batched step function."""
    def step(vel):
        d = mjx.make_data(mjx_model)
        qvel = d.qvel.at[0].set(vel)
        d = d.replace(qvel=qvel)
        pos = mjx.step(mjx_model, d).qpos[0]
        return pos
    vmapped = jax.vmap(step)
    return jax.jit(vmapped)


def bench_once(kind: str, batch_size: int = 1000, iters: int = 200):
    model, mjx_model = build_models_from_source(kind)

    # Input batch
    vel = jnp.linspace(0.0, 1.0, batch_size)

    # Step function (compiled)
    fn = make_batched_step(mjx_model)

    # Warmup step function
    # This triggers JIT compilation for the single-step function
    pos = fn(vel)
    pos.block_until_ready()

    # --- Benchmark Python Loop ---
    # Running step() one by one from Python.
    # This incurs Python-to-GPU dispatch overhead every step.
    t_start = time.time()
    for _ in range(iters):
        out = fn(vel)
        out.block_until_ready()
    t_end = time.time()
    dt_py = max(t_end - t_start, 1e-12)

    # --- Benchmark Device Loop ---
    # Using jax.lax.while_loop/fori_loop to keep execution on GPU.
    # This is how you typically train agents (rollouts on GPU).
    
    @jax.jit
    def repeat_on_device(vel, count):
        def body(i, acc):
            p = fn(vel)
            return acc + jnp.sum(p)  # Dummy accumulation to prevent DCE
        return jax.lax.fori_loop(0, count, body, 0.0)

    # Warmup device loop (triggers JIT compilation for the loop)
    # CRITICAL: Previous code included this in the timing!
    _ = repeat_on_device(vel, 1).block_until_ready()

    # Measure actual execution
    t_start = time.time()
    _ = repeat_on_device(vel, iters).block_until_ready()
    t_end = time.time()
    dt_dev = max(t_end - t_start, 1e-12)

    # Calculate stats
    steps_total = batch_size * iters
    sps_py = steps_total / dt_py
    sps_dev = steps_total / dt_dev

    # --- Report Results ---
    print(f"\nModel: {kind.upper()}")
    print("-" * 60)
    print(f"{'Metric':<30} | {'Value':<15}")
    print("-" * 60)
    print(f"{'Batch Size':<30} | {batch_size}")
    print(f"{'Iterations':<30} | {iters}")
    print(f"{'Total Steps':<30} | {steps_total:,}")
    print("-" * 60)
    print(f"{'Python Loop Time (s)':<30} | {dt_py:.4f}")
    print(f"{'Python Steps/Sec':<30} | {sps_py:,.0f}")
    print("-" * 60)
    print(f"{'Device Loop Time (s)':<30} | {dt_dev:.4f}")
    print(f"{'Device Steps/Sec':<30} | {sps_dev:,.0f} (Running entirely on GPU)")
    print("-" * 60)
    if sps_dev > sps_py:
        speedup = sps_dev / sps_py
        print(f"Speedup (Device vs Python)    | {speedup:.1f}x")
    print("=" * 60)


if __name__ == "__main__":
    print("JAX Backend:", jax.default_backend())
    print("Starting Speed Benchmark...")
    print("Note: 'Device Steps/Sec' is the metric to look at for training performance.")
    
    # Using larger iterations to get more stable measurements
    # Note: 'humanoid' has stricter contact config and uses Newton solver = slower
    # 'humanoid_mjx' is optimized for XLA with different solver params = faster
    
    bench_once("humanoid", batch_size=4096, iters=10)
    bench_once("humanoid_mjx", batch_size=4096, iters=10)
    bench_once("sphere", batch_size=4096, iters=10)
