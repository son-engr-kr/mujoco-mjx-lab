import os
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import imageio
import numpy as np

# Configure JAX
os.environ["JAX_PLATFORMS"] = os.environ.get("JAX_PLATFORMS", "cuda")
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")

def main():
    # 1. Load the model
    # mjx_humanoid_speed_test.py uses 'models/humanoid.xml', attempting to use the same.
    xml_path = "models/humanoid.xml"
    if not os.path.exists(xml_path):
        # Fallback to a simple built-in model if file doesn't exist
        print(f"File {xml_path} not found. Using a simple box model.")
        xml = """
        <mujoco>
          <worldbody>
            <light pos="0 0 10"/>
            <body pos="0 0 1">
              <freejoint/>
              <geom type="box" size=".1 .2 .3" rgba="1 0 0 1"/>
            </body>
            <geom type="plane" size="10 10 .1" rgba="0.8 0.9 0.8 1"/>
          </worldbody>
        </mujoco>
        """
        mj_model = mujoco.MjModel.from_xml_string(xml)
    else:
        print(f"Loading {xml_path}...")
        mj_model = mujoco.MjModel.from_xml_path(xml_path)

    # 2. Create MJX model and data
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.make_data(mjx_model)

    # 3. Define the simulation step function (JIT compiled)
    @jax.jit
    def step_fn(data):
        return mjx.step(mjx_model, data)

    # 4. Run simulation loop (using lax.scan for efficiency on GPU)
    duration = 2.0  # seconds
    fps = 60
    # MJX step size
    dt = mj_model.opt.timestep
    total_steps = int(duration / dt)

    print(f"Simulating {total_steps} steps ({duration}s)...")

    def loop_body(data, _):
        next_data = step_fn(data)
        # Store qpos for rendering
        return next_data, next_data.qpos

    # Run the scan
    final_data, qpos_history = jax.lax.scan(loop_body, mjx_data, None, length=total_steps)

    # 5. Render the results
    # Scan returns JAX arrays on device, convert to numpy CPU arrays
    qpos_history = np.array(qpos_history)
    
    print("Generating frames...")
    frames = []
    renderer = mujoco.Renderer(mj_model)
    mj_data = mujoco.MjData(mj_model)

    # Calculate sampling rate for video FPS
    step_stride = int(1.0 / (fps * dt))
    if step_stride < 1:
        step_stride = 1

    for i in range(0, total_steps, step_stride):
        # Update CPU data with MJX result
        mj_data.qpos[:] = qpos_history[i]
        
        # Forward kinematics to update geom positions
        mujoco.mj_forward(mj_model, mj_data)
        
        renderer.update_scene(mj_data)
        frames.append(renderer.render())

    # 6. Save video using imageio
    output_path = "mjx_render_output.mp4"
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    main()
