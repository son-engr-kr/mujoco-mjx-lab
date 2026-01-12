"""
Environment definitions and factories for Mujoco MJX.
"""
import jax
import jax.numpy as jnp
from jax import jit, random
import mujoco
import mujoco.mjx as mjx
import numpy as np
from typing import Tuple, Callable, Any
from src.config import EnvConfig

# Type alias for ease of use
# EnvState is now (mjx.Data, AuxState)
# AuxState: [flip, tx, ty, tz, close_count, stance_state, stance_last_change_time, last_pot]
AuxState = jax.Array
EnvState = Tuple[mjx.Data, AuxState]
Obs = jax.Array
Action = jax.Array
Reward = jax.Array
Done = jax.Array
EnvStepFn = Callable[[EnvState, Action], Tuple[EnvState, Obs, Reward, Done]]
EnvResetFn = Callable[[jax.Array], Tuple[EnvState, Obs]]

def create_env_functions(sys: mjx.Model, cfg: EnvConfig, q0: jnp.ndarray, nq: int, nv: int) -> Tuple[EnvResetFn, EnvStepFn, EnvResetFn, EnvStepFn]:
    """
    Creates JIT-compiled environment functions (reset, step) for the given system.
    """
    dt = sys.opt.timestep # Model timestep. In reference: 1/300 * 5 frameskip? 
    # Reference: model.opt.timestep = 1/300, frameskip=5 => dt = 1/60.
    # User model timestep is 0.005. 
    # If we want to match reference physics, we might need to adjust, but user said "custom_walker.py" uses humanoid.xml path joined. 
    # I should assume sys.opt.timestep is correct or close enough.
    # Reference frameskip is 5. MJX step usually does 1 step. 
    # I'll implement frameskip loop in single_step.
    frameskip = 1 # MJX usually runs at control freq. If user wants frameskip, we do loop.
    # Reference: self.model.opt.timestep = 1/300 (~0.0033). frameskip=5. dt=0.0166.
    # User humanoid.xml has timestep="0.005".
    # PPO config rollout_length=128.
    # I'll stick to simple 1 step for now unless I see explicit frameskip request in config? 
    # Not in `env` config. I'll assume 1 step = 1 physics step for MJX usually, but for fair comparison...
    # Reference uses frameskip 5. I should probably do `for _ in range(5): mjx.step` or iterate.
    # But for now, let's just do single step to avoid slowdown, or maybe 1 is enough if timestep is 0.005 (200Hz).
    # Reference effective dt = 0.016 (60Hz). 0.005 is 200Hz.
    # I'll just use 1 step per action.

    # Action Permutations for Symmetry
    nu = sys.nu
    act_perm = jnp.arange(nu)
    act_sign = jnp.ones(nu)
    
    # Obs Permutations
    # Obs size: 1 (height) + 3 (rpy) + (nq-7) (joints) + nv (vels) + 2 (target)
    # nq=28, nv=27. Obs dim = 1 + 3 + 21 + 27 + 2 = 54. matches reference.
    obs_dim = 1 + 3 + (nq - 7) + nv + 2
    obs_perm = jnp.arange(obs_dim)
    obs_sign = jnp.ones(obs_dim)
    
    # Build Permutations from Config
    if cfg.random_flip:
        # Action
        r_idxs = jnp.array(cfg.flip_action_right)
        l_idxs = jnp.array(cfg.flip_action_left)
        act_perm = act_perm.at[r_idxs].set(l_idxs)
        act_perm = act_perm.at[l_idxs].set(r_idxs)
        act_sign = act_sign.at[jnp.array(cfg.flip_action_sign)].set(-1.0)
        
        # Obs
        r_idxs_o = jnp.array(cfg.flip_obs_right)
        l_idxs_o = jnp.array(cfg.flip_obs_left)
        obs_perm = obs_perm.at[r_idxs_o].set(l_idxs_o)
        obs_perm = obs_perm.at[l_idxs_o].set(r_idxs_o)
        obs_sign = obs_sign.at[jnp.array(cfg.flip_obs_sign)].set(-1.0)

    # Constants
    target_dist = cfg.target_dist
    target_threshold = cfg.target_threshold # 0.15
    initial_vel_max = cfg.initial_velocity_max
    stance_time_reward_weight = cfg.stance_time_reward_weight
    
    # Touch sensor IDs for stance detection
    touch_right_id = cfg.touch_sensor_right_id
    touch_left_id = cfg.touch_sensor_left_id
    
    # Stance state enumeration (matching FootContactState)
    # DOUBLE=0, RIGHT=1, LEFT=2, FLY=3
    
    @jit
    def get_stance_state(sensor_data: jax.Array) -> jax.Array:
        """Determine foot contact state from touch sensor data."""
        right_contact = sensor_data[touch_right_id] > 0.0
        left_contact = sensor_data[touch_left_id] > 0.0
        
        # Match reference: DOUBLE=0, RIGHT=1, LEFT=2, FLY=3
        stance = jnp.where(
            right_contact & left_contact, 0.0,  # DOUBLE
            jnp.where(
                right_contact & ~left_contact, 1.0,  # RIGHT
                jnp.where(
                    ~right_contact & left_contact, 2.0,  # LEFT
                    3.0  # FLY
                )
            )
        )
        return stance

    @jit
    def single_pipeline_init(q: jax.Array, qd: jax.Array) -> EnvState:
        d = mjx.make_data(sys)
        d = d.replace(qpos=q, qvel=qd)
        d = mjx.forward(sys, d)
        return d

    @jit
    def single_reset(key: jax.Array) -> Tuple[EnvState, Obs]:
        k1, k2, k3, k4 = random.split(key, 4)
        qpos = jnp.array(q0)
        qvel = jnp.zeros(nv, dtype=jnp.float32)
        
        # Init Aux
        # [flip, tx, ty, tz, close_count, stance_state, stance_last_change_time, last_pot]
        flip = jnp.where(random.bernoulli(k3, 0.5), 1.0, 0.0) if cfg.random_flip else 0.0
        
        # Random Initial Pose
        # Hinge joint start = 7
        noise_pos = random.uniform(k1, (nq-7,)) * 2.0 - 1.0
        qpos = qpos.at[7:].add(cfg.random_joint_noise * noise_pos)
        
        noise_vel = random.uniform(k2, (nv,)) * 2.0 - 1.0
        qvel = qvel + (cfg.random_vel_noise * noise_vel)
        
        d = single_pipeline_init(qpos, qvel)
        
        # Initial Target
        body_pos = d.xpos[cfg.pelvis_body_id]
        tx = body_pos[0] + cfg.target_dist
        ty = body_pos[1]
        tz = body_pos[2]
        
        # Apply Initial Velocity Toward Target (matching reference)
        if initial_vel_max > 0.0:
            dx = tx - body_pos[0]
            dy = ty - body_pos[1]
            dist_xy = jnp.sqrt(dx**2 + dy**2)
            # Avoid division by zero
            vel_magnitude = random.uniform(k4, minval=0.0, maxval=initial_vel_max)
            vx = jnp.where(dist_xy > 1e-6, vel_magnitude * dx / dist_xy, 0.0)
            vy = jnp.where(dist_xy > 1e-6, vel_magnitude * dy / dist_xy, 0.0)
            # Apply to root joint velocities (first 2 components of qvel)
            qvel = qvel.at[0].set(vx)
            qvel = qvel.at[1].set(vy)
            # Re-initialize with new velocity
            d = single_pipeline_init(qpos, qvel)
            # Update body_pos after velocity application
            body_pos = d.xpos[cfg.pelvis_body_id]
        
        # Initial Pot
        dx_p = tx - body_pos[0]
        dy_p = ty - body_pos[1]
        dist_p = jnp.sqrt(dx_p**2 + dy_p**2)
        head_pos = d.xpos[cfg.head_body_id]
        dx_h = tx - head_pos[0]
        dy_h = ty - head_pos[1]
        dist_h = jnp.sqrt(dx_h**2 + dy_h**2)
        dist = jnp.maximum(dist_p, dist_h)
        
        last_pot = -dist / sys.opt.timestep
        
        # Initialize stance state
        initial_stance = get_stance_state(d.sensordata)
        
        # Aux: [flip, tx, ty, tz, close_count, stance_state, stance_last_change_time, last_pot]
        aux = jnp.array([flip, tx, ty, tz, 0.0, initial_stance, d.time, last_pot]) 
        
        # Obs construction
        height = body_pos[2]
        
        # RPY
        q = d.xquat[cfg.pelvis_body_id]
        w, x, y, z = q[0], q[1], q[2], q[3]
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = jnp.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (w * y - z * x)
        pitch = jnp.arcsin(jnp.clip(sinp, -1.0, 1.0))
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = jnp.arctan2(siny_cosp, cosy_cosp)
        rpy = (roll, pitch, yaw)
        
        # Target Features for Obs
        angle = jnp.arctan2(dy_p, dx_p) - yaw
        soft = dist / (1.0 + jnp.abs(dist))
        tgt_feat = jnp.array([soft * jnp.sin(angle), soft * jnp.cos(angle)])
        
        qvel_local = get_body_velocities_local(d.qvel, q)
        joint_pos = d.qpos[7:]
        
        obs = compute_obs(flip, height, rpy, joint_pos, qvel_local, tgt_feat)
        
        return (d, aux), obs

    @jit
    def get_body_velocities(d):
        # qvel 0-5 are root (freejoint)
        # 0-2: global linear vel
        # 3-5: global angular vel (for freejoint, qvel is usually local or global depending on impl, but MJX standard freejoint qvel is in local body frame? No, typically global. Let's check constraints.)
        # Reference uses mj_objectVelocity to get global, then rotates. 
        # For standard MuJoCo freejoint: qvel is in local frame? 
        # "The 6 DOFs of the free joint are expressed in the global frame... wait.
        # MuJoCo xml documentation: "free: 3 positions, 4 quaternions. 3 linear vel, 3 angular vel."
        # "velocities are expressed in the local frame of the body for standard joints, but for free joint..."
        # Actually most RL envs assume qvel[:3] is global lin vel.
        # Let's implement rotation assuming qvel is global.
        
        lin_vel_global = d.qvel[:3]
        ang_vel_global = d.qvel[3:6]
        
        # Get quat
        q = get_pelvis_quat(d)
        w, x, y, z = q[0], q[1], q[2], q[3]
        
        # Rotation Matrix (Body to World)
        # R = [ 1-2y^2-2z^2,   2xy-2wz,      2xz+2wy
        #       2xy+2wz,       1-2x^2-2z^2,  2yz-2wx
        #       2xz-2wy,       2yz+2wx,      1-2x^2-2y^2 ]
        
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        
        r00 = 1.0 - 2.0*(yy + zz)
        r01 = 2.0*(xy - wz)
        r02 = 2.0*(xz + wy)
        
        r10 = 2.0*(xy + wz)
        r11 = 1.0 - 2.0*(xx + zz)
        r12 = 2.0*(yz - wx)
        
        r20 = 2.0*(xz - wy)
        r21 = 2.0*(yz + wx)
        r22 = 1.0 - 2.0*(xx + yy)
        
        # We want World to Body (R.T)
        # lin_body = R.T @ lin_global
        # manual matmul
        
        lx, ly, lz = lin_vel_global[0], lin_vel_global[1], lin_vel_global[2]
        lbx = r00*lx + r10*ly + r20*lz
        lby = r01*lx + r11*ly + r21*lz
        lbz = r02*lx + r12*ly + r22*lz
        
        ax, ay, az = ang_vel_global[0], ang_vel_global[1], ang_vel_global[2]
        abx = r00*ax + r10*ay + r20*az
        aby = r01*ax + r11*ay + r21*az
        abz = r02*ax + r12*ay + r22*az
        
        # Joint velocities (hinges) are already local
        joint_vel = d.qvel[6:]
        
        return jnp.concatenate([
            jnp.array([lbx, lby, lbz]),
            jnp.array([abx, aby, abz]),
            joint_vel
        ])

    @jit
    def get_body_velocities_local(qvel, q):
        # Local helper to avoid re-fetching qvel/quat
        lin_vel_global = qvel[:3]
        ang_vel_global = qvel[3:6]
        
        w, x, y, z = q[0], q[1], q[2], q[3]
        
        # Rotation Matrix (Body to World)
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        
        r00 = 1.0 - 2.0*(yy + zz)
        r01 = 2.0*(xy - wz)
        r02 = 2.0*(xz + wy)
        
        r10 = 2.0*(xy + wz)
        r11 = 1.0 - 2.0*(xx + zz)
        r12 = 2.0*(yz - wx)
        
        r20 = 2.0*(xz - wy)
        r21 = 2.0*(yz + wx)
        r22 = 1.0 - 2.0*(xx + yy)
        
        lx, ly, lz = lin_vel_global[0], lin_vel_global[1], lin_vel_global[2]
        lbx = r00*lx + r10*ly + r20*lz
        lby = r01*lx + r11*ly + r21*lz
        lbz = r02*lx + r12*ly + r22*lz
        
        ax, ay, az = ang_vel_global[0], ang_vel_global[1], ang_vel_global[2]
        abx = r00*ax + r10*ay + r20*az
        aby = r01*ax + r11*ay + r21*az
        abz = r02*ax + r12*ay + r22*az
        
        joint_vel = qvel[6:]
        
        return jnp.concatenate([
            jnp.array([lbx, lby, lbz]),
            jnp.array([abx, aby, abz]),
            joint_vel
        ])

    @jit
    def compute_obs(aux_flip, height, rpy, joint_pos, qvel_local, tgt_feat):
        roll, pitch, yaw = rpy
        
        obs_raw = jnp.concatenate([
            jnp.array([height, roll, pitch, yaw]),
            joint_pos,
            qvel_local,
            tgt_feat
        ])
        
        # Flip Obs
        obs_flipped = obs_raw[obs_perm] * obs_sign
        obs_final = jnp.where(aux_flip > 0.5, obs_flipped, obs_raw)
        return obs_final

    @jit
    def single_step(state: EnvState, action: Action) -> Tuple[EnvState, Obs, Reward, Done]:
        d, aux = state
        flip = aux[0]
        
        # Flip Action
        act_flipped = action[act_perm] * act_sign
        act_phys = jnp.where(flip > 0.5, act_flipped, action)
        
        # Apply action scale and clip
        act_phys = act_phys * cfg.action_scale
        act_phys = jnp.clip(act_phys, -1.0, 1.0)
        
        # Step
        d = d.replace(ctrl=act_phys)
        d = mjx.step(sys, d) 
        
        # --- Shared Computations ---
        # USE DYNAMIC IDs
        head_pos = d.xpos[cfg.head_body_id]
        body_pos = d.xpos[cfg.pelvis_body_id]
        height = body_pos[2]
        
        # RPY
        pelvis_quat = d.xquat[cfg.pelvis_body_id]
        # Inline RPY for reuse
        q = pelvis_quat
        w, x, y, z = q[0], q[1], q[2], q[3]
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = jnp.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (w * y - z * x)
        pitch = jnp.arcsin(jnp.clip(sinp, -1.0, 1.0))
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = jnp.arctan2(siny_cosp, cosy_cosp)
        rpy = (roll, pitch, yaw)
        
        # Target Features (Pre-Advance)
        tx, ty, tz = aux[1], aux[2], aux[3]
        
        dx_p = tx - body_pos[0]
        dy_p = ty - body_pos[1]
        dist_p = jnp.sqrt(dx_p**2 + dy_p**2)
        dx_h = tx - head_pos[0]
        dy_h = ty - head_pos[1]
        dist_h = jnp.sqrt(dx_h**2 + dy_h**2)
        dist = jnp.maximum(dist_p, dist_h)
        # angle = jnp.arctan2(dy_p, dx_p) - yaw # Not needed for reward
        
        # --- Rewards ---
        dt = sys.opt.timestep
        linear_potential = -dist / dt
        last_linear_potential = aux[7]
        progress = (linear_potential - last_linear_potential) * cfg.progress_weight
        
        # Energy (Fixed: use actuated joint velocities only)
        # MJX: qvel[6:] are joint velocities, qfrc_actuator indexed by actuator
        # For humanoid with 21 hinge actuators, we compute per-joint energy
        # Reference uses actuated joints only (skips freejoint)
        joint_velocities = d.qvel[6:]  # Hinge joint velocities
        # qfrc_actuator is per-DOF force. For hinges after freejoint (6 DOF), use [6:]
        joint_forces = d.qfrc_actuator[6:]  # Forces on hinge joints
        
        power = jnp.abs(joint_forces * joint_velocities)
        energy_penalty = cfg.electricity_cost * jnp.mean(power)
        stall = jnp.square(joint_forces)
        energy_penalty += cfg.stall_torque_cost * jnp.mean(stall)
        
        # Posture (Match Reference Logic: Deadzone + Abs Penalty)
        # Pitch Safe: [-5, 10] deg -> [-0.087, 0.174] rad
        # Roll Safe: [-10, 10] deg -> [-0.174, 0.174] rad
        p_ok = (pitch > -0.087) & (pitch < 0.174)
        posture_penalty = jnp.where(p_ok, 0.0, jnp.abs(pitch))
        r_ok = (roll > -0.174) & (roll < 0.174)
        posture_penalty += jnp.where(r_ok, 0.0, jnp.abs(roll))
        posture_penalty *= cfg.posture_penalty_weight
        
        # Tall (Match Reference Logic: +1.0 if tall, -1.0 if short)
        tall_raw = jnp.where(height > cfg.tall_height_threshold, 1.0, -1.0)
        tall_bonus = cfg.tall_bonus_weight * tall_raw
        
        # Stance-Duration Reward (matching reference implementation)
        old_stance_state = aux[5]
        stance_last_change_time = aux[6]
        new_stance_state = get_stance_state(d.sensordata)
        
        stance_changed = new_stance_state != old_stance_state
        stance_duration = d.time - stance_last_change_time
        # Reward duration if changed and duration > 0.1s (matching reference)
        stance_reward = jnp.where(
            stance_changed & (stance_duration > 0.1),
            stance_time_reward_weight * stance_duration / dt,
            0.0
        )
        
        # Update stance tracking
        stance_state_updated = jnp.where(stance_changed, new_stance_state, old_stance_state)
        stance_time_updated = jnp.where(stance_changed, d.time, stance_last_change_time)
        
        # Advance Logic
        is_close = dist < target_threshold
        close_count = aux[4]
        close_count = jnp.where(is_close, close_count + 1, 0)
        target_bonus = jnp.where(is_close, 2.0, 0.0)
        
        should_advance = close_count >= cfg.stop_frames
        
        new_tx = body_pos[0] + target_dist
        new_ty = body_pos[1]
        new_tz = body_pos[2]
        
        tx = jnp.where(should_advance, new_tx, tx)
        ty = jnp.where(should_advance, new_ty, ty)
        tz = jnp.where(should_advance, new_tz, tz)
        close_count = jnp.where(should_advance, 0.0, close_count)
        
        # Target Features (Post-Advance / Final) for Obs
        dx_p2 = tx - body_pos[0]
        dy_p2 = ty - body_pos[1]
        dist_p2 = jnp.sqrt(dx_p2**2 + dy_p2**2)
        dx_h2 = tx - head_pos[0]
        dy_h2 = ty - head_pos[1]
        dist_h2 = jnp.sqrt(dx_h2**2 + dy_h2**2)
        dist_new = jnp.maximum(dist_p2, dist_h2)
        angle_new = jnp.arctan2(dy_p2, dx_p2) - yaw
        
        soft = dist_new / (1.0 + jnp.abs(dist_new))
        tgt_feat_final = jnp.array([soft * jnp.sin(angle_new), soft * jnp.cos(angle_new)])
        
        linear_potential_new = -dist_new / dt
        
        joints_penalty = 0.0
        reward = progress + target_bonus + stance_reward - energy_penalty + tall_bonus - posture_penalty - joints_penalty
        
        fallen = height < cfg.terminate_height
        done = jnp.where(fallen, 1.0, 0.0)
        reward = jnp.where(fallen, reward + cfg.terminate_reward, reward)
        
        # Update aux with stance tracking
        aux_new = jnp.array([flip, tx, ty, tz, close_count, stance_state_updated, stance_time_updated, linear_potential_new])
        
        # --- Obs Construction ---
        qvel_local = get_body_velocities_local(d.qvel, pelvis_quat)
        joint_pos = d.qpos[7:]
        
        obs = compute_obs(flip, height, rpy, joint_pos, qvel_local, tgt_feat_final)
        
        return (d, aux_new), obs, reward, done

    v_reset = jit(jax.vmap(single_reset))
    v_step = jit(jax.vmap(single_step, in_axes=(0, 0)))
    
    return single_reset, single_step, v_reset, v_step
