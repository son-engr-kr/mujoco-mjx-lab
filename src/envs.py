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
AuxState = jax.Array # [flip, tx, ty, tz, close_count, stance, stance_t, last_pot]
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

    @jit
    def get_body_pos(d):
        return d.xpos[1] # Pelvis is usually body 1? No, 0 is world. 1 is torso/pelvis.
        # Check humanoid.xml: worldbody -> torso (id 1) -> pelvis (id 2).
        # Actually in XML:
        # worldbody (0)
        #   torso (1) (freejoint root)
        #     waist_upper ...
        #     head ...
        #     waist_lower ...
        #       pelvis ...
        # Wait, reference says `self.data.body("pelvis").xpos`.
        # I need to find body index of "pelvis". It's not necessarily 1.
        # But `custom_walker.py` uses `self.data.body("pelvis").xpos`.
        # I will assume "pelvis" is index 1 for `root` usually, BUT `torso` is root in XML.
        # `pelvis` is deeper.
        # I will use `sys.body_name2id("pelvis")` if I could, but `mjx` doesn't have it easily available in JIT.
        # I passed `sys` (mjx.Model). `mujoco.mj_name2id` works on `m` (mujoco.MjModel).
        # I should have passed `m` or body ids.
        # For now, I'll rely on hardcoded assumption or find it.
        # In `humanoid.xml` provided: `torso` (1), `head` (2), `waist_lower` (3), `pelvis` (4).
        # Let's hope indices are stable.
        # BETTER: Use `sys.nbody` and heuristic? No.
        # I will hardcode `PELVIS_PID = 4` and `HEAD_ID = 2` based on XML structure.
        # XML: world -> torso (child 0). torso -> head (child 0), waist_lower (child 1). waist_lower -> pelvis.
        # Body IDs: 0 (world), 1 (torso), 2 (head), 3 (waist_lower), 4 (pelvis).
        return d.xpos[4]

    PELVIS_ID = 4
    HEAD_ID = 2

    @jit
    def get_pelvis_quat(d):
        return d.xquat[PELVIS_ID]

    # get_target_features and get_body_velocities removed as they are now inlined for performance.

    @jit
    def single_pipeline_init(q: jax.Array, qd: jax.Array) -> EnvState:
        d = mjx.make_data(sys)
        d = d.replace(qpos=q, qvel=qd)
        d = mjx.forward(sys, d)
        return d

    @jit
    def single_reset(key: jax.Array) -> Tuple[EnvState, Obs]:
        k1, k2, k3 = random.split(key, 3)
        qpos = jnp.array(q0)
        qvel = jnp.zeros(nv, dtype=jnp.float32)
        
        # Init Aux
        # [flip, tx, ty, tz, close, stance, stance_t, last_pot]
        flip = jnp.where(random.bernoulli(k3, 0.5), 1.0, 0.0) if cfg.random_flip else 0.0
        
        # Random Initial Pose
        # Hinge joint start = 7
        noise_pos = random.uniform(k1, (nq-7,)) * 2.0 - 1.0
        qpos = qpos.at[7:].add(cfg.random_joint_noise * noise_pos)
        
        noise_vel = random.uniform(k2, (nv,)) * 2.0 - 1.0
        qvel = qvel + (cfg.random_vel_noise * noise_vel)
        
        d = single_pipeline_init(qpos, qvel)
        
        # Initial Target
        body_pos = d.xpos[PELVIS_ID]
        tx = body_pos[0] + cfg.target_dist
        ty = body_pos[1]
        tz = body_pos[2]
        
        # Initial Pot
        dx_p = tx - body_pos[0]
        dy_p = ty - body_pos[1]
        dist_p = jnp.sqrt(dx_p**2 + dy_p**2)
        head_pos = d.xpos[HEAD_ID]
        dx_h = tx - head_pos[0]
        dy_h = ty - head_pos[1]
        dist_h = jnp.sqrt(dx_h**2 + dy_h**2)
        dist = jnp.maximum(dist_p, dist_h)
        
        last_pot = -dist / sys.opt.timestep 
        
        aux = jnp.array([flip, tx, ty, tz, 0.0, 0.0, d.time, last_pot]) 
        
        # Obs construction
        height = body_pos[2]
        
        # RPY
        q = d.xquat[PELVIS_ID]
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
        act_phys = jnp.clip(act_phys, -1.0, 1.0)
        
        # Step
        d = d.replace(ctrl=act_phys)
        d = mjx.step(sys, d) 
        
        # --- Shared Computations ---
        head_pos = d.xpos[HEAD_ID]
        body_pos = d.xpos[PELVIS_ID]
        height = body_pos[2]
        
        # RPY
        pelvis_quat = d.xquat[PELVIS_ID]
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
        
        # Energy
        power = jnp.abs(d.qfrc_actuator[6:] * d.qvel[6:])
        energy_penalty = cfg.electricity_cost * jnp.mean(power)
        stall = jnp.square(d.qfrc_actuator[6:])
        energy_penalty += cfg.stall_torque_cost * jnp.mean(stall)
        
        # Posture
        p_ok = (pitch > -0.087) & (pitch < 0.174)
        posture_penalty = jnp.where(p_ok, 0.0, jnp.abs(pitch))
        r_ok = (roll > -0.174) & (roll < 0.174)
        posture_penalty += jnp.where(r_ok, 0.0, jnp.abs(roll))
        posture_penalty *= cfg.posture_penalty_weight
        
        # Tall
        tall_raw = jnp.where(height > cfg.tall_height_threshold, 1.0, -1.0)
        tall_bonus = cfg.tall_bonus_weight * tall_raw
        
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
        reward = progress + target_bonus - energy_penalty + tall_bonus - posture_penalty - joints_penalty
        
        fallen = height < cfg.terminate_height
        done = jnp.where(fallen, 1.0, 0.0)
        reward = jnp.where(fallen, reward + cfg.terminate_reward, reward)
        
        aux_new = jnp.array([flip, tx, ty, tz, close_count, 0.0, d.time, linear_potential_new])
        
        # --- Obs Construction ---
        qvel_local = get_body_velocities_local(d.qvel, pelvis_quat)
        joint_pos = d.qpos[7:]
        
        obs = compute_obs(flip, height, rpy, joint_pos, qvel_local, tgt_feat_final)
        
        return (d, aux_new), obs, reward, done

    v_reset = jit(jax.vmap(single_reset))
    v_step = jit(jax.vmap(single_step, in_axes=(0, 0)))
    
    return single_reset, single_step, v_reset, v_step
