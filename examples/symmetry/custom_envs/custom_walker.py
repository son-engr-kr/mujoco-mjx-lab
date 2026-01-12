import gymnasium as gym
import numpy as np
from gymnasium import spaces
import mujoco
from mujoco import viewer
import os
from enum import IntEnum
class FootContactState(IntEnum):
    DOUBLE = 0
    RIGHT = 1
    LEFT = 2
    FLY = 3


class CustomWalkerEnv(gym.Env):
    """Test environment using MuJoCo's humanoid.xml model directly."""
    
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 60}
    
    def __init__(self, *, render_mode=None, **kwargs):
        # Load the humanoid model
        model_path = os.path.join("src", "environment", "models", "mujoco", "humanoid.xml")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        # model_path = os.path.join(mujoco., "humanoid", "humanoid.xml")
        # self.model = mujoco.MjModel.from_xml_path(model_path)
        # self.data = mujoco.MjData(self.model)
        
        # Set up observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            # height, roll, pitch, yaw, joint_pos, qvel, target_features(2)
            shape=(1+3+(self.model.nq-7) + self.model.nv + 2,), 
            dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.model.nu,), 
            dtype=np.float32
        )
        
        self.render_mode = render_mode
        self.viewer = None
        
        # Walker3D-like reward shaping parameters (overridable via kwargs)
        self.progress_weight = float(kwargs.get("progress_weight", 1.0))
        self.electricity_cost = float(kwargs.get("electricity_cost", 4.5))
        self.stall_torque_cost = float(kwargs.get("stall_torque_cost", 0.225))
        self.joints_at_limit_cost = float(kwargs.get("joints_at_limit_cost", 0.0))
        self.posture_penalty_weight = float(kwargs.get("posture_penalty_weight", 1.0))
        self.tall_height_threshold = float(kwargs.get("tall_height_threshold", 0.9))
        self.tall_bonus_weight = float(kwargs.get("tall_bonus_weight", 1.0))
        self.target_threshold = float(kwargs.get("target_threshold", 0.15))
        self.target_dist = float(kwargs.get("target_dist", 4.0))
        self.stop_frames = int(kwargs.get("stop_frames", 30))
        # Stance-duration bonus on transitions
        self.stance_time_reward_weight = float(kwargs.get("stance_time_reward_weight", 1.0))
        # Random initial pose parameters
        self.random_joint_noise = float(kwargs.get("random_joint_noise", 0.1))
        self.random_vel_noise = float(kwargs.get("random_vel_noise", 0.1))
        self.initial_velocity_max = float(kwargs.get("initial_velocity_max", 0.0))

        # Joint-limit sensing configuration (sensor-first, kinematic fallback)
        self.joint_limit_force_threshold = float(kwargs.get("joint_limit_force_threshold", 1e-6))
        # Optional list of MuJoCo sensor names that report joint-limit forces
        # Example: ["hip_right_limitfrc", "knee_left_limitfrc", ...]
        self.limit_sensor_names = list(kwargs.get("limit_sensor_names", []))

        self._terminate_height = float(kwargs.get("terminate_height", 1.0))
        self.terminate_reward = float(kwargs.get("terminate_reward", -100.0))
        # Sensor ids for touch-based stance detection
        self._sensor_touch_right = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "touch_foot_right"))
        self._sensor_touch_left = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "touch_foot_left"))
        # Internal target state
        self.walk_target = np.zeros(3, dtype=np.float64)
        self.close_count = 0
        self._last_linear_potential = 0.0
        self.model.opt.timestep = 1/300
        self.frameskip = 5
        self.dt = self.frameskip * self.model.opt.timestep

        # Reset to initial state
        self.reset_model()
    
    def reset_model(self):
        """Reset the model to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        
        # Randomize initial joint poses (hinges) and optionally mirror left/right
        self._apply_random_initial_pose()
        
        mujoco.mj_forward(self.model, self.data)

        # Initialize target ahead of body and last potential
        self._advance_target()

        # Apply initial velocity towards target
        if self.initial_velocity_max > 0.0:
            body_pos = self.get_body_pos()
            dx = float(self.walk_target[0] - body_pos[0])
            dy = float(self.walk_target[1] - body_pos[1])
            dist = float((dx * dx + dy * dy) ** 0.5)
            if dist > 1e-6:
                vel_magnitude = float(self.np_random.uniform(0.0, self.initial_velocity_max))
                vx = vel_magnitude * dx / dist
                vy = vel_magnitude * dy / dist
                # Set root joint velocities (freejoint has 6 DOF: linear[3] + angular[3])
                root_joint = self.data.joint("root")
                root_joint.qvel[0] = vx  # linear velocity x
                root_joint.qvel[1] = vy  # linear velocity y

        # Initialize stance state and timer
        self._stance_state = self._get_stance_state()
        self._stance_last_change_time = float(self.data.time)

        return self._get_obs()
    
    def _get_obs(self):
        """Get current observation."""

        '''
        pelvis position
        pelvis orientation
        spinal
        right leg
        left leg
        arms
      -->
      <key name="squat"
           qpos="0 0 0.596
                 0.988015 0 0.154359 0
                 0 0.4 0
                 -0.25 -0.5 -2.5 -2.65 -0.8 0.56
                 -0.25 -0.5 -2.5 -2.65 -0.8 0.56
                 0 0 0 0 0 0"/>
        '''
        height = self.get_body_pos()[2]
        q = self._get_pelvis_quat()
        roll, pitch, yaw = self._get_pelvis_rpy()
        joint_pos = self.data.qpos[7:]

        qvel_obs = self._get_body_velocities()

        # Target features (egocentric): softsign(dist)*[sin(theta), cos(theta)]
        _, _, tgt_feat = self._compute_target_features()

        return np.concatenate(([height, roll, pitch, yaw], joint_pos, qvel_obs, tgt_feat))
    
    def step(self, action):
        """Execute one step in the environment."""
        # Apply action
        self.data.ctrl[:] = np.clip(action, -1.0, 1.0)
        
        # Step simulation
        for _ in range(self.frameskip):
            mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()

        # Compute Walker3D-like reward
        dist, ang, _tgt_feat = self._compute_target_features()
        linear_potential = -dist / self.dt
        progress = linear_potential - self._last_linear_potential
        progress = self.progress_weight * progress
        self._last_linear_potential = linear_potential

        energy_penalty = self._compute_energy_penalty()
        roll, pitch, _yaw = self._get_pelvis_rpy()

        posture_penalty = 0.0
        if not (-np.deg2rad(5) < pitch < np.deg2rad(10)):
            posture_penalty = abs(pitch)
        if not (-np.deg2rad(10) < roll < np.deg2rad(10)):
            posture_penalty += abs(roll)

        body_height = float(self.get_body_pos()[2])
        # Upright bonus only if sufficient forward progress; otherwise treat as neutral/penalty
        tall_bonus_raw = 1.0 if body_height > self.tall_height_threshold else -1.0
        tall_bonus = self.tall_bonus_weight * tall_bonus_raw

        target_bonus = 0.0
        if dist < self.target_threshold:
            self.close_count += 1
            target_bonus = 2.0
        if self.close_count >= self.stop_frames:
            self._advance_target()
            self.close_count = 0

        joints_at_limit = self._compute_joints_at_limit()
        joints_penalty = self.joints_at_limit_cost * float(joints_at_limit)

        reward = progress + target_bonus - energy_penalty + tall_bonus - self.posture_penalty_weight * posture_penalty - joints_penalty

        # Stance-duration reward: on stance transition, reward time spent in previous stance
        new_stance = self._get_stance_state()
        if new_stance != self._stance_state:
            now_t = float(self.data.time)
            duration = (now_t - self._stance_last_change_time) 
            if duration > 0.1:
                reward += self.stance_time_reward_weight * duration / self.dt
            self._stance_state = new_stance
            self._stance_last_change_time = now_t

        terminated = self._is_terminated()
        if terminated:
            reward += self.terminate_reward
        truncated = False
        info = {
            "reward_progress": progress,
            "reward_target_bonus": target_bonus,
            "penalty_energy": energy_penalty,
            "penalty_posture": posture_penalty,
            "bonus_tall": tall_bonus,
            "bonus_tall_raw": tall_bonus_raw,
            "tall_bonus_weight": self.tall_bonus_weight,
            "distance_to_target": dist,
            "angle_to_target": ang,
            "walk_target": self.walk_target.copy(),
            "joints_at_limit": float(joints_at_limit),
            "stance_state": self._stance_state,
        }
        
        return obs, reward, terminated, truncated, info

    def _is_terminated(self):
        """Check if episode should terminate."""
        height = self.get_body_pos()[2]
        return bool(height < self._terminate_height)  # Terminate if humanoid falls
    
    def reset(self, *, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)
        obs = self.reset_model()
        info = {}
        return obs, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            if self.viewer is None:
                self.viewer = mujoco.Renderer(self.model, height=480, width=640)
                self.viewer._scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
                self.viewer._scene_option.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = 1
                
            cam_id = self.model.camera('track').id
            self.viewer.update_scene(self.data, camera=cam_id)
            return self.viewer.render()
    
    def close(self):
        """Close the environment."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    # -------------------------
    # Target / potential helpers
    # -------------------------

    def _get_pelvis_quat(self):
        """Get pelvis body quaternion (w, x, y, z)."""
        return self.data.body("pelvis").xquat.copy()
    
    def _get_pelvis_rpy(self):
        """Get pelvis body roll, pitch, yaw from quaternion."""
        q = self._get_pelvis_quat()
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        if sinp <= -1.0:
            pitch = -np.pi / 2
        elif sinp >= 1.0:
            pitch = np.pi / 2
        else:
            pitch = np.arcsin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return float(roll), float(pitch), float(yaw)
    
    def _get_body_velocities(self):
        """Get pelvis body velocities using mj_objectVelocity and convert to body-local frame."""
        # Get pelvis body ID
        pelvis_id = self.model.body("pelvis").id
        
        # Get pelvis velocities using mj_objectVelocity
        pelvis_vel = np.zeros(6, dtype=np.float64)  # [ang_vel(3), lin_vel(3)]
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, pelvis_id, pelvis_vel, 0)
        pelvis_ang_vel = pelvis_vel[:3]  # angular velocity
        pelvis_lin_vel = pelvis_vel[3:]  # linear velocity
        
        # Get pelvis quaternion for rotation matrix
        q = self._get_pelvis_quat()
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        ww = w*w; xx = x*x; yy = y*y; zz = z*z
        wx = w*x; wy = w*y; wz = w*z
        xy = x*y; xz = x*z; yz = y*z
        R = np.array([
            [ww + xx - yy - zz, 2*(xy - wz),       2*(xz + wy)      ],
            [2*(xy + wz),       ww - xx + yy - zz, 2*(yz - wx)      ],
            [2*(xz - wy),       2*(yz + wx),       ww - xx - yy + zz]
        ], dtype=float)

        # body_from_world rotation
        R_T = R.T
        lin_body = R_T.dot(pelvis_lin_vel)
        ang_body = R_T.dot(pelvis_ang_vel)

        # Get joint velocities (excluding root joint)
        qvel = self.data.qvel
        joint_velocities = qvel[6:]  # Skip root joint velocities

        return np.concatenate((lin_body, ang_body, joint_velocities))
    def get_body_pos(self):
        return self.data.body("pelvis").xpos.copy()
    def _compute_target_features(self):
        body_pos = self.get_body_pos()
        head_pos = self.data.body("head").xpos.copy()  # Get head position
        
        _, _, yaw = self._get_pelvis_rpy()
        
        # Compute distance from pelvis to target
        dx_pelvis = float(self.walk_target[0] - body_pos[0])
        dy_pelvis = float(self.walk_target[1] - body_pos[1])
        dist_pelvis = float((dx_pelvis * dx_pelvis + dy_pelvis * dy_pelvis) ** 0.5)
        
        # Compute distance from head to target
        dx_head = float(self.walk_target[0] - head_pos[0])
        dy_head = float(self.walk_target[1] - head_pos[1])
        dist_head = float((dx_head * dx_head + dy_head * dy_head) ** 0.5)
        
        # Use the maximum distance (farther body part) to prevent cheating by extending only one part
        dist = max(dist_pelvis, dist_head)
        
        # Angle is still computed from pelvis for consistency
        angle = float(np.arctan2(dy_pelvis, dx_pelvis) - yaw)
        soft = dist / (1.0 + abs(dist))
        tgt_feat = np.array([soft * np.sin(angle), soft * np.cos(angle)], dtype=np.float64)
        return dist, angle, tgt_feat

    def _advance_target(self):
        body_pos = self.get_body_pos()
        self.walk_target[0] = float(body_pos[0]) + self.target_dist
        self.walk_target[1] = float(body_pos[1])
        self.walk_target[2] = float(body_pos[2])

        dist, ang, _tgt_feat = self._compute_target_features()
        linear_potential = -dist / self.dt
        self._last_linear_potential = linear_potential

    def _compute_energy_penalty(self):
        # Compute energy over actuated hinge joints using joint accessors
        trnid = self.model.actuator_trnid
        nu = int(self.model.nu)
        # Collect unique actuated hinge joints
        actuated_joint_ids = []
        seen = set()
        for i in range(nu):
            jid = int(trnid[i, 0]) if np.ndim(trnid) == 2 else int(trnid[2 * i + 0])
            if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE and jid not in seen:
                seen.add(jid)
                actuated_joint_ids.append(jid)

        if len(actuated_joint_ids) == 0:
            return 0.0

        power_vals = []
        torque_vals = []
        for jid in actuated_joint_ids:
            jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            jdat = self.data.joint(jname)
            tau = float(jdat.qfrc_actuator[0])
            qdot = float(jdat.qvel[0])
            power_vals.append(abs(tau * qdot))
            torque_vals.append(tau)

        electricity = self.electricity_cost * float(np.mean(power_vals))
        stall = self.stall_torque_cost * float(np.mean(np.square(torque_vals)))
        return float(electricity + stall)

    def _compute_joints_at_limit(self):
        # Prefer dedicated sensors if provided
        if len(self.limit_sensor_names) > 0:
            active = 0
            total = 0
            for name in self.limit_sensor_names:
                # If a sensor name is invalid, an exception will surface (prefer explicit failure)
                val = float(self.data.sensor(name).data[0])
                total += 1
                if abs(val) > self.joint_limit_force_threshold:
                    active += 1
            if total > 0:
                return active / total

        return 0.0

    # -------------------------
    # Stance helpers
    # -------------------------
    def _foot_in_contact(self, which: str):
        # Use touch sensors directly (site contact magnitude)
        name = "touch_foot_right" if which == "right" else "touch_foot_left"
        val = float(self.data.sensor(name).data[0])
        return bool(val > 0.0)

    def _get_stance_state(self):
        right = self._foot_in_contact("right")
        left = self._foot_in_contact("left")
        if right and left:
            return FootContactState.DOUBLE
        if right and not left:
            return FootContactState.RIGHT
        if left and not right:
            return FootContactState.LEFT
        return FootContactState.FLY

    def _apply_random_initial_pose(self):
        """Apply random initial pose with noise."""
        qpos = self.model.keyframe("normal_stand").qpos.copy()
        # qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        hinge_start = 7  # First 7 qpos are root (pos + quat), then hinge joints
        
        # Add small random perturbations to all hinge joint positions
        qpos[hinge_start:] += self.np_random.uniform(-self.random_joint_noise, self.random_joint_noise, size=int(self.model.nq)-hinge_start)
        
        # Add small random perturbations to all velocities
        qvel[:] += self.np_random.uniform(-self.random_vel_noise, self.random_vel_noise, size=qvel.shape)
        
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
