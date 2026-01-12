"""
Configuration definitions for MJX training using dataclasses.
Provides type-safety and better IDE support.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class BaseConfig:
    # Model
    xml_path: str = "models/humanoid_mjx.xml"
    lighten_solver: bool = False
    
    
    # Misc
    seed: int = 42
    checkpoint_every: int = 50
    log_interval: int = 10
    eval_interval: int = 50
    results_dir: str = "results"
    
    # Rendering
    save_video: bool = True
    render_fps: int = 60
    render_duration: float = 6.0
    camera_name: str = "side_view"


@dataclass
class EnvConfig:
    progress_weight: float = 1.0
    electricity_cost: float = 0.026
    stall_torque_cost: float = 0.0000023
    joints_at_limit_cost: float = 5.0
    posture_penalty_weight: float = 0.60
    tall_height_threshold: float = 0.7
    tall_bonus_weight: float = 0.0
    target_threshold: float = 0.15
    target_dist: float = 2.0
    stop_frames: int = 1
    stance_time_reward_weight: float = 0.0
    random_joint_noise: float = 0.01
    random_vel_noise: float = 0.01
    initial_velocity_max: float = 0.5
    terminate_height: float = 0.7
    terminate_reward: float = 0.0
    random_flip: bool = False
    joint_limit_force_threshold: float = 6.5
    
    # Body IDs (Resolved at runtime)
    pelvis_body_id: int = -1
    head_body_id: int = -1
    
    # Flip params (Indices)
    flip_action_right: List[int] = field(default_factory=lambda: [3, 4, 5, 6, 7, 8, 15, 16, 17])
    flip_action_left: List[int] = field(default_factory=lambda: [9, 10, 11, 12, 13, 14, 18, 19, 20])
    flip_action_sign: List[int] = field(default_factory=lambda: [0, 2])
    
    flip_obs_right: List[int] = field(default_factory=lambda: [7, 8, 9, 10, 11, 12, 19, 20, 21, 34, 35, 36, 37, 38, 39, 46, 47, 48])
    flip_obs_left: List[int] = field(default_factory=lambda: [13, 14, 15, 16, 17, 18, 22, 23, 24, 40, 41, 42, 43, 44, 45, 49, 50, 51])
    flip_obs_sign: List[int] = field(default_factory=lambda: [1, 3, 4, 6, 26, 28, 30, 31, 33, 52])


@dataclass
class APGConfig(BaseConfig):
    # Model - CRITICAL FOR MEMORY
    lighten_solver: bool = True
    
    # Network - REDUCED for memory
    hidden_size: int = 32  # Reduced from 64
    hidden_depth: int = 2
    
    # Training - ULTRA-LIGHT for 4GB VRAM
    batch_size: int = 8     # Minimal batch (was 64)
    horizon: int = 24       # Shortened horizon (was 32)
    gamma: float = 0.99
    lr: float = 5e-5        # ULTRA-conservative for Float32 stability
    total_steps: int = 8000 # Compensate for slower learning
    
    # Optimization
    normalize_observations: bool = True  # NEW: Stabilizes training


@dataclass
class PPOConfig(BaseConfig):
    # Disable lighten_solver for physics stability on Humanoid
    lighten_solver: bool = False

    # Env Config
    env_config: EnvConfig = field(default_factory=EnvConfig)

    # Network (List of (hidden_dim, activation))
    policy_hidden_layer_specs: List[Tuple[int, str]] = field(
        default_factory=lambda: [(256, "tanh"), (256, "tanh"), (256, "tanh")]
    )
    value_hidden_layer_specs: List[Tuple[int, str]] = field(
        default_factory=lambda: [(256, "tanh"), (256, "tanh"), (256, "tanh")]
    )
    
    # Training
    num_envs: int = 2048
    rollout_length: int = 128
    
    gamma: float = 0.999
    lam: float = 0.95
    
    lr_policy: float = 3e-4
    lr_value: float = 1e-3
    
    # PPO specific
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    epochs: int = 4
    minibatch_size: int = 1024
    
    # Iterations
    total_iterations: int = 1000

    @property
    def total_steps(self) -> int:
        return self.total_iterations

    @classmethod
    def from_json(cls, path: str):
        import json
        import os
        if not os.path.exists(path):
            print(f"Config file not found: {path}, using defaults")
            return cls()
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        cfg = cls()
        
        # Load PPO params
        if "ppo" in data:
            ppo_data = data["ppo"]
            for k, v in ppo_data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        
        # Load Env params
        if "env" in data:
            env_data = data["env"]
            env_cfg = cfg.env_config
            for k, v in env_data.items():
                if hasattr(env_cfg, k):
                    setattr(env_cfg, k, v)
        
        # Load Symmetry params
        if "symmetry" in data and "flip_params" in data["symmetry"]:
            fp = data["symmetry"]["flip_params"]
            if "action_index_info" in fp:
                cfg.env_config.flip_action_right = fp["action_index_info"]["right"]
                cfg.env_config.flip_action_left = fp["action_index_info"]["left"]
                cfg.env_config.flip_action_sign = fp["action_index_info"]["negative_sign"]
            if "observation_index_info" in fp:
                cfg.env_config.flip_obs_right = fp["observation_index_info"]["right"]
                cfg.env_config.flip_obs_left = fp["observation_index_info"]["left"]
                cfg.env_config.flip_obs_sign = fp["observation_index_info"]["negative_sign"]
                
        return cfg
