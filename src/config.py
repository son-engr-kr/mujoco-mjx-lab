"""
Configuration definitions for MJX training using dataclasses.
Provides type-safety and better IDE support.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class BaseConfig:
    # Model
    xml_path: str = "models/humanoid.xml"
    lighten_solver: bool = False
    
    # Rendering
    render_duration: float = 6.0 # Approx 600 steps (assuming dt=0.01)
    render_fps: int = 30
    render_every: int = 50  # Render more frequently (was 200)
    
    # Misc
    seed: int = 42
    checkpoint_every: int = 50 # Checkpoint more frequently (was 200)
    results_dir: str = "results"


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

    # Network (List of (hidden_dim, activation))
    policy_hidden_layer_specs: List[Tuple[int, str]] = field(
        default_factory=lambda: [(256, "tanh"), (256, "tanh"), (256, "tanh")]
    )
    value_hidden_layer_specs: List[Tuple[int, str]] = field(
        default_factory=lambda: [(256, "tanh"), (256, "tanh"), (256, "tanh")]
    )
    
    # Training
    num_envs: int = 1024  # Reduced from 1024 for safety
    rollout_length: int = 128  # Reduced from 256
    
    gamma: float = 0.99
    lam: float = 0.95
    
    lr_policy: float = 3e-4
    lr_value: float = 1e-3
    
    # PPO specific
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    epochs: int = 4
    minibatch_size: int = 8192  # Reduced from 8192
    
    # Iterations
    total_iterations: int = 1000

    @property
    def total_steps(self) -> int:
        return self.total_iterations
