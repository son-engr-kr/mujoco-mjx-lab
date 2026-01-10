"""
Neural network definitions using Pure JAX (No Flax dependency).
This ensures compatibility with existing JAX installations.
"""
import jax
import jax.numpy as jnp
from jax import random, jit
from typing import Sequence, Tuple, Any

ACTIVATIONS = {
    "tanh": jnp.tanh,
    "relu": jax.nn.relu,
    "elu": jax.nn.elu,
    "swish": jax.nn.swish,
    "silu": jax.nn.swish,
    "gelu": jax.nn.gelu,
    "linear": lambda x: x,
    "none": lambda x: x,
}

class MLP:
    """Flexible MLP with configurable layers and activations."""
    
    def __init__(self, layer_specs: Sequence[Tuple[int, str]]):
        """
        Args:
            layer_specs: List of (hidden_dim, activation_name) tuples.
                         e.g. [(256, 'tanh'), (256, 'relu')]
        """
        self.layer_specs = layer_specs

    def init(self, rng, input_shape):
        """Initialize parameters."""
        params = []
        k1, k2 = random.split(rng)
        
        if isinstance(input_shape, tuple):
            in_dim = input_shape[-1]
        else:
            in_dim = input_shape
            
        current_dim = in_dim
        
        for i, (feat, act_name) in enumerate(self.layer_specs):
            key_w, key_b = random.split(random.fold_in(k1, i))
            # Xavier/Glorot initialization
            scale = jnp.sqrt(2.0 / (current_dim + feat))
            w = random.normal(key_w, (current_dim, feat)) * scale
            b = jnp.zeros((feat,))
            params.append((w, b))
            current_dim = feat
            
        return params

    def apply(self, params, x):
        """Forward pass."""
        for (w, b), (_, act_name) in zip(params, self.layer_specs):
            x = x @ w + b
            act_fn = ACTIVATIONS.get(act_name.lower(), jnp.tanh)
            x = act_fn(x)
        return x

class APGPolicy:
    """APG Policy: MLP + Tanh squashing."""
    def __init__(self, action_dim, hidden_dim=64, hidden_depth=2, hidden_layer_specs=None):
        if hidden_layer_specs is not None:
            layers = list(hidden_layer_specs)
        else:
            layers = [(hidden_dim, 'tanh')] * hidden_depth
            
        # Append action head (linear body output -> tanh in apply)
        layers.append((action_dim, 'linear'))
        self.mlp = MLP(layers)
        
    def init(self, rng, input_shape):
        return self.mlp.init(rng, input_shape)
        
    def apply(self, params, x):
        x = self.mlp.apply(params, x)
        return jnp.tanh(x) # Squashing

class GaussianPolicy:
    """PPO Policy: MLP mean + Learnable LogStd."""
    def __init__(self, action_dim, hidden_dim=128, hidden_depth=2, hidden_layer_specs=None):
        if hidden_layer_specs is not None:
            layers = list(hidden_layer_specs)
        else:
            layers = [(hidden_dim, 'tanh')] * hidden_depth
            
        # Append action head
        layers.append((action_dim, 'linear'))
        self.mlp = MLP(layers)
        self.action_dim = action_dim
        
    def init(self, rng, input_shape):
        mlp_params = self.mlp.init(rng, input_shape)
        log_std = jnp.zeros((self.action_dim,), dtype=jnp.float32)
        return {"mlp": mlp_params, "log_std": log_std}
        
    def apply(self, params, x):
        mean = self.mlp.apply(params["mlp"], x)
        mean = jnp.tanh(mean) # Often PPO uses tanh for mean to keep it in range
        
        # Clip log_std for numerical stability
        log_std = jnp.clip(params["log_std"], -20.0, 2.0)
        
        return mean, log_std

class ValueNet:
    """Value Function: MLP -> Scalar."""
    def __init__(self, hidden_dim=128, hidden_depth=2, hidden_layer_specs=None):
        if hidden_layer_specs is not None:
            layers = list(hidden_layer_specs)
        else:
            layers = [(hidden_dim, 'tanh')] * hidden_depth
            
        # Append value head
        layers.append((1, 'linear'))
        self.mlp = MLP(layers)
        
    def init(self, rng, input_shape):
        return self.mlp.init(rng, input_shape)
        
    def apply(self, params, x):
        v = self.mlp.apply(params, x)
        return v.squeeze(-1)
