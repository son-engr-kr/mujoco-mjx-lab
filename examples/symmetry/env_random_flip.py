from environment.env_wrapper_base import EnvWrapperBase
from environment.flip_utils import FlipUtils

class EnvRandomFlip(EnvWrapperBase):
    def __init__(self, env_id: str = "Walker2d-v5", flip_params: dict = None, **kwargs):
        super().__init__(env_id, **kwargs)

        self._flip_flag = False
        self._flip_utils = FlipUtils(flip_params)
        self._flip_prob = 0.0
    
    def set_flip_prob(self, flip_prob: float):
        self._flip_prob = flip_prob

    def reset(self, **kwargs):
        # Use self.np_random to randomly choose True or False
        # This can be used for random flipping or other stochastic logic
        self._flip_flag = self.np_random.choice([True, False], p=[self._flip_prob, 1-self._flip_prob])
        obs, info = super().reset(**kwargs)
        if self._flip_flag and self._training_mode:
            obs = self._flip_utils.flip_observation(obs)
        return obs, info

    def step(self, action):
        if self._flip_flag and self._training_mode:
            action = self._flip_utils.flip_action(action)
        obs, reward, terminated, truncated, info = super().step(action)
        
        self._flip_flag = self.np_random.choice([True, False], p=[self._flip_prob, 1-self._flip_prob])

        if self._flip_flag and self._training_mode:
            obs = self._flip_utils.flip_observation(obs)
        return obs, reward, terminated, truncated, info