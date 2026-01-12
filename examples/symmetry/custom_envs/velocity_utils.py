"""Utility classes that generate target velocity time-series for HumanoidTargetVelocityEnv.

Each generator implements `__call__` and returns a 2-D numpy array `[vx, vy]` in the **world frame**.
They are deterministic given an RNG and do **not** mutate global state.
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Dict

__all__ = [
    "RandomGoalLowPass",
    "OUProcess",
    "CircularPath",
    "AccelLimitedRandomWalk",
    "make_velocity_generator",
]


class RandomGoalLowPass:
    """Random goal velocity with first-order low-pass smoothing."""

    def __init__(
        self,
        rng: np.random.Generator,
        *,
        dt: float,
        vmax: float = 3.0,
        alpha: float = 0.02,
        goal_interval: int = 250,
        p_stop: float = 0.1,
    ) -> None:
        self._rng = rng
        self._dt = dt
        self._vmax = vmax
        self._alpha = alpha
        self._goal_interval = goal_interval
        self._p_stop = p_stop

        self._v = np.zeros(2)
        self._goal = np.array([vmax, 0.0])  # bias forward at t=0
        self._cnt = goal_interval

    # ------------------------------------------------------------------
    def _sample_goal(self) -> np.ndarray:
        if self._rng.random() < self._p_stop:
            return np.zeros(2)
        speed = self._rng.uniform(0.0, self._vmax)
        theta = self._rng.uniform(0.0, 2 * np.pi)
        return speed * np.array([np.cos(theta), np.sin(theta)])

    def __call__(self) -> np.ndarray:
        self._cnt -= 1
        if self._cnt <= 0:
            self._goal = self._sample_goal()
            self._cnt = self._goal_interval
        self._v = (1.0 - self._alpha) * self._v + self._alpha * self._goal
        return self._v.copy()


class OUProcess:
    """Ornstein-Uhlenbeck velocity process with drifting mean."""

    def __init__(
        self,
        rng: np.random.Generator,
        *,
        dt: float,
        vmax: float = 8.0,
        theta: float = 1.0,
        sigma: float = 3.0,
        mu_interval: int = 200,
        warmup_steps: int = 125,
    ) -> None:
        self._rng = rng
        self._dt = dt
        self._vmax = vmax
        self._theta = theta
        self._sigma = sigma
        self._mu_interval = mu_interval
        self._warmup = warmup_steps

        self._step = 0
        self._cnt = mu_interval
        self._v = np.zeros(2)
        self._mu = np.array([vmax, 0.0])

    # ------------------------------------------------------------------
    def _sample_mu(self) -> np.ndarray:
        speed = self._rng.uniform(0.0, self._vmax)
        theta = self._rng.uniform(0.0, 2 * np.pi)
        return speed * np.array([np.cos(theta), np.sin(theta)])

    def __call__(self) -> np.ndarray:
        if self._step >= self._warmup:
            self._cnt -= 1
            if self._cnt <= 0:
                self._mu = self._sample_mu()
                self._cnt = self._mu_interval
        noise = self._sigma * np.sqrt(self._dt) * self._rng.standard_normal(2)
        self._v += self._theta * (self._mu - self._v) * self._dt + noise
        speed = np.linalg.norm(self._v)
        if speed > self._vmax and speed > 0.0:
            self._v = self._v / speed * self._vmax
        self._step += 1
        return self._v.copy()


class CircularPath:
    """Circular motion with random radius / speed (sampled once at init)."""

    def __init__(
        self,
        rng: np.random.Generator | None = None,
        *,
        dt: float,
        radius_range: tuple[float, float] = (3.0, 7.0),
        speed_range: tuple[float, float] = (1.0, 4.0),
    ) -> None:
        self._dt = dt
        rng = rng or np.random.default_rng()
        self._radius = rng.uniform(*radius_range)
        self._speed = rng.uniform(*speed_range)
        # random clockwise (-1) or counter-clockwise (+1) direction
        self._dir = rng.choice((-1.0, 1.0))
        # start such that initial velocity points +x (vx>0, vy≈0) → θ = -π/2
        self._theta = -np.pi / 2

    def __call__(self) -> np.ndarray:
        dtheta = self._dir * self._speed * self._dt / self._radius
        self._theta += dtheta
        vx = -self._speed * np.sin(self._theta)
        vy = self._speed * np.cos(self._theta)
        return np.array([vx, vy])


class AccelLimitedRandomWalk:
    """Random walk with bounded acceleration."""

    def __init__(
        self,
        rng: np.random.Generator,
        *,
        dt: float,
        vmax: float = 8.0,
        max_acc: float = 12.0,
        warmup_steps: int = 125,
    ) -> None:
        self._rng = rng
        self._dt = dt
        self._vmax = vmax
        self._max_acc = max_acc
        self._warmup = warmup_steps

        self._step = 0
        self._v = np.zeros(2)

    def __call__(self) -> np.ndarray:
        if self._step == 0:
            # deterministically accelerate toward +x on first step
            dir_angle = 0.0
        elif self._step < self._warmup and self._rng.random() < 0.7:
            dir_angle = 0.0
        else:
            dir_angle = self._rng.uniform(0.0, 2 * np.pi)
        acc = self._max_acc * self._rng.random() * np.array([np.cos(dir_angle), np.sin(dir_angle)])
        self._v += acc * self._dt
        speed = np.linalg.norm(self._v)
        if speed > self._vmax and speed > 0.0:
            self._v = self._v / speed * self._vmax
        self._step += 1
        return self._v.copy()


# ---------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------

GeneratorFactory = Callable[[np.random.Generator, float], Callable[[], np.ndarray]]


def make_velocity_generator(mode: str, rng: np.random.Generator, dt: float):
    """Create a velocity generator given the `mode` string."""
    mode = mode.lower()
    factories: Dict[str, Callable[[], Callable[[], np.ndarray]]] = {
        "lowpass": lambda: RandomGoalLowPass(rng, dt=dt),
        "ou": lambda: OUProcess(rng, dt=dt),
        "circle": lambda: CircularPath(rng, dt=dt),
        "accel": lambda: AccelLimitedRandomWalk(rng, dt=dt),
    }
    if mode not in factories:
        raise ValueError(f"Unsupported velocity_mode '{mode}'. Supported: {list(factories)}")
    return factories[mode]()
