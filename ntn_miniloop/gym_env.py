from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Callable

VENDOR_PATH = Path(__file__).resolve().parent.parent / "_vendor"
if VENDOR_PATH.exists() and str(VENDOR_PATH) not in sys.path:
    sys.path.insert(0, str(VENDOR_PATH))

import gymnasium as gym
from gymnasium import spaces as gym_spaces
from gymnasium.spaces import utils as gym_space_utils
import numpy as np
from dataclasses import replace

from .core import SimulationConfig
from .env import NTNClosedLoopEnv, RewardWeights
from .spaces import Box, Dict, MultiDiscrete


def to_gymnasium_space(space: Any) -> gym_spaces.Space:
    if isinstance(space, Box):
        return gym_spaces.Box(
            low=np.asarray(space.low, dtype=space.dtype),
            high=np.asarray(space.high, dtype=space.dtype),
            shape=space.shape,
            dtype=space.dtype,
        )
    if isinstance(space, MultiDiscrete):
        return gym_spaces.MultiDiscrete(nvec=np.asarray(space.nvec, dtype=np.int64), dtype=space.dtype)
    if isinstance(space, Dict):
        return gym_spaces.Dict({key: to_gymnasium_space(subspace) for key, subspace in space.spaces.items()})
    raise TypeError(f"Unsupported custom space type: {type(space)!r}")


def stack_dict_actions(actions: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not actions:
        raise ValueError("actions cannot be empty")
    keys = actions[0].keys()
    return {
        key: np.stack([np.asarray(action[key]) for action in actions], axis=0)
        for key in keys
    }


class SingleAgentFlattenActionWrapper(gym.ActionWrapper):
    """Flatten dict actions into a single continuous Box for simple MLP policies."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        if not isinstance(env.action_space, gym_spaces.Dict):
            raise TypeError("SingleAgentFlattenActionWrapper requires a Dict action space")
        satellite_space = env.action_space["satellite_choices"]
        if not isinstance(satellite_space, gym_spaces.MultiDiscrete):
            raise TypeError("Expected satellite_choices to be MultiDiscrete")
        self.num_users = int(satellite_space.nvec.shape[0])
        self.num_satellite_choices = int(satellite_space.nvec[0])
        low = np.concatenate(
            [
                np.zeros(self.num_users, dtype=np.float32),
                np.zeros(self.num_users * 3, dtype=np.float32),
            ]
        )
        high = np.concatenate(
            [
                np.full(self.num_users, float(self.num_satellite_choices) - 1e-6, dtype=np.float32),
                np.ones(self.num_users * 3, dtype=np.float32),
            ]
        )
        self.action_space = gym_spaces.Box(low=low, high=high, dtype=np.float32)

    def action(self, action: np.ndarray) -> dict[str, np.ndarray]:
        flat = np.asarray(action, dtype=np.float32).reshape(-1)
        if flat.shape[0] != self.action_space.shape[0]:
            raise ValueError(f"Expected flat action of shape {self.action_space.shape}, got {flat.shape}")
        num = self.num_users
        satellite_choices = np.clip(np.floor(flat[:num]).astype(np.int64), 0, self.num_satellite_choices - 1)
        beam_priority = np.clip(flat[num : 2 * num], 0.0, 1.0).astype(np.float32)
        bandwidth_weight = np.clip(flat[2 * num : 3 * num], 0.0, 1.0).astype(np.float32)
        power_weight = np.clip(flat[3 * num : 4 * num], 0.0, 1.0).astype(np.float32)
        return {
            "satellite_choices": satellite_choices,
            "beam_priority": beam_priority,
            "bandwidth_weight": bandwidth_weight,
            "power_weight": power_weight,
        }

    def reverse_action(self, action: dict[str, np.ndarray]) -> np.ndarray:
        return flatten_dict_action(action)


def flatten_dict_action(action: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(action["satellite_choices"], dtype=np.float32).reshape(-1),
            np.asarray(action["beam_priority"], dtype=np.float32).reshape(-1),
            np.asarray(action["bandwidth_weight"], dtype=np.float32).reshape(-1),
            np.asarray(action["power_weight"], dtype=np.float32).reshape(-1),
        ]
    ).astype(np.float32)


class GymnasiumNTNEnv(gym.Env):
    metadata = {"render_modes": ["human"], "name": "GymnasiumNTNEnv"}

    def __init__(
        self,
        config: SimulationConfig,
        reward_weights: RewardWeights | None = None,
        reward_template: str = "throughput_first",
    ) -> None:
        super().__init__()
        self.inner = NTNClosedLoopEnv(
            config=config,
            reward_weights=reward_weights,
            reward_template=reward_template,
        )
        self.action_space = to_gymnasium_space(self.inner.action_space)
        self.observation_space = to_gymnasium_space(self.inner.observation_space)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        return self.inner.reset(seed=seed, options=options)

    def step(
        self,
        action: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        return self.inner.step(action)

    def render(self) -> str:
        return self.inner.render()

    def close(self) -> None:
        return None


def make_gymnasium_env(
    config: SimulationConfig,
    reward_template: str = "throughput_first",
    reward_weights: RewardWeights | None = None,
) -> GymnasiumNTNEnv:
    return GymnasiumNTNEnv(
        config=config,
        reward_template=reward_template,
        reward_weights=reward_weights,
    )


class SingleAgentFlattenObservationWrapper(gym.ObservationWrapper):
    """Flatten dict observations into a single 1D Box for simple MLP baselines."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = gym_space_utils.flatten_space(env.observation_space)

    def observation(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        flat = gym_space_utils.flatten(self.env.observation_space, observation)
        return np.asarray(flat, dtype=np.float32)


def make_env_factory(
    config: SimulationConfig,
    *,
    reward_template: str = "throughput_first",
    reward_weights: RewardWeights | None = None,
    flatten_observation: bool = False,
    flatten_action: bool = False,
    seed: int | None = None,
) -> Callable[[], gym.Env]:
    def _factory() -> gym.Env:
        env_config = replace(config, random_seed=seed if seed is not None else config.random_seed)
        env: gym.Env = GymnasiumNTNEnv(
            config=env_config,
            reward_template=reward_template,
            reward_weights=reward_weights,
        )
        if flatten_observation:
            env = SingleAgentFlattenObservationWrapper(env)
        if flatten_action:
            env = SingleAgentFlattenActionWrapper(env)
        return env

    return _factory


def make_vec_env_fns(
    config: SimulationConfig,
    num_envs: int,
    *,
    reward_template: str = "throughput_first",
    reward_weights: RewardWeights | None = None,
    flatten_observation: bool = False,
    flatten_action: bool = False,
    base_seed: int | None = None,
) -> list[Callable[[], gym.Env]]:
    seed0 = config.random_seed if base_seed is None else base_seed
    return [
        make_env_factory(
            config=config,
            reward_template=reward_template,
            reward_weights=reward_weights,
            flatten_observation=flatten_observation,
            flatten_action=flatten_action,
            seed=seed0 + rank,
        )
        for rank in range(num_envs)
    ]


def make_gymnasium_vector_env(
    config: SimulationConfig,
    num_envs: int,
    *,
    reward_template: str = "throughput_first",
    reward_weights: RewardWeights | None = None,
    flatten_observation: bool = False,
    flatten_action: bool = False,
    asynchronous: bool = False,
    base_seed: int | None = None,
) -> gym.vector.VectorEnv:
    env_fns = make_vec_env_fns(
        config=config,
        num_envs=num_envs,
        reward_template=reward_template,
        reward_weights=reward_weights,
        flatten_observation=flatten_observation,
        flatten_action=flatten_action,
        base_seed=base_seed,
    )
    if asynchronous:
        return gym.vector.AsyncVectorEnv(env_fns)
    return gym.vector.SyncVectorEnv(env_fns)


def make_sb3_vec_env(
    config: SimulationConfig,
    num_envs: int,
    *,
    reward_template: str = "throughput_first",
    reward_weights: RewardWeights | None = None,
    flatten_observation: bool = False,
    flatten_action: bool = False,
    base_seed: int | None = None,
    use_subprocess: bool = True,
    monitor: bool = True,
    log_dir: str | None = None,
    monitor_kwargs: dict[str, Any] | None = None,
) -> Any:
    try:
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except Exception as exc:
        raise RuntimeError(
            "stable_baselines3 is not installed. Install it, then call make_sb3_vec_env again."
        ) from exc

    seed0 = config.random_seed if base_seed is None else base_seed
    monitor_kwargs = monitor_kwargs or {}
    log_base = Path(log_dir).resolve() if log_dir is not None else None
    if log_base is not None:
        log_base.mkdir(parents=True, exist_ok=True)

    env_fns: list[Callable[[], gym.Env]] = []
    for rank in range(num_envs):
        base_factory = make_env_factory(
            config=config,
            reward_template=reward_template,
            reward_weights=reward_weights,
            flatten_observation=flatten_observation,
            flatten_action=flatten_action,
            seed=seed0 + rank,
        )

        def _factory(base_factory: Callable[[], gym.Env] = base_factory, rank: int = rank) -> gym.Env:
            env = base_factory()
            if monitor:
                monitor_path = None
                if log_base is not None:
                    monitor_path = str(log_base / f"env_{rank}")
                env = Monitor(env, filename=monitor_path, **monitor_kwargs)
            return env

        env_fns.append(_factory)
    if use_subprocess and num_envs > 1:
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)
