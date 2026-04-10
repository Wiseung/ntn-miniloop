from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class Space:
    def sample(self, rng: np.random.Generator | None = None) -> Any:
        raise NotImplementedError

    def contains(self, value: Any) -> bool:
        raise NotImplementedError


@dataclass
class Box(Space):
    low: np.ndarray
    high: np.ndarray
    shape: tuple[int, ...]
    dtype: Any = np.float32

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        sample = rng.uniform(self.low, self.high, size=self.shape)
        return sample.astype(self.dtype)

    def contains(self, value: Any) -> bool:
        array = np.asarray(value, dtype=self.dtype)
        if array.shape != self.shape:
            return False
        return bool(np.all(array >= self.low) and np.all(array <= self.high))


@dataclass
class MultiDiscrete(Space):
    nvec: np.ndarray
    dtype: Any = np.int64

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        sample = np.asarray(
            [rng.integers(0, int(n)) for n in self.nvec],
            dtype=self.dtype,
        )
        return sample

    def contains(self, value: Any) -> bool:
        array = np.asarray(value, dtype=self.dtype)
        if array.shape != self.nvec.shape:
            return False
        return bool(np.all(array >= 0) and np.all(array < self.nvec))


@dataclass
class Dict(Space):
    spaces: dict[str, Space]

    def sample(self, rng: np.random.Generator | None = None) -> dict[str, Any]:
        return {
            key: space.sample(rng=rng)
            for key, space in self.spaces.items()
        }

    def contains(self, value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        for key, space in self.spaces.items():
            if key not in value or not space.contains(value[key]):
                return False
        return True
