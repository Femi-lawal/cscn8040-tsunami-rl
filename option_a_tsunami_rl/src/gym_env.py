from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

from .environment import TsunamiWarningEnv


class TsunamiGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        event_catalog: pd.DataFrame,
        seed: int = 42,
        weight_column: str | None = None,
    ) -> None:
        super().__init__()
        self.event_catalog = event_catalog.reset_index(drop=True).copy()
        self.weight_column = weight_column
        self.base_seed = seed
        self.env = TsunamiWarningEnv(self.event_catalog, seed=seed, weight_column=weight_column)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.env.observation_size,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(self.env.action_size)

    def _reset_rng(self, seed: int | None) -> None:
        if seed is None:
            return
        self.base_seed = seed
        self.env = TsunamiWarningEnv(self.event_catalog, seed=seed, weight_column=self.weight_column)

    def get_action_mask(self) -> np.ndarray:
        return self.env.action_mask().copy()

    def action_masks(self) -> np.ndarray:
        return self.get_action_mask()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._reset_rng(seed)
        observation = self.env.reset().astype(np.float32, copy=False)
        info = {
            "valid_actions": self.env.valid_actions(),
            "action_mask": self.env.action_mask().copy(),
            "danger_tier": self.env.current_scenario.danger_tier if self.env.current_scenario else None,
            "event_group_id": self.env.current_scenario.event_group_id if self.env.current_scenario else None,
            "options": options or {},
        }
        return observation.copy(), info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, done, info = self.env.step(int(action))
        terminated = bool(done)
        truncated = False
        step_info = dict(info)
        step_info.update(
            {
                "valid_actions": [] if terminated else self.env.valid_actions(),
                "action_mask": np.zeros(self.action_space.n, dtype=np.float32)
                if terminated
                else self.env.action_mask().copy(),
                "danger_tier": info.get("danger_tier", info.get("terminal_danger_tier")),
                "event_group_id": info.get("event_group_id"),
            }
        )
        return observation.astype(np.float32, copy=False), float(reward), terminated, truncated, step_info
