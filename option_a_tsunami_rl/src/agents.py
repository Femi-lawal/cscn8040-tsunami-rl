from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .environment import ACTION_NAMES, TsunamiWarningEnv, discretize_observation
from .metrics import compute_operational_score

PolicyFn = Callable[[np.ndarray, list[int], TsunamiWarningEnv], int]


@dataclass
class TrainConfig:
    episodes: int = 8000
    alpha: float = 0.08
    gamma: float = 0.985
    epsilon_start: float = 0.30
    epsilon_end: float = 0.02
    rule_warm_start_episodes: int = 500
    warm_start_alpha: float = 0.18


def epsilon_schedule(config: TrainConfig, episode: int) -> float:
    progress = min(1.0, episode / max(1, config.episodes - 1))
    return float(
        config.epsilon_start
        + progress * (config.epsilon_end - config.epsilon_start)
    )


def make_q_table(action_size: int) -> defaultdict:
    return defaultdict(lambda: np.zeros(action_size, dtype=np.float32))


def _best_valid_action(values: np.ndarray, valid_actions: list[int]) -> int:
    return max(valid_actions, key=lambda action: (float(values[action]), -action))


def _best_valid_actions(values: np.ndarray, valid_actions: list[int]) -> list[int]:
    best_value = max(float(values[action]) for action in valid_actions)
    return [
        action
        for action in valid_actions
        if float(values[action]) == best_value
    ]


def select_action(
    q_table: defaultdict,
    state: tuple[int, ...],
    valid_actions: list[int],
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return int(rng.choice(valid_actions))
    best_actions = _best_valid_actions(q_table[state], valid_actions)
    return int(rng.choice(best_actions))


def pure_greedy_policy(q_table: defaultdict) -> PolicyFn:
    def _policy(
        observation: np.ndarray,
        valid_actions: list[int],
        _env: TsunamiWarningEnv,
    ) -> int:
        state = discretize_observation(observation)
        values = q_table[state]
        return int(_best_valid_action(values, valid_actions))

    return _policy


def safe_greedy_policy(q_table: defaultdict, fallback_margin: float = 3.0) -> PolicyFn:
    def _policy(
        observation: np.ndarray,
        valid_actions: list[int],
        env: TsunamiWarningEnv,
    ) -> int:
        state = discretize_observation(observation)
        values = q_table[state]
        best_action = int(_best_valid_action(values, valid_actions))
        fallback_action = int(rule_based_policy(observation, valid_actions, env))

        if best_action == fallback_action:
            return best_action

        best_value = float(values[best_action])
        fallback_value = float(values[fallback_action])
        sorted_values = sorted((float(values[action]) for action in valid_actions), reverse=True)
        runner_up = sorted_values[1] if len(sorted_values) > 1 else best_value

        if best_value - fallback_value < fallback_margin:
            return fallback_action
        if best_value - runner_up < fallback_margin:
            return fallback_action
        return best_action

    return _policy


def _target_alert_to_action(
    target_alert: int,
    current_alert: int,
    valid_actions: list[int],
) -> int:
    if target_alert >= 4 and 4 in valid_actions:
        return 4
    if target_alert >= 2:
        if current_alert < 2 and 3 in valid_actions:
            return 3
        if current_alert < target_alert and 1 in valid_actions:
            return 1
        if current_alert > target_alert and 2 in valid_actions:
            return 2
        return 0
    if target_alert == 1:
        if current_alert == 0 and 1 in valid_actions:
            return 1
        if current_alert > 1 and 2 in valid_actions:
            return 2
        return 0
    if current_alert > 0 and 2 in valid_actions:
        return 2
    return 0


def rule_based_policy(
    observation: np.ndarray,
    valid_actions: list[int],
    _env: TsunamiWarningEnv,
) -> int:
    magnitude = float(observation[0])
    depth = float(observation[1])
    coastal = float(observation[2])
    wave = float(observation[3])
    buoy = int(round(float(observation[4])))
    tide = int(round(float(observation[5])))
    uncertainty = float(observation[6])
    time_fraction = float(observation[7])
    current_alert = int(round(float(observation[8]) * 4))
    delta_magnitude = float(observation[10])
    delta_wave = float(observation[11])
    delta_uncertainty = float(observation[12])
    buoy_age = float(observation[13])
    tide_age = float(observation[14])

    high_source_risk = magnitude >= 7.35 and depth <= 55.0 and coastal >= 0.42
    deep_quake = depth > 70.0
    wave_rising = delta_wave >= 0.012
    wave_falling = delta_wave <= -0.010
    confidence_rising = delta_uncertainty <= -0.025
    long_buoy_confirmation = buoy_age >= 0.12
    long_tide_confirmation = tide_age >= 0.05
    likely_severe = (
        tide == 1
        or wave >= 0.16
        or (buoy == 1 and wave >= 0.09 and not deep_quake)
        or (wave >= 0.11 and wave_rising and (high_source_risk or magnitude >= 7.5))
        or (high_source_risk and wave >= 0.08 and confidence_rising and time_fraction >= 0.25)
    )
    likely_potential = (
        (buoy == 1 and not deep_quake)
        or (buoy == 1 and deep_quake and wave >= 0.08)
        or wave >= 0.07
        or (wave >= 0.05 and wave_rising and confidence_rising and (high_source_risk or magnitude >= 7.3))
        or (high_source_risk and wave >= 0.06)
    )
    info_risk = (
        current_alert > 0
        and (
            high_source_risk
            or (magnitude >= 7.1 and uncertainty >= 0.55)
            or (delta_magnitude >= 0.03 and uncertainty >= 0.45)
        )
    )
    stable_collapse = (
        wave < 0.035
        and wave_falling
        and buoy == 0
        and tide == 0
        and uncertainty < 0.28
        and delta_magnitude <= 0.01
        and time_fraction >= 0.55
    )

    if stable_collapse:
        if current_alert == 4 and 5 in valid_actions:
            return 5
        if current_alert > 0 and 2 in valid_actions:
            return 2
        return 0

    if tide == 1 and long_tide_confirmation:
        target_alert = 4
    elif likely_severe:
        if deep_quake and tide == 0:
            target_alert = 2  # cap at watch for deep quakes without tide
        else:
            target_alert = 4 if current_alert >= 2 else 3
    elif likely_potential and not deep_quake and (wave >= 0.08 or buoy == 1 or long_buoy_confirmation):
        target_alert = 3 if current_alert >= 2 else 2
    elif likely_potential:
        target_alert = 2
    elif info_risk and time_fraction < 0.45:
        target_alert = 1
    else:
        target_alert = 0

    return _target_alert_to_action(target_alert, current_alert, valid_actions)


def train_q_learning(
    env: TsunamiWarningEnv,
    config: TrainConfig,
    seed: int = 42,
    run_seed: int | None = None,
) -> tuple[defaultdict, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    q_table = make_q_table(env.action_size)
    history: list[dict] = []

    if config.rule_warm_start_episodes > 0:
        _warm_start_q_table_with_rule_policy(
            env,
            q_table,
            episodes=config.rule_warm_start_episodes,
            gamma=config.gamma,
            alpha=config.warm_start_alpha,
        )

    for episode in range(config.episodes):
        observation = env.reset()
        state = discretize_observation(observation)
        epsilon = epsilon_schedule(config, episode)
        total_reward = 0.0
        invalid_actions = 0
        done = False

        while not done:
            valid_actions = env.valid_actions()
            action = select_action(q_table, state, valid_actions, epsilon, rng)
            next_observation, reward, done, info = env.step(action)
            invalid_actions += int(info["invalid_action"])
            next_state = discretize_observation(next_observation) if not done else None

            target = reward
            if not done and next_state is not None:
                next_valid_actions = env.valid_actions()
                next_best = _best_valid_action(q_table[next_state], next_valid_actions)
                target += config.gamma * float(q_table[next_state][next_best])
            q_table[state][action] += config.alpha * (target - q_table[state][action])

            total_reward += reward
            state = next_state if next_state is not None else state

        history.append(
            {
                "episode": episode + 1,
                "algorithm": "q_learning",
                "epsilon": epsilon,
                "return": total_reward,
                "danger_tier": info["terminal_danger_tier"],
                "missed_severe": int(info["missed_severe"]),
                "false_warning": int(info["false_warning"]),
                "invalid_action_count": invalid_actions,
                "run_seed": run_seed if run_seed is not None else seed,
            }
        )

    return q_table, pd.DataFrame(history)


def train_sarsa(
    env: TsunamiWarningEnv,
    config: TrainConfig,
    seed: int = 84,
    run_seed: int | None = None,
) -> tuple[defaultdict, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    q_table = make_q_table(env.action_size)
    history: list[dict] = []

    if config.rule_warm_start_episodes > 0:
        _warm_start_q_table_with_rule_policy(
            env,
            q_table,
            episodes=config.rule_warm_start_episodes,
            gamma=config.gamma,
            alpha=config.warm_start_alpha,
            on_policy=True,
        )

    for episode in range(config.episodes):
        observation = env.reset()
        state = discretize_observation(observation)
        epsilon = epsilon_schedule(config, episode)
        valid_actions = env.valid_actions()
        action = select_action(q_table, state, valid_actions, epsilon, rng)
        total_reward = 0.0
        invalid_actions = 0
        done = False

        while not done:
            next_observation, reward, done, info = env.step(action)
            invalid_actions += int(info["invalid_action"])
            next_state = discretize_observation(next_observation) if not done else None
            next_valid_actions = env.valid_actions() if not done else []
            next_action = (
                select_action(q_table, next_state, next_valid_actions, epsilon, rng)
                if not done and next_state is not None
                else None
            )

            target = reward
            if not done and next_state is not None and next_action is not None:
                target += config.gamma * float(q_table[next_state][next_action])
            q_table[state][action] += config.alpha * (target - q_table[state][action])

            total_reward += reward
            state = next_state if next_state is not None else state
            action = next_action if next_action is not None else action

        history.append(
            {
                "episode": episode + 1,
                "algorithm": "sarsa",
                "epsilon": epsilon,
                "return": total_reward,
                "danger_tier": info["terminal_danger_tier"],
                "missed_severe": int(info["missed_severe"]),
                "false_warning": int(info["false_warning"]),
                "invalid_action_count": invalid_actions,
                "run_seed": run_seed if run_seed is not None else seed,
            }
        )

    return q_table, pd.DataFrame(history)


def evaluate_policy(
    env: TsunamiWarningEnv,
    policy: PolicyFn,
    *,
    episodes: int = 500,
    algorithm_name: str,
    split_name: str,
    run_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_rows: list[dict] = []

    for episode in range(episodes):
        observation = env.reset()
        total_reward = 0.0
        done = False
        invalid_actions = 0

        while not done:
            valid_actions = env.valid_actions()
            action = int(policy(observation, valid_actions, env))
            next_observation, reward, done, info = env.step(action)
            total_reward += reward
            invalid_actions += int(info["invalid_action"])
            observation = next_observation

        episode_rows.append(
            {
                "episode": episode + 1,
                "algorithm": algorithm_name,
                "split": split_name,
                "run_seed": run_seed,
                "return": total_reward,
                "danger_tier": info["terminal_danger_tier"],
                "target_alert_level": info["target_alert_level"],
                "missed_severe": int(info["missed_severe"]),
                "false_warning": int(info["false_warning"]),
                "alert_changes": info["alert_changes"],
                "warning_step": info["warning_step"],
                "watch_step": info["watch_step"],
                "cancel_step": info["cancel_step"],
                "max_alert": info["max_alert"],
                "invalid_actions": invalid_actions,
                "wave_imputed_flag": int(info["wave_imputed_flag"]),
            }
        )

    episode_df = pd.DataFrame(episode_rows)
    avg_return = float(episode_df["return"].mean())
    severe_miss_rate = float(
        episode_df.loc[episode_df["danger_tier"] == 2, "missed_severe"].mean()
    )
    false_warning_rate = float(
        episode_df.loc[episode_df["danger_tier"] == 0, "false_warning"].mean()
    )
    avg_warning_step_on_severe = float(
        episode_df.loc[episode_df["danger_tier"] == 2, "warning_step"].dropna().mean()
    )
    avg_watch_step_on_potential = float(
        episode_df.loc[episode_df["danger_tier"] == 1, "watch_step"].dropna().mean()
    )
    summary = pd.DataFrame(
        [
            {
                "algorithm": algorithm_name,
                "split": split_name,
                "run_seed": run_seed,
                "avg_return": avg_return,
                "median_return": episode_df["return"].median(),
                "severe_miss_rate": severe_miss_rate,
                "false_warning_rate": false_warning_rate,
                "avg_alert_changes": episode_df["alert_changes"].mean(),
                "avg_invalid_actions": episode_df["invalid_actions"].mean(),
                "avg_warning_step_on_severe": avg_warning_step_on_severe,
                "avg_watch_step_on_potential": avg_watch_step_on_potential,
                "safety_score": compute_operational_score(
                    avg_return,
                    severe_miss_rate,
                    false_warning_rate,
                    avg_warning_step_on_severe,
                    avg_watch_step_on_potential,
                ),
            }
        ]
    )
    return episode_df, summary


def evaluate_policy_on_catalog(
    catalog: pd.DataFrame,
    policy: PolicyFn,
    *,
    algorithm_name: str,
    split_name: str,
    run_seed: int,
    seed_base: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_rows: list[dict] = []
    catalog = catalog.reset_index(drop=True)

    for event_index, row in catalog.iterrows():
        env = TsunamiWarningEnv(pd.DataFrame([row]), seed=seed_base + event_index)
        observation = env.reset()
        total_reward = 0.0
        done = False
        invalid_actions = 0

        while not done:
            valid_actions = env.valid_actions()
            action = int(policy(observation, valid_actions, env))
            next_observation, reward, done, info = env.step(action)
            total_reward += reward
            invalid_actions += int(info["invalid_action"])
            observation = next_observation

        episode_rows.append(
            {
                "episode": event_index + 1,
                "algorithm": algorithm_name,
                "split": split_name,
                "run_seed": run_seed,
                "event_group_id": row["event_group_id"],
                "origin_time_utc": row.get("origin_time_utc"),
                "danger_label": row.get("danger_label"),
                "return": total_reward,
                "danger_tier": info["terminal_danger_tier"],
                "target_alert_level": info["target_alert_level"],
                "missed_severe": int(info["missed_severe"]),
                "false_warning": int(info["false_warning"]),
                "alert_changes": info["alert_changes"],
                "warning_step": info["warning_step"],
                "watch_step": info["watch_step"],
                "cancel_step": info["cancel_step"],
                "max_alert": info["max_alert"],
                "invalid_actions": invalid_actions,
                "wave_imputed_flag": int(info["wave_imputed_flag"]),
            }
        )

    episode_df = pd.DataFrame(episode_rows)
    avg_return = float(episode_df["return"].mean())
    severe_miss_rate = float(
        episode_df.loc[episode_df["danger_tier"] == 2, "missed_severe"].mean()
    )
    false_warning_rate = float(
        episode_df.loc[episode_df["danger_tier"] == 0, "false_warning"].mean()
    )
    avg_warning_step_on_severe = float(
        episode_df.loc[episode_df["danger_tier"] == 2, "warning_step"].dropna().mean()
    )
    avg_watch_step_on_potential = float(
        episode_df.loc[episode_df["danger_tier"] == 1, "watch_step"].dropna().mean()
    )
    summary = pd.DataFrame(
        [
            {
                "algorithm": algorithm_name,
                "split": split_name,
                "run_seed": run_seed,
                "episode_count": len(episode_df),
                "no_threat_event_count": int((episode_df["danger_tier"] == 0).sum()),
                "potential_event_count": int((episode_df["danger_tier"] == 1).sum()),
                "severe_event_count": int((episode_df["danger_tier"] == 2).sum()),
                "avg_return": avg_return,
                "median_return": episode_df["return"].median(),
                "severe_miss_rate": severe_miss_rate,
                "false_warning_rate": false_warning_rate,
                "avg_alert_changes": episode_df["alert_changes"].mean(),
                "avg_invalid_actions": episode_df["invalid_actions"].mean(),
                "avg_warning_step_on_severe": avg_warning_step_on_severe,
                "avg_watch_step_on_potential": avg_watch_step_on_potential,
                "safety_score": compute_operational_score(
                    avg_return,
                    severe_miss_rate,
                    false_warning_rate,
                    avg_warning_step_on_severe,
                    avg_watch_step_on_potential,
                ),
            }
        ]
    )
    return episode_df, summary


def _warm_start_q_table_with_rule_policy(
    env: TsunamiWarningEnv,
    q_table: defaultdict,
    *,
    episodes: int,
    gamma: float,
    alpha: float,
    on_policy: bool = False,
) -> None:
    for _ in range(episodes):
        observation = env.reset()
        state = discretize_observation(observation)
        done = False

        while not done:
            valid_actions = env.valid_actions()
            action = int(rule_based_policy(observation, valid_actions, env))
            next_observation, reward, done, _info = env.step(action)
            next_state = discretize_observation(next_observation) if not done else None

            target = reward
            if not done and next_state is not None:
                next_valid_actions = env.valid_actions()
                if on_policy:
                    next_action = int(rule_based_policy(next_observation, next_valid_actions, env))
                    target += gamma * float(q_table[next_state][next_action])
                else:
                    next_best = _best_valid_action(q_table[next_state], next_valid_actions)
                    target += gamma * float(q_table[next_state][next_best])
            q_table[state][action] += alpha * (target - q_table[state][action])

            observation = next_observation
            state = next_state if next_state is not None else state
