from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .gym_env import TsunamiGymEnv
from .metrics import OperationalScoreWeights, compute_operational_score


@dataclass
class PPOConfig:
    total_steps: int = 2_000_000
    n_envs: int = 16
    rollout_steps: int = 256
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    learning_rate: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatches: int = 8
    hidden_size: int = 256
    lstm_size: int = 128
    seed: int = 42
    eval_interval: int = 50_000
    device: str = "cuda"
    severe_miss_penalty: float = 300.0
    false_warning_penalty: float = 50.0
    warning_delay_penalty: float = 0.0
    potential_delay_penalty: float = 0.0
    normalize_observations: bool = True
    use_curriculum: bool = True
    curriculum_stage_fractions: tuple[float, float, float] = (0.20, 0.30, 0.50)
    init_checkpoint_path: str | None = None


def score_weights_from_config(config: PPOConfig) -> OperationalScoreWeights:
    return OperationalScoreWeights(
        severe_miss_penalty=config.severe_miss_penalty,
        false_warning_penalty=config.false_warning_penalty,
        warning_delay_penalty=config.warning_delay_penalty,
        potential_delay_penalty=config.potential_delay_penalty,
    )


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested_device)


class RecurrentActorCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        lstm_size: int = 128,
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.lstm_size = lstm_size
        self.normalize_observations = True

        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_size),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden_size, lstm_size)
        self.policy_head = nn.Linear(lstm_size, action_dim)
        self.value_head = nn.Linear(lstm_size, 1)
        self.register_buffer("obs_mean", torch.zeros(observation_dim))
        self.register_buffer("obs_var", torch.ones(observation_dim))
        self.register_buffer("obs_count", torch.tensor(1e-4))

    def initial_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(1, batch_size, self.lstm_size, device=device)
        c = torch.zeros(1, batch_size, self.lstm_size, device=device)
        return h, c

    def set_observation_normalization(self, enabled: bool) -> None:
        self.normalize_observations = enabled

    def update_observation_stats(self, observations: np.ndarray | torch.Tensor) -> None:
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.obs_mean.device)
        if obs_tensor.numel() == 0:
            return
        obs_tensor = obs_tensor.reshape(-1, self.observation_dim)
        batch_count = torch.tensor(float(obs_tensor.shape[0]), device=self.obs_mean.device)
        batch_mean = obs_tensor.mean(dim=0)
        batch_var = obs_tensor.var(dim=0, unbiased=False)

        delta = batch_mean - self.obs_mean
        total_count = self.obs_count + batch_count
        new_mean = self.obs_mean + delta * (batch_count / total_count)
        m_a = self.obs_var * self.obs_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.obs_count * batch_count / total_count
        new_var = torch.clamp(m2 / total_count, min=1e-6)

        self.obs_mean.copy_(new_mean)
        self.obs_var.copy_(new_var)
        self.obs_count.copy_(total_count)

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.normalize_observations:
            return obs
        obs = (obs - self.obs_mean) / torch.sqrt(torch.clamp(self.obs_var, min=1e-6))
        return torch.clamp(obs, -10.0, 10.0)

    def _ensure_sequence(
        self,
        obs: torch.Tensor,
        done_mask: torch.Tensor,
        action_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, bool]:
        single_step = obs.dim() == 2
        if single_step:
            obs = obs.unsqueeze(0)
        if done_mask.dim() == 1:
            done_mask = done_mask.unsqueeze(0)
        if action_mask is not None and action_mask.dim() == 2:
            action_mask = action_mask.unsqueeze(0)
        obs = self._normalize_obs(obs)
        return obs, done_mask, action_mask, single_step

    def forward(
        self,
        obs: torch.Tensor,
        lstm_state: tuple[torch.Tensor, torch.Tensor],
        done_mask: torch.Tensor,
        action_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        obs, done_mask, action_mask, single_step = self._ensure_sequence(obs, done_mask, action_mask)
        time_steps, batch_size, _ = obs.shape
        encoded = self.encoder(obs.reshape(time_steps * batch_size, self.observation_dim))
        encoded = encoded.view(time_steps, batch_size, self.hidden_size)

        h, c = lstm_state
        outputs: list[torch.Tensor] = []
        for step in range(time_steps):
            reset_mask = (1.0 - done_mask[step].float()).view(1, batch_size, 1)
            h = h * reset_mask
            c = c * reset_mask
            output, (h, c) = self.lstm(encoded[step : step + 1], (h, c))
            outputs.append(output)

        recurrent_output = torch.cat(outputs, dim=0)
        logits = self.policy_head(recurrent_output)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask <= 0, -1e9)
        values = self.value_head(recurrent_output).squeeze(-1)

        if single_step:
            return logits.squeeze(0), values.squeeze(0), (h, c)
        return logits, values, (h, c)

    def act(
        self,
        obs: torch.Tensor,
        lstm_state: tuple[torch.Tensor, torch.Tensor],
        done_mask: torch.Tensor,
        action_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        logits, values, next_state = self.forward(obs, lstm_state, done_mask, action_mask)
        dist = Categorical(logits=logits)
        if deterministic:
            actions = torch.argmax(logits, dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return actions, log_probs, entropy, values, next_state, dist.probs

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        lstm_state: tuple[torch.Tensor, torch.Tensor],
        done_mask: torch.Tensor,
        action_mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values, _ = self.forward(obs, lstm_state, done_mask, action_mask)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values, logits


class RecurrentRolloutBuffer:
    def __init__(
        self,
        rollout_steps: int,
        n_envs: int,
        observation_dim: int,
        action_dim: int,
        lstm_size: int,
    ) -> None:
        self.rollout_steps = rollout_steps
        self.n_envs = n_envs
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.lstm_size = lstm_size

        self.observations = np.zeros((rollout_steps, n_envs, observation_dim), dtype=np.float32)
        self.action_masks = np.zeros((rollout_steps, n_envs, action_dim), dtype=np.float32)
        self.actions = np.zeros((rollout_steps, n_envs), dtype=np.int64)
        self.rewards = np.zeros((rollout_steps, n_envs), dtype=np.float32)
        self.start_dones = np.zeros((rollout_steps, n_envs), dtype=np.float32)
        self.next_dones = np.zeros((rollout_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((rollout_steps, n_envs), dtype=np.float32)
        self.log_probs = np.zeros((rollout_steps, n_envs), dtype=np.float32)
        self.advantages = np.zeros((rollout_steps, n_envs), dtype=np.float32)
        self.returns = np.zeros((rollout_steps, n_envs), dtype=np.float32)
        self.initial_h = np.zeros((1, n_envs, lstm_size), dtype=np.float32)
        self.initial_c = np.zeros((1, n_envs, lstm_size), dtype=np.float32)

    def set_initial_state(self, lstm_state: tuple[torch.Tensor, torch.Tensor]) -> None:
        h, c = lstm_state
        self.initial_h[...] = h.detach().cpu().numpy()
        self.initial_c[...] = c.detach().cpu().numpy()

    def add(
        self,
        step: int,
        *,
        observations: np.ndarray,
        action_masks: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        start_dones: np.ndarray,
        next_dones: np.ndarray,
        values: np.ndarray,
        log_probs: np.ndarray,
    ) -> None:
        self.observations[step] = observations
        self.action_masks[step] = action_masks
        self.actions[step] = actions
        self.rewards[step] = rewards
        self.start_dones[step] = start_dones
        self.next_dones[step] = next_dones
        self.values[step] = values
        self.log_probs[step] = log_probs

    def compute_returns_and_advantages(
        self,
        next_value: np.ndarray,
        next_done: np.ndarray,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        last_gae = np.zeros(self.n_envs, dtype=np.float32)
        for step in reversed(range(self.rollout_steps)):
            if step == self.rollout_steps - 1:
                next_nonterminal = 1.0 - next_done.astype(np.float32)
                next_values = next_value.astype(np.float32)
            else:
                next_nonterminal = 1.0 - self.next_dones[step].astype(np.float32)
                next_values = self.values[step + 1]
            delta = self.rewards[step] + gamma * next_values * next_nonterminal - self.values[step]
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            self.advantages[step] = last_gae
        self.returns = self.advantages + self.values
        flat_advantages = self.advantages.reshape(-1)
        std = float(flat_advantages.std())
        if std > 1e-8:
            self.advantages = (self.advantages - flat_advantages.mean()) / (std + 1e-8)

    def iter_minibatches(
        self,
        minibatches: int,
        device: torch.device,
        rng: np.random.Generator,
    ) -> list[dict[str, torch.Tensor]]:
        env_indices = rng.permutation(self.n_envs)
        chunks = [chunk for chunk in np.array_split(env_indices, minibatches) if len(chunk) > 0]
        batches: list[dict[str, torch.Tensor]] = []
        for chunk in chunks:
            batches.append(
                {
                    "observations": torch.as_tensor(self.observations[:, chunk], device=device),
                    "action_masks": torch.as_tensor(self.action_masks[:, chunk], device=device),
                    "actions": torch.as_tensor(self.actions[:, chunk], device=device),
                    "start_dones": torch.as_tensor(self.start_dones[:, chunk], device=device),
                    "old_log_probs": torch.as_tensor(self.log_probs[:, chunk], device=device),
                    "advantages": torch.as_tensor(self.advantages[:, chunk], device=device),
                    "returns": torch.as_tensor(self.returns[:, chunk], device=device),
                    "initial_h": torch.as_tensor(self.initial_h[:, chunk], device=device),
                    "initial_c": torch.as_tensor(self.initial_c[:, chunk], device=device),
                }
            )
        return batches


def _evaluate_policy_metrics(
    episodes: list[dict[str, Any]],
    *,
    weights: OperationalScoreWeights | None = None,
) -> dict[str, float]:
    episode_df = pd.DataFrame(episodes)
    if episode_df.empty:
        return {
            "avg_return": float("nan"),
            "severe_miss_rate": float("nan"),
            "false_warning_rate": float("nan"),
            "avg_warning_step_on_severe": float("nan"),
            "avg_watch_step_on_potential": float("nan"),
            "safety_score": float("nan"),
        }
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
    return {
        "avg_return": avg_return,
        "severe_miss_rate": severe_miss_rate,
        "false_warning_rate": false_warning_rate,
        "avg_warning_step_on_severe": avg_warning_step_on_severe,
        "avg_watch_step_on_potential": avg_watch_step_on_potential,
        "safety_score": compute_operational_score(
            avg_return,
            severe_miss_rate,
            false_warning_rate,
            avg_warning_step_on_severe,
            avg_watch_step_on_potential,
            weights=weights,
        ),
    }


def _policy_checkpoint_payload(
    model: RecurrentActorCritic,
    config: PPOConfig,
    best_metric: float | None = None,
    best_metric_name: str = "validation_safety_score",
) -> dict[str, Any]:
    return {
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "best_metric": best_metric,
        "best_metric_name": best_metric_name,
    }


def _start_vector_envs(
    catalog: pd.DataFrame,
    config: PPOConfig,
    weight_column: str | None,
) -> tuple[list[TsunamiGymEnv], np.ndarray, np.ndarray, np.ndarray]:
    envs = [
        TsunamiGymEnv(catalog, seed=config.seed + env_index, weight_column=weight_column)
        for env_index in range(config.n_envs)
    ]
    observations: list[np.ndarray] = []
    action_masks: list[np.ndarray] = []
    done_flags: list[float] = []
    for env_index, env in enumerate(envs):
        observation, info = env.reset(seed=config.seed + env_index)
        observations.append(observation)
        action_masks.append(np.asarray(info["action_mask"], dtype=np.float32))
        done_flags.append(1.0)
    return envs, np.asarray(observations, dtype=np.float32), np.asarray(action_masks, dtype=np.float32), np.asarray(done_flags, dtype=np.float32)


def _tensorize_step_inputs(
    observations: np.ndarray,
    action_masks: np.ndarray,
    done_flags: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    mask_tensor = torch.as_tensor(action_masks, dtype=torch.float32, device=device)
    done_tensor = torch.as_tensor(done_flags, dtype=torch.float32, device=device)
    return obs_tensor, mask_tensor, done_tensor


def _normalized_stage_updates(total_updates: int, fractions: tuple[float, ...]) -> list[int]:
    raw = np.asarray(fractions, dtype=float)
    if raw.size == 0 or raw.sum() <= 0:
        return [total_updates]
    raw = raw / raw.sum()
    exact = raw * total_updates
    updates = np.floor(exact).astype(int)
    remainder = total_updates - int(updates.sum())
    if remainder > 0:
        ordering = np.argsort(-(exact - updates))
        for index in ordering[:remainder]:
            updates[index] += 1
    return updates.tolist()


def _build_curriculum_stages(
    train_catalog: pd.DataFrame,
    *,
    weight_column: str | None,
    use_curriculum: bool,
) -> list[dict[str, Any]]:
    base = train_catalog.copy()
    if weight_column is not None and weight_column in base.columns:
        base_weights = pd.to_numeric(base[weight_column], errors="coerce").fillna(1.0)
    else:
        base_weights = pd.Series(np.ones(len(base), dtype=float), index=base.index)

    if not use_curriculum:
        stage = base.copy()
        stage["_curriculum_weight"] = base_weights.to_numpy(dtype=float)
        return [{"name": "full", "catalog": stage, "weight_column": "_curriculum_weight"}]

    stage_one = base[base["danger_tier"].isin([0, 2])].copy()
    if stage_one.empty:
        stage_one = base.copy()
    stage_one["_curriculum_weight"] = base_weights.loc[stage_one.index].to_numpy(dtype=float)

    stage_two = base.copy()
    stage_two["_curriculum_weight"] = (
        base_weights * np.where(stage_two["danger_tier"] == 1, 1.5, 1.0)
    ).to_numpy(dtype=float)

    stage_three = base.copy()
    stage_three["_curriculum_weight"] = base_weights.to_numpy(dtype=float)

    return [
        {"name": "binary_easy", "catalog": stage_one, "weight_column": "_curriculum_weight"},
        {"name": "potential_added", "catalog": stage_two, "weight_column": "_curriculum_weight"},
        {"name": "full", "catalog": stage_three, "weight_column": "_curriculum_weight"},
    ]


def _stage_schedule(
    train_catalog: pd.DataFrame,
    *,
    total_updates: int,
    weight_column: str | None,
    use_curriculum: bool,
    fractions: tuple[float, ...],
) -> list[dict[str, Any]]:
    stages = _build_curriculum_stages(
        train_catalog,
        weight_column=weight_column,
        use_curriculum=use_curriculum,
    )
    updates_per_stage = _normalized_stage_updates(total_updates, fractions[: len(stages)])
    schedule: list[dict[str, Any]] = []
    end_update = 0
    for stage, stage_updates in zip(stages, updates_per_stage):
        if stage_updates <= 0:
            continue
        end_update += int(stage_updates)
        schedule.append({**stage, "end_update": end_update})
    return schedule or [{"name": "full", "catalog": train_catalog.copy(), "weight_column": weight_column, "end_update": total_updates}]


def evaluate_ppo_lstm_on_catalog(
    catalog: pd.DataFrame,
    model: RecurrentActorCritic,
    *,
    algorithm_name: str,
    split_name: str,
    run_seed: int,
    seed_base: int = 0,
    device: str | torch.device = "cpu",
    score_weights: OperationalScoreWeights | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_device = torch.device(device) if not isinstance(device, torch.device) else device
    model.eval()
    episode_rows: list[dict[str, Any]] = []
    catalog = catalog.reset_index(drop=True)

    for event_index, row in catalog.iterrows():
        env = TsunamiGymEnv(pd.DataFrame([row]), seed=seed_base + event_index)
        observation, info = env.reset(seed=seed_base + event_index)
        done_flag = np.array([1.0], dtype=np.float32)
        action_mask = np.asarray(info["action_mask"], dtype=np.float32)[None, :]
        lstm_state = model.initial_state(1, model_device)
        total_reward = 0.0
        invalid_actions = 0
        terminated = False

        while not terminated:
            obs_tensor, mask_tensor, done_tensor = _tensorize_step_inputs(
                observation[None, :],
                action_mask,
                done_flag,
                model_device,
            )
            with torch.no_grad():
                actions, _, _, _, lstm_state, _ = model.act(
                    obs_tensor,
                    lstm_state,
                    done_tensor,
                    mask_tensor,
                    deterministic=True,
                )
            action = int(actions.item())
            observation, reward, terminated, truncated, step_info = env.step(action)
            total_reward += reward
            invalid_actions += int(step_info["invalid_action"])
            done_flag = np.array([1.0 if terminated or truncated else 0.0], dtype=np.float32)
            action_mask = np.asarray(step_info["action_mask"], dtype=np.float32)[None, :]

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
                "danger_tier": step_info["terminal_danger_tier"],
                "target_alert_level": step_info["target_alert_level"],
                "missed_severe": int(step_info["missed_severe"]),
                "false_warning": int(step_info["false_warning"]),
                "alert_changes": step_info["alert_changes"],
                "warning_step": step_info["warning_step"],
                "watch_step": step_info["watch_step"],
                "cancel_step": step_info["cancel_step"],
                "max_alert": step_info["max_alert"],
                "invalid_actions": invalid_actions,
                "wave_imputed_flag": int(step_info["wave_imputed_flag"]),
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
    safety_score = compute_operational_score(
        avg_return,
        severe_miss_rate,
        false_warning_rate,
        avg_warning_step_on_severe,
        avg_watch_step_on_potential,
        weights=score_weights,
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
                "safety_score": safety_score,
            }
        ]
    )
    return episode_df, summary


def train_ppo_lstm(
    train_catalog: pd.DataFrame,
    config: PPOConfig,
    *,
    validation_catalog: pd.DataFrame | None = None,
    output_dir: Path | None = None,
    weight_column: str | None = "training_weight",
    artifact_prefix: str = "ppo_lstm",
) -> tuple[RecurrentActorCritic, pd.DataFrame]:
    device = resolve_device(config.device)
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    sample_env = TsunamiGymEnv(train_catalog, seed=config.seed, weight_column=weight_column)
    model = RecurrentActorCritic(
        sample_env.observation_space.shape[0],
        sample_env.action_space.n,
        hidden_size=config.hidden_size,
        lstm_size=config.lstm_size,
    ).to(device)
    model.set_observation_normalization(config.normalize_observations)
    if config.init_checkpoint_path:
        checkpoint_path = Path(config.init_checkpoint_path)
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["state_dict"], strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    score_weights = score_weights_from_config(config)

    model_dir = None if output_dir is None else output_dir / "models"
    table_dir = None if output_dir is None else output_dir / "tables"
    tensorboard_dir = None if output_dir is None else output_dir / "tensorboard"
    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)
    if table_dir is not None:
        table_dir.mkdir(parents=True, exist_ok=True)
    if tensorboard_dir is not None:
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_dir / artifact_prefix)) if tensorboard_dir is not None else None

    total_batch_steps = config.n_envs * config.rollout_steps
    total_updates = max(1, config.total_steps // total_batch_steps)
    curriculum_schedule = _stage_schedule(
        train_catalog,
        total_updates=total_updates,
        weight_column=weight_column,
        use_curriculum=config.use_curriculum,
        fractions=config.curriculum_stage_fractions,
    )
    current_stage_index = 0
    current_stage = curriculum_schedule[current_stage_index]
    envs, observations, action_masks, done_flags = _start_vector_envs(
        current_stage["catalog"],
        config,
        current_stage["weight_column"],
    )
    lstm_state = model.initial_state(config.n_envs, device)
    if config.normalize_observations:
        model.update_observation_stats(observations)
    best_checkpoint_path = None if model_dir is None else model_dir / f"{artifact_prefix}_best.pt"
    last_checkpoint_path = None if model_dir is None else model_dir / f"{artifact_prefix}_last.pt"

    completed_episodes: list[dict[str, Any]] = []
    episode_returns = np.zeros(config.n_envs, dtype=np.float32)
    best_validation_score = -float("inf")
    history_rows: list[dict[str, Any]] = []

    progress = tqdm(
        total=total_updates,
        desc="ppo_lstm",
        leave=False,
        disable=not sys.stdout.isatty(),
    )
    env_steps = 0
    for update in range(1, total_updates + 1):
        if update > current_stage["end_update"] and current_stage_index + 1 < len(curriculum_schedule):
            current_stage_index += 1
            current_stage = curriculum_schedule[current_stage_index]
            envs, observations, action_masks, done_flags = _start_vector_envs(
                current_stage["catalog"],
                config,
                current_stage["weight_column"],
            )
            lstm_state = model.initial_state(config.n_envs, device)
            if config.normalize_observations:
                model.update_observation_stats(observations)
        buffer = RecurrentRolloutBuffer(
            config.rollout_steps,
            config.n_envs,
            sample_env.observation_space.shape[0],
            sample_env.action_space.n,
            config.lstm_size,
        )
        buffer.set_initial_state(lstm_state)

        for step in range(config.rollout_steps):
            if config.normalize_observations:
                model.update_observation_stats(observations)
            obs_tensor, mask_tensor, done_tensor = _tensorize_step_inputs(
                observations,
                action_masks,
                done_flags,
                device,
            )
            with torch.no_grad():
                actions, log_probs, _, values, next_lstm_state, _ = model.act(
                    obs_tensor,
                    lstm_state,
                    done_tensor,
                    mask_tensor,
                    deterministic=False,
                )

            action_array = actions.detach().cpu().numpy()
            value_array = values.detach().cpu().numpy()
            log_prob_array = log_probs.detach().cpu().numpy()
            next_observations = np.zeros_like(observations)
            next_masks = np.zeros_like(action_masks)
            next_done_flags = np.zeros_like(done_flags)
            rewards = np.zeros(config.n_envs, dtype=np.float32)

            for env_index, env in enumerate(envs):
                next_obs, reward, terminated, truncated, info = env.step(int(action_array[env_index]))
                done = bool(terminated or truncated)
                episode_returns[env_index] += float(reward)
                rewards[env_index] = float(reward)

                if done:
                    completed_episodes.append(
                        {
                            "return": float(episode_returns[env_index]),
                            "danger_tier": int(info["terminal_danger_tier"]),
                            "missed_severe": int(info["missed_severe"]),
                            "false_warning": int(info["false_warning"]),
                            "warning_step": info["warning_step"],
                            "watch_step": info["watch_step"],
                        }
                    )
                    episode_returns[env_index] = 0.0
                    next_obs, reset_info = env.reset()
                    next_masks[env_index] = np.asarray(reset_info["action_mask"], dtype=np.float32)
                    next_done_flags[env_index] = 1.0
                else:
                    next_masks[env_index] = np.asarray(info["action_mask"], dtype=np.float32)
                    next_done_flags[env_index] = 0.0
                next_observations[env_index] = next_obs

            buffer.add(
                step,
                observations=observations,
                action_masks=action_masks,
                actions=action_array,
                rewards=rewards,
                start_dones=done_flags,
                next_dones=next_done_flags,
                values=value_array,
                log_probs=log_prob_array,
            )

            observations = next_observations
            action_masks = next_masks
            done_flags = next_done_flags
            lstm_state = next_lstm_state
            env_steps += config.n_envs

        with torch.no_grad():
            obs_tensor, mask_tensor, done_tensor = _tensorize_step_inputs(
                observations,
                action_masks,
                done_flags,
                device,
            )
            _, _, _, next_values, _, _ = model.act(
                obs_tensor,
                lstm_state,
                done_tensor,
                mask_tensor,
                deterministic=True,
            )
        buffer.compute_returns_and_advantages(
            next_values.detach().cpu().numpy(),
            done_flags,
            config.gamma,
            config.gae_lambda,
        )

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        clip_fractions: list[float] = []

        model.train()
        for _epoch in range(config.update_epochs):
            for batch in buffer.iter_minibatches(config.minibatches, device, rng):
                log_probs, entropy, values, _ = model.evaluate_actions(
                    batch["observations"],
                    (batch["initial_h"], batch["initial_c"]),
                    batch["start_dones"],
                    batch["action_masks"],
                    batch["actions"],
                )
                ratio = torch.exp(log_probs - batch["old_log_probs"])
                unclipped = ratio * batch["advantages"]
                clipped = torch.clamp(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef) * batch["advantages"]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(values, batch["returns"])
                entropy_mean = entropy.mean()
                loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_mean

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy_mean.item()))
                clip_fractions.append(float(((ratio - 1.0).abs() > config.clip_coef).float().mean().item()))

        model.eval()
        recent_metrics = _evaluate_policy_metrics(completed_episodes[-100:], weights=score_weights)
        validation_metrics = {
            "avg_return": float("nan"),
            "severe_miss_rate": float("nan"),
            "false_warning_rate": float("nan"),
            "avg_warning_step_on_severe": float("nan"),
            "avg_watch_step_on_potential": float("nan"),
            "safety_score": float("nan"),
        }
        should_evaluate = validation_catalog is not None and (
            env_steps % config.eval_interval < total_batch_steps or update == total_updates
        )
        if should_evaluate and validation_catalog is not None:
            _, validation_summary = evaluate_ppo_lstm_on_catalog(
                validation_catalog,
                model,
                algorithm_name="ppo_lstm_pure",
                split_name="validation",
                run_seed=config.seed,
                seed_base=91000 + update * 100,
                device=device,
                score_weights=score_weights,
            )
            validation_row = validation_summary.iloc[0]
            validation_metrics = {
                "avg_return": float(validation_row["avg_return"]),
                "severe_miss_rate": float(validation_row["severe_miss_rate"]),
                "false_warning_rate": float(validation_row["false_warning_rate"]),
                "avg_warning_step_on_severe": float(validation_row["avg_warning_step_on_severe"]),
                "avg_watch_step_on_potential": float(validation_row["avg_watch_step_on_potential"]),
                "safety_score": float(validation_row["safety_score"]),
            }
            if model_dir is not None and validation_metrics["safety_score"] > best_validation_score:
                best_validation_score = validation_metrics["safety_score"]
                torch.save(
                    _policy_checkpoint_payload(model, config, best_validation_score),
                    best_checkpoint_path,
                )

        if model_dir is not None:
            torch.save(
                _policy_checkpoint_payload(model, config, best_validation_score),
                last_checkpoint_path,
            )

        row = {
            "update": update,
            "env_steps": env_steps,
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else float("nan"),
            "value_loss": float(np.mean(value_losses)) if value_losses else float("nan"),
            "entropy": float(np.mean(entropies)) if entropies else float("nan"),
            "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else float("nan"),
            "recent_avg_return": recent_metrics["avg_return"],
            "recent_severe_miss_rate": recent_metrics["severe_miss_rate"],
            "recent_false_warning_rate": recent_metrics["false_warning_rate"],
            "recent_avg_warning_step_on_severe": recent_metrics["avg_warning_step_on_severe"],
            "recent_avg_watch_step_on_potential": recent_metrics["avg_watch_step_on_potential"],
            "recent_safety_score": recent_metrics["safety_score"],
            "validation_avg_return": validation_metrics["avg_return"],
            "validation_severe_miss_rate": validation_metrics["severe_miss_rate"],
            "validation_false_warning_rate": validation_metrics["false_warning_rate"],
            "validation_avg_warning_step_on_severe": validation_metrics["avg_warning_step_on_severe"],
            "validation_avg_watch_step_on_potential": validation_metrics["avg_watch_step_on_potential"],
            "validation_safety_score": validation_metrics["safety_score"],
            "device": str(device),
            "artifact_prefix": artifact_prefix,
            "curriculum_stage": current_stage["name"],
            "normalize_observations": int(config.normalize_observations),
        }
        history_rows.append(row)
        if writer is not None:
            writer.add_scalar("ppo/recent_avg_return", row["recent_avg_return"], env_steps)
            writer.add_scalar("ppo/recent_safety_score", row["recent_safety_score"], env_steps)
            writer.add_scalar("ppo/policy_loss", row["policy_loss"], env_steps)
            writer.add_scalar("ppo/value_loss", row["value_loss"], env_steps)
            writer.add_scalar("ppo/entropy", row["entropy"], env_steps)
            writer.add_scalar("ppo/clip_fraction", row["clip_fraction"], env_steps)
            if not np.isnan(row["validation_avg_return"]):
                writer.add_scalar("ppo/validation_avg_return", row["validation_avg_return"], env_steps)
                writer.add_scalar("ppo/validation_safety_score", row["validation_safety_score"], env_steps)

        progress.update(1)

    progress.close()
    if writer is not None:
        writer.flush()
        writer.close()

    history_df = pd.DataFrame(history_rows)
    if table_dir is not None:
        history_df.to_csv(table_dir / f"{artifact_prefix}_training_history.csv", index=False)
    if best_checkpoint_path is not None and best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    return model, history_df
