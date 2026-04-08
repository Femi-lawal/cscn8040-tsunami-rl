from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .agents import rule_based_policy
from .deep_agents import RecurrentActorCritic, resolve_device
from .environment import ACTION_NAMES, ALERT_LEVELS
from .gym_env import TsunamiGymEnv

OBSERVATION_FIELDS = [
    "magnitude_estimate",
    "depth_estimate_km",
    "coastal_proximity_index",
    "wave_estimate_m",
    "buoy_confirmation",
    "tide_confirmation",
    "uncertainty",
    "time_fraction",
    "alert_level_norm",
    "cancel_issued_flag",
    "delta_magnitude",
    "delta_wave_m",
    "delta_uncertainty",
    "time_since_buoy_norm",
    "time_since_tide_norm",
]

DANGER_LABELS = {
    0: "No Threat",
    1: "Potential Threat",
    2: "Confirmed Threat",
}

CURATED_CHECKPOINT_LABELS = [
    ("Recommended PPO Policy", "ppo_lstm_recommended.pt"),
    ("Stable PPO Policy", "ppo_lstm_stable.pt"),
    ("Baseline PPO Policy", "ppo_lstm_baseline.pt"),
]


@dataclass
class PolicyDecision:
    action: int
    action_name: str
    value_estimate: float
    probabilities: np.ndarray
    valid_actions: list[int]


@dataclass
class RecurrentPolicyRuntime:
    model: RecurrentActorCritic
    device: torch.device
    checkpoint_path: Path
    config: dict[str, Any]
    lstm_state: tuple[torch.Tensor, torch.Tensor]
    done_mask: torch.Tensor

    def reset(self) -> None:
        self.lstm_state = self.model.initial_state(1, self.device)
        self.done_mask = torch.ones(1, dtype=torch.float32, device=self.device)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_catalog_path() -> Path:
    return project_root() / "data" / "processed" / "bmkg_event_summary_enriched.csv"


def default_checkpoint_path() -> Path:
    models_dir = project_root() / "outputs" / "models"
    preferred = [models_dir / filename for _, filename in CURATED_CHECKPOINT_LABELS]
    for path in preferred:
        if path.exists():
            return path
    raise FileNotFoundError(f"No PPO checkpoint found in {models_dir}")


def available_checkpoint_paths(models_dir: Path | None = None) -> dict[str, Path]:
    models_dir = models_dir or (project_root() / "outputs" / "models")
    ordered: list[tuple[str, Path]] = []

    for label, filename in CURATED_CHECKPOINT_LABELS:
        path = models_dir / filename
        if path.exists():
            ordered.append((label, path))

    if not ordered:
        fallback_candidates = sorted(models_dir.glob("ppo_lstm*.pt"))
        for index, path in enumerate(fallback_candidates[:3], start=1):
            ordered.append((f"PPO Checkpoint {index}", path))

    return dict(ordered)


def load_event_catalog(catalog_path: Path | None = None) -> pd.DataFrame:
    path = catalog_path or default_catalog_path()
    catalog = pd.read_csv(path)
    if "origin_time_utc" in catalog.columns:
        catalog["origin_time_utc"] = pd.to_datetime(catalog["origin_time_utc"], utc=True, errors="coerce")
    return catalog


def load_policy_checkpoint(
    checkpoint_path: Path | None = None,
    device: str = "cpu",
) -> tuple[RecurrentActorCritic, dict[str, Any], torch.device]:
    checkpoint_path = checkpoint_path or default_checkpoint_path()
    resolved_device = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    config = dict(checkpoint.get("config", {}))
    hidden_size = int(config.get("hidden_size", 256))
    lstm_size = int(config.get("lstm_size", 128))
    model = RecurrentActorCritic(15, len(ACTION_NAMES), hidden_size=hidden_size, lstm_size=lstm_size)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.set_observation_normalization(bool(config.get("normalize_observations", True)))
    model.to(resolved_device)
    model.eval()
    return model, config, resolved_device


def build_runtime(
    checkpoint_path: Path | None = None,
    device: str = "cpu",
) -> RecurrentPolicyRuntime:
    checkpoint_path = checkpoint_path or default_checkpoint_path()
    model, config, resolved_device = load_policy_checkpoint(checkpoint_path, device=device)
    runtime = RecurrentPolicyRuntime(
        model=model,
        device=resolved_device,
        checkpoint_path=checkpoint_path,
        config=config,
        lstm_state=model.initial_state(1, resolved_device),
        done_mask=torch.ones(1, dtype=torch.float32, device=resolved_device),
    )
    return runtime


def _decision_from_state(
    model: RecurrentActorCritic,
    device: torch.device,
    lstm_state: tuple[torch.Tensor, torch.Tensor],
    done_mask: torch.Tensor,
    observation: np.ndarray,
    action_mask: np.ndarray,
    deterministic: bool,
) -> tuple[PolicyDecision, tuple[torch.Tensor, torch.Tensor]]:
    obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    mask_tensor = torch.as_tensor(action_mask, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        actions, _, _, values, next_state, probs = model.act(
            obs_tensor,
            lstm_state,
            done_mask,
            mask_tensor,
            deterministic=deterministic,
        )
    valid_actions = [index for index, flag in enumerate(action_mask.tolist()) if flag > 0]
    decision = PolicyDecision(
        action=int(actions.item()),
        action_name=ACTION_NAMES[int(actions.item())],
        value_estimate=float(values.item()),
        probabilities=probs.detach().cpu().numpy().reshape(-1),
        valid_actions=valid_actions,
    )
    return decision, next_state


def preview_ppo_decision(
    runtime: RecurrentPolicyRuntime,
    observation: np.ndarray,
    action_mask: np.ndarray,
    deterministic: bool = True,
) -> PolicyDecision:
    decision, _ = _decision_from_state(
        runtime.model,
        runtime.device,
        (runtime.lstm_state[0].clone(), runtime.lstm_state[1].clone()),
        runtime.done_mask.clone(),
        observation,
        action_mask,
        deterministic,
    )
    return decision


def commit_ppo_decision(
    runtime: RecurrentPolicyRuntime,
    observation: np.ndarray,
    action_mask: np.ndarray,
    deterministic: bool = True,
) -> PolicyDecision:
    decision, next_state = _decision_from_state(
        runtime.model,
        runtime.device,
        runtime.lstm_state,
        runtime.done_mask,
        observation,
        action_mask,
        deterministic,
    )
    runtime.lstm_state = next_state
    runtime.done_mask = torch.zeros(1, dtype=torch.float32, device=runtime.device)
    return decision


def keep_runtime_in_sync(
    runtime: RecurrentPolicyRuntime,
    observation: np.ndarray,
    action_mask: np.ndarray,
) -> None:
    _ = commit_ppo_decision(runtime, observation, action_mask, deterministic=True)


def rule_decision(
    observation: np.ndarray,
    valid_actions: list[int],
    env: TsunamiGymEnv,
) -> PolicyDecision:
    action = int(rule_based_policy(observation, valid_actions, env.env))
    probabilities = np.zeros(len(ACTION_NAMES), dtype=np.float32)
    probabilities[action] = 1.0
    return PolicyDecision(
        action=action,
        action_name=ACTION_NAMES[action],
        value_estimate=float("nan"),
        probabilities=probabilities,
        valid_actions=list(valid_actions),
    )


def observation_to_dict(observation: np.ndarray) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in zip(OBSERVATION_FIELDS, observation.tolist(), strict=True)
    }


def observation_frame(observation: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "feature": OBSERVATION_FIELDS,
            "value": observation.astype(float),
        }
    )
    return frame


def probabilities_frame(decision: PolicyDecision, action_mask: np.ndarray) -> pd.DataFrame:
    rows = []
    for index, action_name in enumerate(ACTION_NAMES):
        rows.append(
            {
                "action": action_name,
                "probability": float(decision.probabilities[index]),
                "valid": bool(action_mask[index] > 0),
            }
        )
    return pd.DataFrame(rows)


def history_frame(history: list[dict[str, Any]]) -> pd.DataFrame:
    if not history:
        return pd.DataFrame()
    return pd.DataFrame(history)


def hidden_trace_frame(env: TsunamiGymEnv) -> pd.DataFrame:
    trace = env.env.hidden_trace
    time_grid = env.env.time_grid_min
    return pd.DataFrame(
        {
            "time_min": time_grid,
            "magnitude_estimate": np.asarray(trace["magnitude_estimates"], dtype=float),
            "depth_estimate_km": np.asarray(trace["depth_estimates"], dtype=float),
            "wave_estimate_m": np.asarray(trace["wave_estimates"], dtype=float),
            "buoy_confirmation": np.asarray(trace["buoy_confirmations"], dtype=float),
            "tide_confirmation": np.asarray(trace["tide_confirmations"], dtype=float),
            "uncertainty": np.asarray(trace["uncertainty"], dtype=float),
        }
    )


def scenario_metadata(env: TsunamiGymEnv) -> dict[str, Any]:
    scenario = env.env.current_scenario
    if scenario is None:
        return {}
    event_rows = env.event_catalog.loc[env.event_catalog["event_group_id"] == scenario.event_group_id]
    row = event_rows.iloc[0] if not event_rows.empty else pd.Series(dtype=object)
    target_alert = scenario.target_alert_level
    return {
        "event_group_id": scenario.event_group_id,
        "danger_tier": int(scenario.danger_tier),
        "danger_label": DANGER_LABELS.get(int(scenario.danger_tier), "Unknown"),
        "location_name": row.get("location_name", "Unknown location"),
        "origin_time_utc": row.get("origin_time_utc"),
        "latitude": row.get("latitude"),
        "longitude": row.get("longitude"),
        "target_alert_level": ALERT_LEVELS[target_alert] if 0 <= target_alert < len(ALERT_LEVELS) else target_alert,
        "bulletin_count": int(scenario.bulletin_count),
        "first_bulletin_delay_min": float(scenario.first_bulletin_delay_min),
        "final_bulletin_delay_min": float(scenario.final_bulletin_delay_min),
        "sea_level_confirmed_flag": bool(scenario.sea_level_confirmed_flag),
        "has_threat_assessment": bool(scenario.has_threat_assessment),
        "wave_imputed_flag": bool(scenario.wave_imputed_flag),
        "observed_max_wave_m": float(scenario.observed_max_wave_m),
        "coastal_proximity_index": float(scenario.coastal_proximity_index),
    }


def alert_level_label(alert_level: int) -> str:
    return ALERT_LEVELS[max(0, min(alert_level, len(ALERT_LEVELS) - 1))]


def filtered_catalog(
    catalog: pd.DataFrame,
    danger_filter: str,
    event_group_id: str | None = None,
) -> pd.DataFrame:
    view = catalog.copy()
    if danger_filter != "All":
        mapping = {
            "No Threat": 0,
            "Potential Threat": 1,
            "Confirmed Threat": 2,
        }
        view = view[view["danger_tier"] == mapping[danger_filter]].copy()
    if event_group_id and event_group_id != "Random from filtered catalog":
        view = view[view["event_group_id"] == event_group_id].copy()
    return view.reset_index(drop=True)
