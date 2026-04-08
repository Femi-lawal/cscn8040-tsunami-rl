"""FastAPI backend for the Tsunami RL Operations Console.

Provides endpoints to:
- List the event catalog
- List available model checkpoints
- Simulate a full episode and return all frames for frontend replay
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is on sys.path so imports resolve
_project_root = Path(__file__).resolve().parents[1]
_workspace_root = _project_root.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from option_a_tsunami_rl.src.agents import rule_based_policy
from option_a_tsunami_rl.src.environment import ACTION_NAMES, ALERT_LEVELS
from option_a_tsunami_rl.src.gym_env import TsunamiGymEnv
from option_a_tsunami_rl.src.viewer import (
    DANGER_LABELS,
    OBSERVATION_FIELDS,
    available_checkpoint_paths,
    build_runtime,
    commit_ppo_decision,
    default_catalog_path,
    hidden_trace_frame,
    load_event_catalog,
    observation_to_dict,
    rule_decision,
    scenario_metadata,
)

app = FastAPI(title="Tsunami RL Console API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

_catalog: pd.DataFrame | None = None
_runtime_cache: dict[str, Any] = {}


def _get_catalog() -> pd.DataFrame:
    global _catalog
    if _catalog is None:
        _catalog = load_event_catalog()
    return _catalog


def _get_runtime(checkpoint_name: str | None = None):
    """Return a cached RecurrentPolicyRuntime, lazily loading if needed."""
    key = checkpoint_name or "__default__"
    if key not in _runtime_cache:
        checkpoints = available_checkpoint_paths()
        if checkpoint_name and checkpoint_name in checkpoints:
            path = checkpoints[checkpoint_name]
        else:
            path = None  # will use default_checkpoint_path()
        _runtime_cache[key] = build_runtime(checkpoint_path=path, device="cpu")
    return _runtime_cache[key]


def _safe_json(value: Any) -> Any:
    """Convert numpy/pandas/NaN values to JSON-safe Python types."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return [_safe_json(v) for v in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat() if pd.notna(value) else None
    if isinstance(value, dict):
        return {k: _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if pd.isna(value):
        return None
    return value


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class CatalogEvent(BaseModel):
    event_group_id: str
    danger_tier: int
    danger_label: str
    location_name: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    initial_magnitude: float | None = None
    max_magnitude: float | None = None
    initial_depth_km: float | None = None
    coastal_proximity_index: float | None = None


class SimulateRequest(BaseModel):
    event_group_id: str | None = Field(None, description="Specific event or null for random")
    danger_filter: Literal["All", "No Threat", "Potential Threat", "Confirmed Threat"] = Field(
        "All",
        description="All | No Threat | Potential Threat | Confirmed Threat",
    )
    agent_type: Literal["ppo", "rule", "manual"] = Field(
        "ppo",
        description="ppo | rule | manual",
    )
    checkpoint_name: str | None = Field(None, description="PPO checkpoint name")
    seed: int = Field(42, description="Episode RNG seed")


class SensorState(BaseModel):
    id: str
    type: str
    status: str
    lat: float | None = None
    lon: float | None = None


class ObservationData(BaseModel):
    magnitude_estimate: float
    depth_estimate_km: float
    coastal_proximity_index: float
    wave_estimate_m: float
    buoy_confirmation: float
    tide_confirmation: float
    uncertainty: float
    time_fraction: float
    alert_level_norm: float
    cancel_issued_flag: float
    delta_magnitude: float
    delta_wave_m: float
    delta_uncertainty: float
    time_since_buoy_norm: float
    time_since_tide_norm: float


class EpisodeFrame(BaseModel):
    t: int
    time_min: float
    epicenter: dict[str, float]
    wave_radius_km: float
    sensors: list[SensorState]
    observation: dict[str, float]
    state_summary: str
    action: str
    action_index: int
    reward: float
    step_reward: float
    terminal_bonus: float | None
    cumulative_reward: float
    alert_level: str
    alert_level_index: int
    danger_tier: int
    danger_label: str
    valid_actions: list[str]
    action_mask: list[float]
    agent_probabilities: dict[str, float]
    value_estimate: float | None
    rule_recommendation: str
    done: bool
    missed_severe: bool
    false_warning: bool
    hidden_trace: dict[str, list[float]]


class EpisodeResponse(BaseModel):
    event_metadata: dict[str, Any]
    frames: list[EpisodeFrame]
    total_return: float
    outcome_summary: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TIME_GRID = [0.0, 2.0, 5.0, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0]


def _wave_radius_km(time_min: float, danger_tier: int, wave_est: float) -> float:
    """Approximate expanding wave radius for visualization."""
    if danger_tier == 0:
        return 0.0
    speed_km_per_min = 12.0 if danger_tier >= 2 else 6.0
    base = time_min * speed_km_per_min
    scale = min(1.0, wave_est / 0.18) if wave_est > 0.02 else 0.0
    return base * scale


def _state_summary(obs_dict: dict, alert: str, danger_tier: int) -> str:
    """Generate a human-readable state summary."""
    mag = obs_dict.get("magnitude_estimate", 0)
    wave = obs_dict.get("wave_estimate_m", 0)
    unc = obs_dict.get("uncertainty", 0)
    buoy = obs_dict.get("buoy_confirmation", 0)
    tide = obs_dict.get("tide_confirmation", 0)

    parts: list[str] = []
    if mag >= 7.8:
        parts.append("major earthquake")
    elif mag >= 7.2:
        parts.append("strong earthquake")
    elif mag >= 6.5:
        parts.append("moderate earthquake")
    else:
        parts.append("earthquake detected")

    if tide > 0.5:
        parts.append("tide gauge confirmed")
    if buoy > 0.5:
        parts.append("buoy anomaly detected")
    if wave >= 0.18:
        parts.append("significant wave activity")
    elif wave >= 0.06:
        parts.append("minor wave observed")

    if unc >= 0.6:
        parts.append("high uncertainty")
    elif unc <= 0.2:
        parts.append("low uncertainty")

    return " · ".join(parts)


def _synthetic_sensors(
    obs_dict: dict,
    epicenter_lat: float,
    epicenter_lon: float,
    time_min: float,
) -> list[dict]:
    """Generate synthetic sensor positions near epicenter."""
    sensors = []
    buoy_confirmed = obs_dict.get("buoy_confirmation", 0) > 0.5
    tide_confirmed = obs_dict.get("tide_confirmation", 0) > 0.5

    # Place buoys at offsets from epicenter
    buoy_offsets = [(1.5, 2.0), (-1.0, 2.5), (0.5, -1.8), (-1.8, -0.8)]
    for i, (dlat, dlon) in enumerate(buoy_offsets):
        sensors.append({
            "id": f"B{i + 1}",
            "type": "buoy",
            "status": "triggered" if buoy_confirmed and i == 0 else "monitoring",
            "lat": epicenter_lat + dlat,
            "lon": epicenter_lon + dlon,
        })

    # Place tide gauges along coast (positive latitude = coastal)
    gauge_offsets = [(2.5, 0.5), (-2.0, 1.5), (0.0, -2.5)]
    for i, (dlat, dlon) in enumerate(gauge_offsets):
        sensors.append({
            "id": f"G{i + 1}",
            "type": "tide_gauge",
            "status": "triggered" if tide_confirmed and i == 0 else "inactive",
            "lat": epicenter_lat + dlat,
            "lon": epicenter_lon + dlon,
        })

    return sensors


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/catalog")
def get_catalog() -> list[dict]:
    """Return the event catalog for the dropdown."""
    catalog = _get_catalog()
    events: list[dict] = []
    for _, row in catalog.iterrows():
        tier = int(row.get("danger_tier", 0))
        events.append(_safe_json({
            "event_group_id": str(row["event_group_id"]),
            "danger_tier": tier,
            "danger_label": DANGER_LABELS.get(tier, "Unknown"),
            "location_name": row.get("location_name"),
            "latitude": row.get("latitude"),
            "longitude": row.get("longitude"),
            "initial_magnitude": row.get("initial_magnitude"),
            "max_magnitude": row.get("max_magnitude"),
            "initial_depth_km": row.get("initial_depth_km"),
            "coastal_proximity_index": row.get("coastal_proximity_index"),
        }))
    return events


@app.get("/api/checkpoints")
def get_checkpoints() -> list[dict]:
    """Return available PPO checkpoint names."""
    checkpoints = available_checkpoint_paths()
    return [{"name": name, "path": str(path)} for name, path in checkpoints.items()]


@app.post("/api/simulate", response_model=EpisodeResponse)
def simulate_episode(req: SimulateRequest) -> dict:
    """Simulate a full episode and return all frames for replay."""
    catalog = _get_catalog().copy()

    # Normalise event_group_id to str for consistent lookup
    catalog["event_group_id"] = catalog["event_group_id"].astype(str)

    # Filter catalog
    if req.danger_filter != "All":
        tier_map = {"No Threat": 0, "Potential Threat": 1, "Confirmed Threat": 2}
        tier_val = tier_map.get(req.danger_filter)
        if tier_val is not None:
            catalog = catalog[catalog["danger_tier"] == tier_val].reset_index(drop=True)
        if catalog.empty:
            raise HTTPException(status_code=404, detail="No events match the filter")

    # Select event
    if req.event_group_id:
        subset = catalog[catalog["event_group_id"] == req.event_group_id]
        if subset.empty:
            raise HTTPException(status_code=404, detail=f"Event {req.event_group_id} not found")
        event_catalog = subset.reset_index(drop=True)
    else:
        # Random event from filtered catalog
        rng = np.random.default_rng(req.seed)
        idx = int(rng.integers(0, len(catalog)))
        event_catalog = catalog.iloc[[idx]].reset_index(drop=True)

    # Create env
    env = TsunamiGymEnv(event_catalog, seed=req.seed)
    obs, info = env.reset(seed=req.seed)

    # Get metadata
    meta = _safe_json(scenario_metadata(env))
    epicenter_lat = float(meta["latitude"]) if meta.get("latitude") is not None else -2.0
    epicenter_lon = float(meta["longitude"]) if meta.get("longitude") is not None else 120.0
    danger_tier = int(meta.get("danger_tier", 0))

    # Get hidden trace for full episode
    trace_df = hidden_trace_frame(env)
    hidden_trace = {
        col: trace_df[col].tolist()
        for col in trace_df.columns
    }

    # Load agent
    runtime = None
    if req.agent_type == "ppo":
        runtime = _get_runtime(req.checkpoint_name)
        runtime.reset()

    frames: list[dict] = []
    cumulative_reward = 0.0
    done = False

    for step_idx in range(12):
        if done:
            break

        obs_dict = observation_to_dict(obs)
        action_mask = info["action_mask"].tolist()
        valid_action_indices = info["valid_actions"]
        valid_action_names = [ACTION_NAMES[a] for a in valid_action_indices]

        # Get rule recommendation
        rule_act = int(rule_based_policy(obs, valid_action_indices, env.env))
        rule_rec_name = ACTION_NAMES[rule_act]

        # Get agent decision
        if req.agent_type == "ppo" and runtime:
            decision = commit_ppo_decision(runtime, obs, np.array(action_mask))
            action_idx = decision.action
            probs = {ACTION_NAMES[i]: float(decision.probabilities[i]) for i in range(6)}
            value_est = float(decision.value_estimate)
        elif req.agent_type == "rule":
            action_idx = rule_act
            probs = {name: (1.0 if i == action_idx else 0.0) for i, name in enumerate(ACTION_NAMES)}
            value_est = None
        else:
            # Manual mode: default to rule
            action_idx = rule_act
            probs = {name: (1.0 if i == action_idx else 0.0) for i, name in enumerate(ACTION_NAMES)}
            value_est = None

        # Step
        next_obs, reward, terminated, truncated, next_info = env.step(action_idx)
        cumulative_reward += reward
        done = terminated or truncated

        # Split terminal bonus from step reward on the final frame
        terminal_bonus: float | None = None
        if done:
            tb, _ = env.env._terminal_outcome()
            terminal_bonus = float(tb)
            step_reward = float(reward - tb)
        else:
            step_reward = float(reward)

        # Value estimate is 0 at terminal step (no future rewards)
        if done and value_est is not None:
            value_est = 0.0

        time_min = TIME_GRID[step_idx] if step_idx < len(TIME_GRID) else 60.0
        alert_idx = int(obs_dict.get("alert_level_norm", 0) * 4)
        # After action, the resulting alert might differ
        if step_idx + 1 < 12 and not done:
            next_obs_dict = observation_to_dict(next_obs)
            resulting_alert_idx = int(round(next_obs_dict.get("alert_level_norm", 0) * 4))
        else:
            resulting_alert_idx = alert_idx

        # Apply action to get the resulting alert (approximate)
        if action_idx == 1:  # escalate
            resulting_alert_idx = min(alert_idx + 1, 4)
        elif action_idx == 2:  # deescalate
            resulting_alert_idx = max(alert_idx - 1, 0)
        elif action_idx == 3:  # issue_watch
            resulting_alert_idx = 2
        elif action_idx == 4:  # issue_warning
            resulting_alert_idx = 4
        elif action_idx == 5:  # cancel
            resulting_alert_idx = 0

        sensors = _synthetic_sensors(obs_dict, epicenter_lat, epicenter_lon, time_min)
        wave_radius = _wave_radius_km(time_min, danger_tier, obs_dict.get("wave_estimate_m", 0))

        frame = {
            "t": step_idx,
            "time_min": time_min,
            "epicenter": {"lat": epicenter_lat, "lon": epicenter_lon},
            "wave_radius_km": wave_radius,
            "sensors": sensors,
            "observation": _safe_json(obs_dict),
            "state_summary": _state_summary(obs_dict, ALERT_LEVELS[resulting_alert_idx], danger_tier),
            "action": ACTION_NAMES[action_idx],
            "action_index": action_idx,
            "reward": float(reward),
            "step_reward": step_reward,
            "terminal_bonus": terminal_bonus,
            "cumulative_reward": cumulative_reward,
            "alert_level": ALERT_LEVELS[resulting_alert_idx],
            "alert_level_index": resulting_alert_idx,
            "danger_tier": danger_tier,
            "danger_label": DANGER_LABELS.get(danger_tier, "Unknown"),
            "valid_actions": valid_action_names,
            "action_mask": action_mask,
            "agent_probabilities": _safe_json(probs),
            "value_estimate": _safe_json(value_est),
            "rule_recommendation": rule_rec_name,
            "done": done,
            "missed_severe": bool(next_info.get("missed_severe", False)),
            "false_warning": bool(next_info.get("false_warning", False)),
            "hidden_trace": _safe_json(hidden_trace),
        }
        frames.append(frame)
        obs = next_obs
        info = next_info

    # Outcome summary
    last_frame = frames[-1] if frames else {}
    if last_frame.get("missed_severe"):
        outcome = "MISSED SEVERE EVENT - Warning was not issued for a confirmed threat"
    elif last_frame.get("false_warning"):
        outcome = "FALSE WARNING - Warning issued for a non-threatening event"
    elif danger_tier == 2 and any(f["alert_level"] == "warning" for f in frames):
        first_warn = next((f["t"] for f in frames if f["alert_level"] == "warning"), None)
        outcome = f"CORRECT WARNING - Severe threat detected and warning issued at step {first_warn}"
    elif danger_tier == 0 and all(f["alert_level_index"] <= 1 for f in frames):
        outcome = "CORRECT RESTRAINT - No unnecessary alerts on non-threatening event"
    elif danger_tier == 1:
        if any(f["alert_level_index"] >= 2 for f in frames):
            outcome = "WATCH ISSUED - Potential threat monitored with appropriate alerting"
        else:
            outcome = "POTENTIAL THREAT - Event monitored without formal watch"
    else:
        outcome = f"Episode completed with alert level {last_frame.get('alert_level', 'monitor')}"

    return _safe_json({
        "event_metadata": meta,
        "frames": frames,
        "total_return": cumulative_reward,
        "outcome_summary": outcome,
    })


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
