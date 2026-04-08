from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from option_a_tsunami_rl.src.environment import ACTION_NAMES, ALERT_LEVELS
    from option_a_tsunami_rl.src.gym_env import TsunamiGymEnv
    from option_a_tsunami_rl.src.viewer import (
        DANGER_LABELS,
        alert_level_label,
        available_checkpoint_paths,
        build_runtime,
        commit_ppo_decision,
        filtered_catalog,
        hidden_trace_frame,
        history_frame,
        keep_runtime_in_sync,
        load_event_catalog,
        observation_frame,
        observation_to_dict,
        preview_ppo_decision,
        probabilities_frame,
        rule_decision,
        scenario_metadata,
    )
except ModuleNotFoundError:
    from src.environment import ACTION_NAMES, ALERT_LEVELS
    from src.gym_env import TsunamiGymEnv
    from src.viewer import (
        DANGER_LABELS,
        alert_level_label,
        available_checkpoint_paths,
        build_runtime,
        commit_ppo_decision,
        filtered_catalog,
        hidden_trace_frame,
        history_frame,
        keep_runtime_in_sync,
        load_event_catalog,
        observation_frame,
        observation_to_dict,
        preview_ppo_decision,
        probabilities_frame,
        rule_decision,
        scenario_metadata,
    )


st.set_page_config(
    page_title="Tsunami RL Viewer",
    page_icon="🌊",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def _catalog() -> pd.DataFrame:
    return load_event_catalog()


@st.cache_resource(show_spinner=False)
def _runtime(checkpoint_path: str) -> Any:
    return build_runtime(Path(checkpoint_path), device="cpu")


def _event_options(catalog: pd.DataFrame) -> dict[str, str | None]:
    options: dict[str, str | None] = {"Random from filtered catalog": None}
    for _, row in catalog.sort_values(["danger_tier", "origin_time_utc"], na_position="last").iterrows():
        origin = row.get("origin_time_utc")
        origin_label = origin.strftime("%Y-%m-%d") if hasattr(origin, "strftime") and not pd.isna(origin) else "unknown-date"
        label = f"{row['event_group_id']} | {row.get('danger_label', 'unknown')} | {origin_label}"
        options[label] = str(row["event_group_id"])
    return options


def _reset_episode(
    catalog_view: pd.DataFrame,
    checkpoint_path: str,
    episode_seed: int,
) -> None:
    env = TsunamiGymEnv(catalog_view, seed=int(episode_seed), weight_column=None)
    observation, info = env.reset(seed=int(episode_seed))
    runtime = _runtime(checkpoint_path)
    runtime.reset()
    st.session_state.viewer_state = {
        "env": env,
        "observation": observation,
        "info": info,
        "runtime": runtime,
        "history": [],
        "cumulative_reward": 0.0,
        "done": False,
        "last_reward": 0.0,
        "terminal_info": None,
    }


def _render_metric_row(metadata: dict[str, Any], state: dict[str, Any]) -> None:
    env: TsunamiGymEnv = state["env"]
    current_step = int(env.env.current_step)
    current_time = float(env.env.time_grid_min[min(current_step, len(env.env.time_grid_min) - 1)])
    alert_level = int(env.env.current_alert)
    danger_label = metadata.get("danger_label", "Unknown")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Threat Class", danger_label)
    col2.metric("Current Alert", alert_level_label(alert_level))
    col3.metric("Step / Horizon", f"{current_step} / {env.env.horizon}")
    col4.metric("Current Time", f"{current_time:.0f} min")
    col5.metric("Cumulative Reward", f"{state['cumulative_reward']:.2f}")


def _make_history_record(
    state: dict[str, Any],
    action: int,
    controller_label: str,
    ppo_preview: Any,
    rule_preview: Any,
    reward: float,
    info_before_step: dict[str, Any],
    done: bool,
    terminal_info: dict[str, Any] | None,
) -> dict[str, Any]:
    env: TsunamiGymEnv = state["env"]
    observation_used = np.asarray(info_before_step["observation_used"], dtype=np.float32)
    observation_values = observation_to_dict(observation_used)
    step_index = int(env.env.current_step - 1)
    time_min = float(env.env.time_grid_min[max(step_index, 0)])
    resulting_alert = int(env.env.current_alert)

    return {
        "step": step_index,
        "time_min": time_min,
        "controller": controller_label,
        "action_name": ACTION_NAMES[action],
        "resulting_alert": ALERT_LEVELS[resulting_alert],
        "reward": float(reward),
        "cumulative_reward": float(state["cumulative_reward"]),
        "ppo_action": ppo_preview.action_name,
        "ppo_value_estimate": ppo_preview.value_estimate,
        "rule_action": rule_preview.action_name,
        "magnitude_estimate": observation_values["magnitude_estimate"],
        "depth_estimate_km": observation_values["depth_estimate_km"],
        "wave_estimate_m": observation_values["wave_estimate_m"],
        "buoy_confirmation": int(round(observation_values["buoy_confirmation"])),
        "tide_confirmation": int(round(observation_values["tide_confirmation"])),
        "uncertainty": observation_values["uncertainty"],
        "invalid_action": bool(info_before_step["invalid_action"]),
        "done": bool(done),
        "missed_severe": None if not terminal_info else terminal_info.get("missed_severe"),
        "false_warning": None if not terminal_info else terminal_info.get("false_warning"),
    }


def _advance_episode(controller_label: str, manual_action: int | None = None) -> None:
    state = st.session_state.viewer_state
    if state["done"]:
        return

    env: TsunamiGymEnv = state["env"]
    runtime = state["runtime"]
    observation = np.asarray(state["observation"], dtype=np.float32)
    info = dict(state["info"])
    action_mask = np.asarray(info["action_mask"], dtype=np.float32)
    valid_actions = list(info["valid_actions"])

    ppo_preview = preview_ppo_decision(runtime, observation, action_mask)
    rule_preview = rule_decision(observation, valid_actions, env)

    if controller_label == "PPO-LSTM":
        decision = commit_ppo_decision(runtime, observation, action_mask)
        action = int(decision.action)
    elif controller_label == "Rule-based":
        keep_runtime_in_sync(runtime, observation, action_mask)
        action = int(rule_preview.action)
    else:
        keep_runtime_in_sync(runtime, observation, action_mask)
        action = int(manual_action if manual_action is not None else valid_actions[0])

    next_observation, reward, terminated, truncated, next_info = env.step(action)
    done = bool(terminated or truncated)

    state["cumulative_reward"] += float(reward)
    state["last_reward"] = float(reward)
    state["terminal_info"] = dict(next_info) if done else None
    record = _make_history_record(
        state,
        action,
        controller_label,
        ppo_preview,
        rule_preview,
        reward,
        next_info,
        done,
        state["terminal_info"],
    )
    state["history"].append(record)
    state["observation"] = next_observation
    state["info"] = next_info
    state["done"] = done


def _render_policy_panel(title: str, decision: Any, action_mask: np.ndarray) -> None:
    st.subheader(title)
    st.metric("Recommended Action", decision.action_name)
    value_text = "n/a" if math.isnan(decision.value_estimate) else f"{decision.value_estimate:.3f}"
    st.metric("Value Estimate", value_text)
    st.dataframe(
        probabilities_frame(decision, action_mask),
        use_container_width=True,
        hide_index=True,
    )


catalog = _catalog()
checkpoints = available_checkpoint_paths()
if not checkpoints:
    st.error("No PPO checkpoints were found under outputs/models.")
    st.stop()

st.title("Tsunami Warning Agent Viewer")
st.caption(
    "Interactive Gym + Streamlit viewer for the tsunami early-warning environment. "
    "You can inspect event metadata, evolving evidence, valid actions, and the PPO or rule policy decisions step by step."
)

with st.sidebar:
    st.header("Episode Setup")
    controller_mode = st.selectbox("Controller", ["PPO-LSTM", "Rule-based", "Manual"], key="controller_mode")
    checkpoint_label = st.selectbox("PPO checkpoint", list(checkpoints.keys()), key="checkpoint_label")
    danger_filter = st.selectbox("Threat filter", ["All", "No Threat", "Potential Threat", "Confirmed Threat"], index=0)

filtered_view = filtered_catalog(catalog, danger_filter)
event_options = _event_options(filtered_view)
with st.sidebar:
    selected_event_label = st.selectbox("Scenario", list(event_options.keys()), key="event_choice")
    episode_seed = int(st.number_input("Episode seed", min_value=1, value=88, step=1))

selected_event_id = event_options[selected_event_label]
catalog_view = filtered_catalog(catalog, danger_filter, event_group_id=selected_event_id)
if catalog_view.empty:
    st.error("No events matched the current filters.")
    st.stop()

signature = (
    controller_mode,
    checkpoint_label,
    danger_filter,
    selected_event_id or "__random__",
    episode_seed,
)
if "viewer_signature" not in st.session_state or st.session_state.viewer_signature != signature:
    _reset_episode(catalog_view, str(checkpoints[checkpoint_label]), episode_seed)
    st.session_state.viewer_signature = signature

state = st.session_state.viewer_state
current_info = state["info"]
current_observation = np.asarray(state["observation"], dtype=np.float32)
current_mask = np.asarray(current_info["action_mask"], dtype=np.float32)
current_valid_actions = list(current_info["valid_actions"])

with st.sidebar:
    st.header("Controls")
    if st.button("Reset Episode", key="reset_episode_button", use_container_width=True):
        _reset_episode(catalog_view, str(checkpoints[checkpoint_label]), episode_seed)
        st.session_state.viewer_signature = signature
        st.rerun()
    policy_disabled = bool(state["done"]) or controller_mode == "Manual"
    if st.button("Policy Step", key="policy_step_button", disabled=policy_disabled, use_container_width=True):
        _advance_episode(controller_mode)
        st.rerun()
    run_disabled = bool(state["done"]) or controller_mode == "Manual"
    if st.button("Run To End", key="run_to_end_button", disabled=run_disabled, use_container_width=True):
        while not st.session_state.viewer_state["done"]:
            _advance_episode(controller_mode)
        st.rerun()

    manual_labels = [ACTION_NAMES[action] for action in current_valid_actions] if current_valid_actions else ["hold"]
    manual_label = st.selectbox("Manual action", manual_labels, key="manual_action_choice")
    manual_disabled = bool(state["done"])
    if st.button("Apply Manual Action", key="manual_step_button", disabled=manual_disabled, use_container_width=True):
        chosen_action = ACTION_NAMES.index(manual_label)
        _advance_episode("Manual", manual_action=chosen_action)
        st.rerun()

metadata = scenario_metadata(state["env"])
_render_metric_row(metadata, state)

if metadata.get("latitude") is not None and metadata.get("longitude") is not None:
    st.map(pd.DataFrame({"lat": [metadata["latitude"]], "lon": [metadata["longitude"]]}), zoom=4)

meta_col1, meta_col2, meta_col3 = st.columns(3)
meta_col1.subheader("Event Metadata")
meta_col1.write(
    {
        "event_group_id": metadata.get("event_group_id"),
        "location_name": metadata.get("location_name"),
        "origin_time_utc": str(metadata.get("origin_time_utc")),
        "target_alert_level": metadata.get("target_alert_level"),
        "coastal_proximity_index": metadata.get("coastal_proximity_index"),
        "observed_max_wave_m": metadata.get("observed_max_wave_m"),
        "wave_imputed_flag": metadata.get("wave_imputed_flag"),
    }
)
meta_col2.subheader("Operational Context")
meta_col2.write(
    {
        "bulletin_count": metadata.get("bulletin_count"),
        "first_bulletin_delay_min": metadata.get("first_bulletin_delay_min"),
        "final_bulletin_delay_min": metadata.get("final_bulletin_delay_min"),
        "sea_level_confirmed_flag": metadata.get("sea_level_confirmed_flag"),
        "has_threat_assessment": metadata.get("has_threat_assessment"),
    }
)
meta_col3.subheader("Controller State")
meta_col3.write(
    {
        "controller_mode": controller_mode,
        "checkpoint": checkpoint_label,
        "valid_actions": [ACTION_NAMES[action] for action in current_valid_actions],
        "last_reward": round(float(state["last_reward"]), 3),
        "done": bool(state["done"]),
    }
)

if not state["done"]:
    ppo_preview = preview_ppo_decision(state["runtime"], current_observation, current_mask)
    rule_preview = rule_decision(current_observation, current_valid_actions, state["env"])
    decision_col1, decision_col2 = st.columns(2)
    with decision_col1:
        _render_policy_panel("PPO Recommendation", ppo_preview, current_mask)
    with decision_col2:
        _render_policy_panel("Rule Baseline Recommendation", rule_preview, current_mask)
else:
    st.success("Episode finished. Use Reset Episode to start another scenario.")

trace_df = hidden_trace_frame(state["env"])
trace_col1, trace_col2 = st.columns(2)
with trace_col1:
    st.subheader("Simulated Evidence Trace")
    st.line_chart(
        trace_df.set_index("time_min")[["magnitude_estimate", "wave_estimate_m", "uncertainty"]],
        use_container_width=True,
    )
with trace_col2:
    st.subheader("Sensor Confirmations")
    st.line_chart(
        trace_df.set_index("time_min")[["buoy_confirmation", "tide_confirmation"]],
        use_container_width=True,
    )

obs_col1, obs_col2 = st.columns([1, 2])
with obs_col1:
    st.subheader("Current Observation")
    st.dataframe(observation_frame(current_observation), use_container_width=True, hide_index=True)
with obs_col2:
    st.subheader("Episode History")
    history_df = history_frame(state["history"])
    if history_df.empty:
        st.info("No actions taken yet.")
    else:
        st.dataframe(history_df, use_container_width=True, hide_index=True)

if state["done"] and state["terminal_info"] is not None:
    st.subheader("Terminal Outcome")
    st.write(
        {
            "missed_severe": state["terminal_info"].get("missed_severe"),
            "false_warning": state["terminal_info"].get("false_warning"),
            "max_alert": alert_level_label(int(state["terminal_info"].get("max_alert", 0))),
            "target_alert_level": alert_level_label(int(state["terminal_info"].get("target_alert_level", 0))),
            "wave_imputed_flag": state["terminal_info"].get("wave_imputed_flag"),
        }
    )
