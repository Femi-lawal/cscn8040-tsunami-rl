# Tsunami Warning RL Console — README

An interactive dashboard for replaying and inspecting episodes from a Reinforcement Learning agent trained to issue tsunami warnings.

---

## How It Works

A seismic event occurs. The agent has **12 decision steps** spread across the first 60 minutes (`T+0, 2, 5, 8, 12, 16, 20, 25, 30, 40, 50, 60 min`). At each step it reads sensor data and decides whether to escalate, hold, or cancel the public alert level. The goal: issue timely warnings for real threats while avoiding false alarms.

---

## Where to Find RL Concepts in the Dashboard

### States (Observations)

**Panel → Telemetry → OBSERVATIONS**

Each step exposes 15 features to the agent:

| Feature                   | What it tells the agent                    |
| ------------------------- | ------------------------------------------ |
| `magnitude_estimate`      | Current earthquake magnitude               |
| `depth_estimate_km`       | Hypocentral depth                          |
| `coastal_proximity_index` | How close to the coast (0–1)               |
| `wave_estimate_m`         | Estimated wave height                      |
| `buoy_confirmation`       | 1 if a buoy anomaly triggered              |
| `tide_confirmation`       | 1 if a tide gauge confirmed                |
| `uncertainty`             | Observation uncertainty (high → risky)     |
| `time_fraction`           | Progress through the episode (0–1)         |
| `alert_level_norm`        | Current alert level / 4                    |
| `cancel_issued_flag`      | 1 if cancel was previously issued          |
| `delta_magnitude`         | Change in magnitude from last step         |
| `delta_wave_m`            | Change in wave estimate from last step     |
| `delta_uncertainty`       | Change in uncertainty from last step       |
| `time_since_buoy_norm`    | Normalized time since buoy first triggered |
| `time_since_tide_norm`    | Normalized time since tide first triggered |

Source: `src/environment.py` observation builder, `src/viewer.py` OBSERVATION_FIELDS.

---

### Actions

**Panel → Timeline cards & Telemetry → ACTION**

The agent chooses from 6 discrete actions:

| Action          | Effect                           |
| --------------- | -------------------------------- |
| `hold`          | Keep current alert level         |
| `escalate`      | Raise alert by one level         |
| `deescalate`    | Lower alert by one level         |
| `issue_watch`   | Jump to at least Watch (level 2) |
| `issue_warning` | Jump to Warning (level 4)        |
| `cancel`        | Drop to Monitor, set cancel flag |

Not all actions are available every step — **action masking** enforces cooldown (must hold an alert level for ≥ 2 steps) and evidence thresholds (e.g. `issue_warning` requires buoy or tide confirmation).

Source: `src/environment.py` `ACTION_NAMES`, `valid_actions()`, and `action_mask()`.

---

### Alert Levels

**Panel → Map alert banner & Timeline cards**

The banner colour at the top of the map shows the current alert level:

| Level | Name     | Colour    |
| ----- | -------- | --------- |
| 0     | Monitor  | Dark/grey |
| 1     | Info     | Blue      |
| 2     | Watch    | Yellow    |
| 3     | Advisory | Orange    |
| 4     | Warning  | Red       |

---

### Rewards

**Panel → Telemetry → REWARD / CUMULATIVE**

Per-step costs (selected):

| Condition                          | Reward |
| ---------------------------------- | ------ |
| Each step (base cost)              | −0.1   |
| Changing alert level               | −0.5   |
| Invalid action                     | −12.0  |
| Buoy triggered but alert < Watch   | −3.0   |
| Tide triggered but alert < Warning | −8.0   |

Terminal rewards (at episode end):

| Outcome                           | Reward                         |
| --------------------------------- | ------------------------------ |
| Confirmed Threat correctly warned | up to +140 (decays with delay) |
| Confirmed Threat missed           | −250                           |
| No Threat, stayed at Monitor/Info | +15                            |
| No Threat, false Warning issued   | −50                            |

The **CUMULATIVE** metric in the telemetry panel is the running sum across all steps. The final value equals **total_return** — the single number the agent was trained to maximise.

Source: `src/environment.py` `_step_reward()` and `_terminal_outcome()`.

---

### Transitions

**Panel → Timeline (step-by-step replay)**

Each timeline card represents one state → action → next-state transition:

```
Step 1 (T+0 min)  →  action: hold  →  Step 2 (T+2 min)
Step 2 (T+2 min)  →  action: escalate  →  Step 3 (T+5 min)
...
Step 12 (T+60 min) →  terminal reward  →  Episode ends
```

Use **Step** to advance one transition at a time, or **Play** to auto-advance. The timeline cards accumulate showing the full action-reward history.

---

### Policy (Agent Diagnostics)

**Panel → Telemetry → AGENT DIAGNOSTICS**

- **PPO agent**: probability bars show π(a|s) for all 6 actions + the critic's value estimate V(s). Masked actions show 0%.
- **Rule-based agent**: shows the rule recommendation and valid actions, but hides learned-policy probability bars and value estimates.

---

## Dashboard Layout

```
┌──────────────────┬─────────────────────────────────────────┐
│  CONTROL PANEL   │               MAP                      │
│                  │  • Epicenter marker (red)               │
│  Threat Filter   │  • Expanding wave radius                │
│  Scenario        │  • Sensor markers (buoys, tide gauges)  │
│  Agent Type      │  • Alert banner (colour = alert level)  │
│  Checkpoint      │  • Event info overlay                   │
│  Seed            ├─────────────────────────────────────────┤
│                  │           TELEMETRY                     │
│  Speed controls  │  Step • Alert • Action • Reward         │
│  Start / Play /  │  Observations table                    │
│  Pause / Step /  │  Agent diagnostics (π, V)              │
│  Reset           │  Outcome banner (at episode end)        │
│                  ├─────────────────────────────────────────┤
│                  │         EPISODE TIMELINE                │
│                  │  Card per step: action, alert, reward   │
└──────────────────┴─────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Run with Docker
docker run -d --name tsunami-console -p 4000:3000 -p 4001:8000 tsunami-rl-console
# Open http://localhost:4000

# Or run locally
cd option_a_tsunami_rl
python -m uvicorn api.server:app --port 8000 &
cd console && npm run dev
# Open http://localhost:3000
```

1. Pick a **Threat Filter** (or leave All)
2. Choose **PPO-LSTM** or **Rule-Based** agent
2.5. If you stay on PPO, pick from the curated checkpoint list:
   `Recommended PPO Policy`, `Stable PPO Policy`, or `Baseline PPO Policy`
3. Click **Start Episode**
4. Use **Step** to advance one transition at a time, or **Play** to auto-advance
5. Observe how observations, actions, rewards, and alert levels evolve

---

## Key Files

| File                      | RL Concept                                                            |
| ------------------------- | --------------------------------------------------------------------- |
| `src/environment.py`      | MDP definition: states, actions, rewards, transitions, action masking |
| `src/gym_env.py`          | Gymnasium wrapper (observation/action spaces)                         |
| `src/agents.py`           | Rule-based policy, Q-table policies                                   |
| `src/deep_agents.py`      | PPO-LSTM network architecture and training                            |
| `src/viewer.py`           | Constants (OBSERVATION_FIELDS, DANGER_LABELS), runtime inference      |
| `api/server.py`           | API that runs episodes and returns frame-by-frame data                |
| `console/src/components/` | Dashboard UI components                                               |
