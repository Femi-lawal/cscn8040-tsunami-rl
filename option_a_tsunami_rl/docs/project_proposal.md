# CSCN 8040 Project Proposal

**Title:** Reinforcement Learning for Tsunami Early Warning Decision Support

**Author:** Femi Lawal

**Date:** March 2026

## 1. Problem Statement

Tsunami warning centers face a sequential decision problem under uncertainty. After a major earthquake, decision-makers do not immediately know whether the event will generate a dangerous tsunami. Instead, they receive evolving evidence such as revised magnitude, revised depth, coastal exposure, buoy readings, tide-gauge confirmations, and uncertainty estimates.

The central challenge is not only identifying dangerous events, but deciding when to stay in monitoring mode, when to escalate, when to issue a warning, and when to cancel. Warning too slowly can be catastrophic. Warning too aggressively can produce false alarms, unnecessary evacuations, and public trust erosion.

This project proposes a reinforcement learning approach for the warning-policy layer of tsunami early warning, where the task is to learn better alert decisions over time as new evidence arrives.

## 2. Project Goal

The goal is to learn and evaluate a warning policy that:

- detects dangerous tsunami-generating events early
- reduces missed severe cases
- reduces false warnings on non-threatening events
- manages ambiguous "potential threat" cases more effectively
- avoids unnecessary alert switching

The project is not trying to replace tsunami physics models. Instead, it focuses on the operational decision layer: given evolving evidence, what alert action should be taken now?

## 3. Proposed Environment

The project uses a custom time-stepped tsunami warning environment framed as a partially observable Markov decision process.

### Environment design

Each episode represents the first hour after a significant earthquake and unfolds over 12 non-uniform timesteps:

`[0, 2, 5, 8, 12, 16, 20, 25, 30, 40, 50, 60]` minutes

The agent does not observe the true event severity directly. Instead, it receives a continuous observation vector built from warning-relevant evidence, including:

- magnitude estimate
- depth estimate
- coastal exposure proxy
- wave estimate
- buoy confirmation
- tide-gauge confirmation
- uncertainty
- elapsed-time features
- current alert level
- short-term change features such as deltas and sensor ages

The agent selects from a discrete action space:

- `hold`
- `escalate`
- `deescalate`
- `issue_watch`
- `issue_warning`
- `cancel`

To keep the policy operationally reasonable, invalid actions are masked based on the current alert state and alert-hold rules.

### Reward design

The reward is primarily outcome-driven rather than schedule-imitation-based. It emphasizes:

- strong positive reward for timely severe-event warnings
- moderate reward for timely watch-level handling of potential threats
- strong penalties for missed severe events
- strong penalties for false warnings on non-threat events
- smaller penalties for delay, ignored strong evidence, and excessive alert changes

This makes the project a true sequential decision problem rather than a one-step classifier.

## 4. Data Sources

The environment is seeded from real warning and event archives. The main data source is the BMKG InaTSP bulletin archive, which provides earthquake metadata, bulletin sequences, and warning-related information for Indonesian events.

To strengthen the event catalog, the project also uses external official sources:

- NOAA NCEI tsunami event records
- USGS earthquake catalog data

These sources are used to enrich BMKG events with additional tsunami and earthquake context. The training pipeline also supports train-only synthetic augmentation derived from the real training split.

The project should be described as **data-informed**, not as a fully empirical hydrodynamic tsunami simulator. Some parts of the sequential evidence process are still modeled heuristically because wave-observation coverage in the archive is sparse.

## 5. Methods

The project compares several policy types rather than relying on a single RL method.

### Baselines

- **Rule-based expert policy:** a hand-crafted operational baseline using thresholds on magnitude, depth, wave estimates, and sensor confirmations
- **Dynamic programming oracle:** a small fully observed toy reference policy used only as an upper-bound comparison, not as a fair operational baseline

### Tabular RL baselines

- **Q-learning**
- **SARSA**

These are included as interpretable reinforcement learning baselines. They are useful for showing the limits of discretized tabular methods on a partially observed warning task.

### Main deep RL method

The main learned policy is a **masked recurrent PPO agent with LSTM memory**. This is a natural fit because:

- the warning task is partially observed
- evidence evolves over time
- the agent benefits from short-term memory
- action masking helps enforce operational constraints during learning

The PPO agent supports observation normalization, curriculum-style training, and longer GPU-based training runs when needed.

## 6. Evaluation Plan

The project evaluates policies on a class-aware chronological split so that training, validation, and test reflect older-to-newer events within each danger class.

Policies are compared using operationally meaningful metrics such as:

- average episodic return
- severe miss rate
- false warning rate
- average warning step on severe events
- average watch step on potential-threat events
- alert change count
- a safety-aware composite score

The main comparisons are:

- rule-based policy vs tabular RL
- pure tabular RL vs safe rule-backed tabular deployment
- tabular RL vs PPO-LSTM
- PPO-LSTM vs rule-based policy

The project also keeps a frozen baseline snapshot so later improvements can be judged against a fixed reference.

## 7. Deliverables

The expected deliverables are:

1. a custom POMDP tsunami warning environment
2. a processed BMKG-centered event catalog with NOAA and USGS enrichment
3. rule-based, tabular RL, and PPO-LSTM policy implementations
4. evaluation outputs and comparison tables
5. an interactive dashboard for replaying episodes and inspecting states, actions, and outcomes
6. a written analysis of performance, safety tradeoffs, and limitations

## 8. Scope and Limits

This proposal is intentionally realistic about what the project is and is not.

The project **is**:

- a reinforcement learning warning-policy project
- data-informed and operationally motivated
- focused on sequential alert decisions under uncertainty

The project **is not**:

- a full tsunami physics simulator
- a direct replacement for official warning centers
- a claim that RL alone should control real-world alerts without safeguards

The strongest framing is that the project studies whether reinforcement learning can improve the **decision-support layer** of tsunami early warning, especially in ambiguous cases where timing and escalation matter.
