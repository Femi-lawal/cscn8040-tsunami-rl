# Option A Pitch: RL for Tsunami Early Warning Decision Support

## Environment

Our environment is a time-stepped tsunami warning simulator informed by real BMKG InaTSP bulletin history, plus external enrichment from NOAA NCEI tsunami records and the USGS earthquake catalog. Each episode begins with a major earthquake and then reveals new evidence over time, such as revised magnitude, revised depth, a coastal exposure proxy, model-based wave estimate, buoy confirmation, tide-gauge confirmation, uncertainty, elapsed time, and the current alert level.

## Use Case

The use case is a decision-support system for a tsunami warning center. The agent decides whether to keep monitoring, issue an information statement, escalate to watch or advisory, issue a warning, or cancel when the threat is low.

## Target

The target is the optimal alert policy. In practical terms, that means:

- detect truly dangerous events
- escalate fast enough on severe cases
- reduce missed warnings
- reduce false alarms
- avoid unnecessary alert switching

## Utility / Reward

The reward function reflects warning-center priorities:

- strong positive reward for correct early warning on severe tsunami cases
- medium positive reward for correct watch or advisory on moderate cases
- positive reward for correct cancellation or low-alert handling on no-threat events
- per-step penalties for delaying action during dangerous events
- large penalties for false warnings
- very large penalties for missing a confirmed dangerous event
- penalties for excessive alert changes

The current implementation uses mostly outcome-driven reward. During the episode, it only applies small local penalties for delay, unnecessary alert switching, ignored buoy or tide evidence, and obvious over-warning. The dominant reward comes from terminal outcomes such as missed severe events, timely severe warnings, timely potential-threat watches, and false alarms on no-threat cases.

## RL Concepts We Implement

- **Markov Decision Process:** states, actions, transitions, and rewards define the warning problem.
- **Bellman equations:** the value of an alert depends on both immediate payoff and downstream consequences.
- **Dynamic programming:** a small discrete toy MDP is solved with value iteration as an oracle reference policy.
- **Monte Carlo-style evaluation:** trained policies are evaluated over full held-out episodes using average episodic return and operational metrics. This is evaluation by rollout, not Monte Carlo control.
- **Temporal-difference learning:** the main learning setup updates values from partial episodes instead of waiting until the end.
- **SARSA:** used as the on-policy control baseline. We report both a pure greedy SARSA policy and a rule-backed safe deployment variant.
- **Q-learning:** used as the off-policy control baseline. We report both a pure greedy Q-learning policy and a rule-backed safe deployment variant.
- **Exploration vs exploitation:** epsilon-greedy exploration is used during training and decayed over time.

## Scope Limits

This prototype does not implement DQN, actor-critic, policy gradients, or a hydrodynamic tsunami simulator. It is best described as a BMKG-seeded warning-policy environment with tabular RL baselines, external NOAA/USGS enrichment, train-only synthetic augmentation, a strong rule-based comparator, and an optional hybrid safe-deployment layer that falls back to rules when learned values are uncertain.

## Reporting Note

The project now freezes a pre-upgrade baseline in `baseline_results/` and reports pure RL, safe hybrid RL, and the rule baseline separately. The DP policy should still be presented only as a fully observed oracle reference, not as a like-for-like operational comparator.

## Current best result

The strongest current pure-RL result comes from the bundled local reproducibility PPO-LSTM run. On the held-out test split, `ppo_lstm_pure` reached about `25.34` average return with `0.00` severe miss rate and `0.00` false warning rate, improving on the rule baseline's roughly `20.40` average return on the same split.

## Why This Is a Strong Option A Project

This project uses reinforcement learning where it naturally fits: sequential warning decisions under uncertainty. We are not claiming RL replaces tsunami physics. Instead, we use RL for the policy layer, where the real challenge is deciding when to wait, escalate, warn, or cancel as evidence changes.
