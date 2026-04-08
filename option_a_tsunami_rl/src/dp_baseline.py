from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .environment import ACTION_NAMES, ALERT_LEVELS, CANCEL, DEESCALATE, ESCALATE, HOLD, ISSUE_WARNING, ISSUE_WATCH


@dataclass(frozen=True)
class ToyState:
    time_index: int
    danger_tier: int
    current_alert: int
    cancel_issued: bool


class ToyTsunamiMDP:
    def __init__(self, horizon: int = 6, gamma: float = 0.96):
        self.horizon = horizon
        self.gamma = gamma
        self.actions = list(range(len(ACTION_NAMES)))
        self.states = [
            ToyState(time_index=t, danger_tier=d, current_alert=a, cancel_issued=c)
            for t in range(horizon)
            for d in range(3)
            for a in range(len(ALERT_LEVELS))
            for c in (False, True)
        ]

    @staticmethod
    def target_alert_from_danger(danger_tier: int) -> int:
        return 0 if danger_tier == 0 else (2 if danger_tier == 1 else 4)

    def valid_actions(self, state: ToyState) -> list[int]:
        valid = [HOLD]
        if state.current_alert < len(ALERT_LEVELS) - 1:
            valid.append(ESCALATE)
            valid.append(ISSUE_WARNING)
        if state.current_alert > 0:
            valid.append(DEESCALATE)
            valid.append(CANCEL)
        if state.current_alert < 2:
            valid.append(ISSUE_WATCH)
        return sorted(set(valid))

    def apply_action(self, state: ToyState, action: int) -> tuple[int, bool, bool]:
        if action not in self.valid_actions(state):
            return state.current_alert, state.cancel_issued, True
        if action == HOLD:
            return state.current_alert, state.cancel_issued, False
        if action == ESCALATE:
            return min(state.current_alert + 1, len(ALERT_LEVELS) - 1), False, False
        if action == DEESCALATE:
            return max(state.current_alert - 1, 0), False, False
        if action == ISSUE_WATCH:
            return max(state.current_alert, 2), False, False
        if action == ISSUE_WARNING:
            return len(ALERT_LEVELS) - 1, False, False
        if action == CANCEL:
            return 0, True, False
        return state.current_alert, state.cancel_issued, True

    def transition(self, state: ToyState, action: int) -> ToyState | None:
        if state.time_index >= self.horizon - 1:
            return None
        next_alert, cancel_issued, _ = self.apply_action(state, action)
        return ToyState(state.time_index + 1, state.danger_tier, next_alert, cancel_issued)

    def reward(self, state: ToyState, action: int) -> float:
        target_alert = self.target_alert_from_danger(state.danger_tier)
        next_alert, cancel_issued, invalid_action = self.apply_action(state, action)
        reward = -0.5

        if invalid_action:
            return reward - 12.0

        expected_alert = (
            1 if target_alert == 0 and state.time_index == 0 else
            1 if target_alert >= 2 and state.time_index == 0 else
            2 if target_alert == 4 and state.time_index < 3 else
            target_alert
        )
        reward -= 1.5 * abs(next_alert - expected_alert)

        if next_alert == expected_alert:
            reward += 3.0
        if cancel_issued and expected_alert >= 2:
            reward -= 30.0

        if state.time_index == self.horizon - 1:
            reward += self.terminal_reward(state.danger_tier, next_alert, cancel_issued)
        return reward

    @staticmethod
    def terminal_reward(danger_tier: int, next_alert: int, cancel_issued: bool) -> float:
        if danger_tier == 2:
            return 100.0 if next_alert == 4 else -300.0
        if danger_tier == 1:
            return 50.0 if next_alert >= 2 else -80.0
        if next_alert >= 4:
            return -80.0
        if cancel_issued or next_alert <= 1:
            return 25.0
        return -20.0


def value_iteration(mdp: ToyTsunamiMDP, tolerance: float = 1e-8) -> tuple[dict, dict]:
    values = {state: 0.0 for state in mdp.states}
    policy = {state: HOLD for state in mdp.states}

    while True:
        delta = 0.0
        for state in mdp.states:
            action_values = []
            for action in mdp.valid_actions(state):
                reward = mdp.reward(state, action)
                next_state = mdp.transition(state, action)
                future_value = 0.0 if next_state is None else mdp.gamma * values[next_state]
                action_values.append((action, reward + future_value))

            best_action, best_value = max(action_values, key=lambda item: item[1])
            delta = max(delta, abs(best_value - values[state]))
            values[state] = best_value
            policy[state] = best_action

        if delta < tolerance:
            break

    return values, policy


def policy_table(policy: dict) -> pd.DataFrame:
    rows = []
    for state, action in sorted(
        policy.items(),
        key=lambda item: (
            item[0].danger_tier,
            item[0].time_index,
            item[0].current_alert,
            item[0].cancel_issued,
        ),
    ):
        rows.append(
            {
                "time_index": state.time_index,
                "danger_tier": state.danger_tier,
                "current_alert": ALERT_LEVELS[state.current_alert],
                "cancel_issued": state.cancel_issued,
                "recommended_action": ACTION_NAMES[action],
            }
        )
    return pd.DataFrame(rows)
