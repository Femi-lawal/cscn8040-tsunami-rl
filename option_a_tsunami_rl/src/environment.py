from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ALERT_LEVELS = ["monitor", "info", "watch", "advisory", "warning"]
ACTION_NAMES = ["hold", "escalate", "deescalate", "issue_watch", "issue_warning", "cancel"]

HOLD = 0
ESCALATE = 1
DEESCALATE = 2
ISSUE_WATCH = 3
ISSUE_WARNING = 4
CANCEL = 5


@dataclass
class Scenario:
    event_group_id: str
    danger_tier: int
    target_alert_level: int
    initial_magnitude: float
    max_magnitude: float
    initial_depth_km: float
    final_depth_km: float
    coastal_proximity_index: float
    observed_max_wave_m: float
    first_bulletin_delay_min: float
    final_bulletin_delay_min: float
    bulletin_count: int
    sea_level_confirmed_flag: bool
    has_threat_assessment: bool
    wave_imputed_flag: bool


def _safe_float(value: float | int | None, default: float) -> float:
    if value is None or pd.isna(value):
        return float(default)
    return float(value)


def _safe_bool(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    return bool(value)


def scenario_from_row(row: pd.Series) -> Scenario:
    danger_tier = int(_safe_float(row["danger_tier"], 0))
    wave_imputed_flag = bool(pd.isna(row["observed_max_wave_m"]))
    observed_wave = row["observed_max_wave_m"]
    external_wave_proxy = row.get("external_wave_proxy_m")
    if wave_imputed_flag and pd.notna(external_wave_proxy):
        external_wave = float(external_wave_proxy)
        if danger_tier <= 0:
            observed_wave = min(external_wave, 0.05)
        elif danger_tier == 1:
            observed_wave = min(external_wave, 0.25)
            wave_imputed_flag = False
        else:
            observed_wave = min(external_wave, 1.20)
            wave_imputed_flag = False
    elif wave_imputed_flag:
        magnitude = _safe_float(row["max_magnitude"], 7.0)
        coastal = _safe_float(row["coastal_proximity_index"], 0.5)
        target = int(_safe_float(row["target_alert_level"], 0))
        if danger_tier == 0:
            observed_wave = min(
                0.06,
                max(0.0, (magnitude - 7.1) * 0.02 + coastal * 0.03),
            )
        elif danger_tier == 1:
            observed_wave = min(
                0.25,
                max(0.05, (magnitude - 6.9) * 0.05 + coastal * 0.06 + 0.04 + target * 0.01),
            )
        else:
            observed_wave = min(
                1.20,
                max(0.18, (magnitude - 6.8) * 0.08 + coastal * 0.10 + 0.10 + target * 0.02),
            )

    return Scenario(
        event_group_id=str(row["event_group_id"]),
        danger_tier=danger_tier,
        target_alert_level=int(_safe_float(row["target_alert_level"], 0)),
        initial_magnitude=_safe_float(row["initial_magnitude"], 7.0),
        max_magnitude=_safe_float(row["max_magnitude"], row["initial_magnitude"]),
        initial_depth_km=_safe_float(row["initial_depth_km"], 20.0),
        final_depth_km=_safe_float(row["final_depth_km"], row["initial_depth_km"]),
        coastal_proximity_index=_safe_float(row["coastal_proximity_index"], 0.5),
        observed_max_wave_m=float(observed_wave),
        first_bulletin_delay_min=_safe_float(row["first_bulletin_delay_min"], 8.0),
        final_bulletin_delay_min=_safe_float(row["final_bulletin_delay_min"], 30.0),
        bulletin_count=int(_safe_float(row["bulletin_count"], 1)),
        sea_level_confirmed_flag=_safe_bool(row["sea_level_confirmed_flag"]),
        has_threat_assessment=_safe_bool(row["has_threat_assessment"]),
        wave_imputed_flag=wave_imputed_flag,
    )


class TsunamiWarningEnv:
    """Data-informed tsunami warning environment with persistent episode evidence."""

    def __init__(
        self,
        event_catalog: pd.DataFrame,
        seed: int = 42,
        weight_column: str | None = None,
    ):
        self.catalog = event_catalog.reset_index(drop=True).copy()
        self.rng = np.random.default_rng(seed)
        self.weight_column = weight_column
        self.time_grid_min = np.array([0.0, 2.0, 5.0, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0], dtype=float)
        self.horizon = len(self.time_grid_min)
        self.min_alert_hold_steps = 2
        self.min_warning_cancel_steps = 3

        self.current_scenario: Scenario | None = None
        self.current_step = 0
        self.current_alert = 0
        self.max_alert = 0
        self.alert_changes = 0
        self.warning_step: int | None = None
        self.watch_step: int | None = None
        self.cancel_step: int | None = None
        self.cancel_issued = False
        self.ever_alerted = False
        self.last_alert_change_step = -999
        self.hidden_trace: dict[str, np.ndarray | int | None] = {}
        self.current_observation: np.ndarray | None = None

    @property
    def observation_size(self) -> int:
        return 15

    @property
    def action_size(self) -> int:
        return len(ACTION_NAMES)

    def _sample_catalog_row(self) -> pd.Series:
        weights = None
        if self.weight_column and self.weight_column in self.catalog.columns:
            weights = self.catalog[self.weight_column].to_numpy(dtype=float)
            weights = weights / weights.sum()
        index = int(self.rng.choice(len(self.catalog), p=weights))
        return self.catalog.iloc[index]

    def _delay_to_step(self, delay_minutes: float) -> int:
        index = int(np.searchsorted(self.time_grid_min, max(0.0, delay_minutes), side="left"))
        return min(max(index, 0), self.horizon - 1)

    def _build_hidden_trace(self, scenario: Scenario) -> dict[str, np.ndarray | int | None]:
        archive_first_delay = max(0.0, scenario.first_bulletin_delay_min)
        archive_final_delay = max(archive_first_delay, scenario.final_bulletin_delay_min)
        archive_first_factor = float(
            np.clip(np.log1p(min(archive_first_delay, 360.0)) / np.log1p(360.0), 0.0, 1.0)
        )
        archive_final_factor = float(
            np.clip(np.log1p(min(archive_final_delay, 720.0)) / np.log1p(720.0), 0.0, 1.0)
        )
        wave_strength = float(
            np.clip(
                scenario.observed_max_wave_m
                / (0.10 if scenario.danger_tier == 0 else 0.18 if scenario.danger_tier == 1 else 0.35),
                0.0,
                1.5,
            )
        )
        bulletin_factor = float(np.clip((scenario.bulletin_count - 1) / 3.0, 0.0, 1.5))
        assessment_factor = 1.0 if scenario.has_threat_assessment else 0.0
        sea_level_factor = 1.0 if scenario.sea_level_confirmed_flag else 0.0
        coastal_factor = float(
            np.clip((scenario.coastal_proximity_index - 0.35) / 0.40, 0.0, 1.0)
        )

        if scenario.danger_tier == 0:
            first_signal_min = 8.0 + 14.0 * (1.0 - min(wave_strength, 1.0)) + 6.0 * archive_first_factor
            review_span_min = 5.0 + 4.0 * archive_final_factor
        elif scenario.danger_tier == 1:
            first_signal_min = (
                5.0
                + 10.0 * (1.0 - min(wave_strength, 1.0))
                + 4.0 * archive_first_factor
                - 2.5 * assessment_factor
                - 1.5 * bulletin_factor
            )
            review_span_min = 7.0 + 5.0 * archive_final_factor + 3.0 * bulletin_factor
        else:
            first_signal_min = (
                3.0
                + 8.0 * (1.0 - min(wave_strength, 1.0))
                + 4.0 * archive_first_factor
                - 3.5 * sea_level_factor
                - 2.5 * assessment_factor
                - 2.0 * bulletin_factor
                - 1.5 * coastal_factor
            )
            review_span_min = 6.0 + 4.0 * archive_final_factor + 2.0 * bulletin_factor

        first_signal_min = float(np.clip(first_signal_min, 1.0, 28.0))
        final_signal_min = float(
            np.clip(max(first_signal_min + 3.0, first_signal_min + review_span_min), 6.0, 55.0)
        )
        first_signal_step = self._delay_to_step(first_signal_min)
        final_signal_step = self._delay_to_step(final_signal_min)

        progress = self.time_grid_min / max(float(self.time_grid_min[-1]), 1.0)
        magnitude_noise = self.rng.normal(0.0, 0.04, self.horizon)
        depth_noise = self.rng.normal(0.0, 1.8, self.horizon)
        wave_noise = self.rng.normal(0.0, 0.015, self.horizon)
        uncertainty_noise = self.rng.normal(0.0, 0.015, self.horizon)

        rise_curve = (1.0 - np.exp(-3.5 * progress)) / (1.0 - np.exp(-3.5))
        magnitude_estimates = (
            scenario.initial_magnitude
            + (scenario.max_magnitude - scenario.initial_magnitude) * rise_curve
            + magnitude_noise
        )
        depth_revision_curve = (1.0 - np.exp(-2.5 * progress)) / (1.0 - np.exp(-2.5))
        depth_estimates = (
            scenario.initial_depth_km
            + (scenario.final_depth_km - scenario.initial_depth_km) * depth_revision_curve
            + depth_noise
        )
        depth_estimates = np.maximum(depth_estimates, 1.0)

        if scenario.danger_tier == 0:
            wave_curve = 0.12 + 0.88 * np.exp(-((progress - 0.32) / 0.24) ** 2)
            base_wave = max(0.01, scenario.observed_max_wave_m)
            buoy_on_step = None
            tide_on_step = None
        elif scenario.danger_tier == 1:
            rise = 1.0 / (1.0 + np.exp(-10.0 * (progress - 0.34)))
            decay = 1.0 - 0.30 * np.clip((progress - 0.78) / 0.22, 0.0, 1.0)
            wave_curve = np.clip(0.08 + 0.92 * rise * decay, 0.0, 1.0)
            base_wave = max(0.08, scenario.observed_max_wave_m)
            buoy_on_step = min(self.horizon - 1, max(first_signal_step, 2))
            if scenario.sea_level_confirmed_flag and base_wave >= 0.12:
                tide_on_step = min(self.horizon - 1, max(final_signal_step, buoy_on_step + 1))
            else:
                tide_on_step = None
        else:
            rise = 1.0 / (1.0 + np.exp(-12.0 * (progress - 0.28)))
            decay = 1.0 - 0.15 * np.clip((progress - 0.86) / 0.14, 0.0, 1.0)
            wave_curve = np.clip(0.10 + 0.90 * rise * decay, 0.0, 1.0)
            base_wave = max(0.18, scenario.observed_max_wave_m)
            buoy_on_step = min(self.horizon - 1, max(first_signal_step, 1))
            tide_on_step = min(self.horizon - 1, max(final_signal_step - 1, buoy_on_step + 1))

        onset_scale = np.ones(self.horizon, dtype=float)
        if first_signal_step > 0:
            onset_scale[:first_signal_step] = np.linspace(0.18, 0.60, first_signal_step, endpoint=False)
        wave_estimates = np.maximum(0.0, base_wave * wave_curve * onset_scale + wave_noise)

        buoy_confirmations = np.zeros(self.horizon, dtype=float)
        tide_confirmations = np.zeros(self.horizon, dtype=float)
        if buoy_on_step is not None:
            buoy_confirmations[buoy_on_step:] = 1.0
        if tide_on_step is not None:
            tide_confirmations[tide_on_step:] = 1.0

        uncertainty = np.clip(0.92 - 0.06 * np.arange(self.horizon) + uncertainty_noise, 0.05, 1.0)
        uncertainty -= 0.08 * buoy_confirmations
        uncertainty -= 0.12 * tide_confirmations
        uncertainty = np.clip(uncertainty, 0.05, 1.0)

        return {
            "magnitude_estimates": magnitude_estimates.astype(np.float32),
            "depth_estimates": depth_estimates.astype(np.float32),
            "wave_estimates": wave_estimates.astype(np.float32),
            "buoy_confirmations": buoy_confirmations.astype(np.float32),
            "tide_confirmations": tide_confirmations.astype(np.float32),
            "uncertainty": uncertainty.astype(np.float32),
            "time_fraction": progress.astype(np.float32),
            "first_buoy_step": buoy_on_step,
            "first_tide_step": tide_on_step,
        }

    def _compose_observation(self, step: int) -> np.ndarray:
        assert self.hidden_trace
        magnitude_estimates = self.hidden_trace["magnitude_estimates"]
        depth_estimates = self.hidden_trace["depth_estimates"]
        wave_estimates = self.hidden_trace["wave_estimates"]
        buoy_confirmations = self.hidden_trace["buoy_confirmations"]
        tide_confirmations = self.hidden_trace["tide_confirmations"]
        uncertainty = self.hidden_trace["uncertainty"]
        time_fraction = self.hidden_trace["time_fraction"]

        magnitude_est = float(magnitude_estimates[step])
        depth_est = float(depth_estimates[step])
        wave_est = float(wave_estimates[step])
        buoy = float(buoy_confirmations[step])
        tide = float(tide_confirmations[step])
        uncertainty_value = float(uncertainty[step])
        delta_magnitude = 0.0 if step == 0 else magnitude_est - float(magnitude_estimates[step - 1])
        delta_wave = 0.0 if step == 0 else wave_est - float(wave_estimates[step - 1])
        delta_uncertainty = 0.0 if step == 0 else uncertainty_value - float(uncertainty[step - 1])

        first_buoy_step = self.hidden_trace.get("first_buoy_step")
        first_tide_step = self.hidden_trace.get("first_tide_step")
        buoy_time_norm = -1.0
        tide_time_norm = -1.0
        if first_buoy_step is not None and step >= int(first_buoy_step):
            buoy_time_norm = float(
                np.clip(
                    (self.time_grid_min[step] - self.time_grid_min[int(first_buoy_step)])
                    / self.time_grid_min[-1],
                    0.0,
                    1.0,
                )
            )
        if first_tide_step is not None and step >= int(first_tide_step):
            tide_time_norm = float(
                np.clip(
                    (self.time_grid_min[step] - self.time_grid_min[int(first_tide_step)])
                    / self.time_grid_min[-1],
                    0.0,
                    1.0,
                )
            )

        return np.array(
            [
                magnitude_est,
                depth_est,
                self.current_scenario.coastal_proximity_index if self.current_scenario is not None else 0.5,
                wave_est,
                buoy,
                tide,
                uncertainty_value,
                float(time_fraction[step]),
                self.current_alert / (len(ALERT_LEVELS) - 1),
                float(self.cancel_issued),
                delta_magnitude,
                delta_wave,
                delta_uncertainty,
                buoy_time_norm,
                tide_time_norm,
            ],
            dtype=np.float32,
        )

    def reset(self) -> np.ndarray:
        self.current_scenario = scenario_from_row(self._sample_catalog_row())
        self.current_step = 0
        self.current_alert = 0
        self.max_alert = 0
        self.alert_changes = 0
        self.warning_step = None
        self.watch_step = None
        self.cancel_step = None
        self.cancel_issued = False
        self.ever_alerted = False
        self.last_alert_change_step = -999
        self.hidden_trace = self._build_hidden_trace(self.current_scenario)
        self.current_observation = self._compose_observation(self.current_step)
        return self.current_observation.copy()

    def valid_actions(self) -> list[int]:
        assert self.current_observation is not None
        valid = [HOLD]
        magnitude = float(self.current_observation[0])
        wave = float(self.current_observation[3])
        buoy = int(round(float(self.current_observation[4])))
        tide = int(round(float(self.current_observation[5])))
        uncertainty = float(self.current_observation[6])
        can_deescalate = (
            self.current_step - self.last_alert_change_step >= self.min_alert_hold_steps
        )
        strong_watch_evidence = magnitude >= 7.2 or wave >= 0.06 or buoy == 1 or tide == 1
        strong_warning_evidence = tide == 1 or wave >= 0.18 or (buoy == 1 and wave >= 0.08)
        collapsed_evidence = wave < 0.05 and buoy == 0 and tide == 0 and uncertainty < 0.30

        if self.current_alert == 0:
            valid.append(ESCALATE)
            if strong_watch_evidence:
                valid.append(ISSUE_WATCH)
        elif self.current_alert == 1:
            valid.extend([ESCALATE, DEESCALATE])
            if strong_watch_evidence:
                valid.append(ISSUE_WATCH)
        elif self.current_alert == 2:
            if can_deescalate:
                valid.append(DEESCALATE)
            valid.append(ESCALATE)
            if strong_warning_evidence and self.current_step >= 2:
                valid.append(ISSUE_WARNING)
        elif self.current_alert == 3:
            if can_deescalate:
                valid.append(DEESCALATE)
            valid.append(ESCALATE)
            if strong_warning_evidence:
                valid.append(ISSUE_WARNING)
        elif self.current_alert == 4:
            if can_deescalate:
                valid.append(DEESCALATE)
            if (
                self.warning_step is not None
                and self.current_step - self.warning_step >= self.min_warning_cancel_steps
                and collapsed_evidence
            ):
                valid.append(CANCEL)
        return sorted(set(valid))

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_size, dtype=np.float32)
        for action in self.valid_actions():
            mask[action] = 1.0
        return mask

    def _apply_action(self, action: int) -> tuple[int, bool, bool]:
        invalid_action = action not in self.valid_actions()
        next_alert = self.current_alert
        cancel_issued = self.cancel_issued

        if invalid_action:
            return next_alert, cancel_issued, True

        if action == HOLD:
            return next_alert, cancel_issued, False
        if action == ESCALATE:
            return min(self.current_alert + 1, len(ALERT_LEVELS) - 1), False, False
        if action == DEESCALATE:
            return max(self.current_alert - 1, 0), False, False
        if action == ISSUE_WATCH:
            return max(self.current_alert, 2), False, False
        if action == ISSUE_WARNING:
            return len(ALERT_LEVELS) - 1, False, False
        if action == CANCEL:
            return 0, True, False
        return next_alert, cancel_issued, True

    def _step_reward(
        self,
        action: int,
        observation: np.ndarray,
        previous_alert: int,
        resulting_alert: int,
        invalid_action: bool,
        cancel_issued_after_action: bool,
    ) -> float:
        assert self.current_scenario is not None
        danger = self.current_scenario.danger_tier
        reward = -0.1

        if invalid_action:
            return reward - 12.0

        if resulting_alert != previous_alert:
            reward -= 0.5

        buoy = int(round(float(observation[4])))
        tide = int(round(float(observation[5])))
        if buoy == 1 and resulting_alert < 2:
            reward -= 3.0
        if tide == 1 and resulting_alert < 4:
            reward -= 8.0

        if danger == 0 and resulting_alert >= 2:
            reward -= {2: 3.0, 3: 6.0, 4: 20.0}.get(resulting_alert, 0.0)
        if cancel_issued_after_action and danger >= 1:
            reward -= 6.0
        return reward

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        assert self.current_scenario is not None
        assert self.current_observation is not None

        observation_used = self.current_observation.copy()
        previous_alert = self.current_alert
        resulting_alert, cancel_issued_after_action, invalid_action = self._apply_action(action)
        reward = self._step_reward(
            action,
            observation_used,
            previous_alert,
            resulting_alert,
            invalid_action,
            cancel_issued_after_action,
        )

        if not invalid_action:
            if resulting_alert != previous_alert:
                self.alert_changes += 1
                self.last_alert_change_step = self.current_step
            self.current_alert = resulting_alert
            self.cancel_issued = cancel_issued_after_action
            self.max_alert = max(self.max_alert, self.current_alert)
            self.ever_alerted = self.ever_alerted or self.current_alert > 0
            if self.current_alert >= 2 and self.watch_step is None:
                self.watch_step = self.current_step
            if self.current_alert == 4 and self.warning_step is None:
                self.warning_step = self.current_step
            if action == CANCEL and self.cancel_step is None:
                self.cancel_step = self.current_step

        self.current_step += 1
        done = self.current_step >= self.horizon

        info = {
            "danger_tier": self.current_scenario.danger_tier,
            "event_group_id": self.current_scenario.event_group_id,
            "alert_changes": self.alert_changes,
            "warning_step": self.warning_step,
            "watch_step": self.watch_step,
            "cancel_step": self.cancel_step,
            "invalid_action": invalid_action,
            "observation_used": observation_used.copy(),
        }

        if done:
            terminal_reward, terminal_info = self._terminal_outcome()
            reward += terminal_reward
            info.update(terminal_info)
            next_observation = np.zeros(self.observation_size, dtype=np.float32)
            self.current_observation = next_observation.copy()
        else:
            self.current_observation = self._compose_observation(self.current_step)
            next_observation = self.current_observation.copy()

        return next_observation, reward, done, info

    def _terminal_outcome(self) -> tuple[float, dict]:
        assert self.current_scenario is not None
        danger = self.current_scenario.danger_tier
        reward = 0.0
        missed_severe = False
        false_warning = False

        if danger == 2:
            if self.warning_step is None:
                reward -= 250.0
                missed_severe = True
            else:
                reward += max(40.0, 140.0 - 12.0 * self.warning_step)
        elif danger == 1:
            if self.watch_step is None:
                reward -= 60.0
            else:
                reward += max(8.0, 45.0 - 5.0 * self.watch_step)
            if self.max_alert == 4:
                reward -= 15.0
        else:
            if self.max_alert <= 1:
                reward += 15.0
            elif self.max_alert >= 4:
                reward -= 50.0
                false_warning = True
            else:
                reward -= 5.0 * (self.max_alert - 1)

        reward -= max(0, self.alert_changes - 2) * 6.0
        terminal_info = {
            "missed_severe": missed_severe,
            "false_warning": false_warning,
            "max_alert": self.max_alert,
            "cancel_issued": self.cancel_issued,
            "terminal_danger_tier": self.current_scenario.danger_tier,
            "target_alert_level": self.current_scenario.target_alert_level,
            "wave_imputed_flag": self.current_scenario.wave_imputed_flag,
        }
        return reward, terminal_info


def discretize_observation(observation: np.ndarray) -> tuple[int, ...]:
    magnitude_bin = int(np.digitize(observation[0], [7.0, 7.4, 7.8]))
    depth_bin = int(np.digitize(observation[1], [40.0, 120.0]))
    coastal_bin = int(np.digitize(observation[2], [0.45, 0.60]))
    wave_bin = int(np.digitize(observation[3], [0.04, 0.10, 0.18]))
    uncertainty_bin = int(np.digitize(observation[6], [0.30, 0.60]))
    buoy = int(round(float(observation[4])))
    tide = int(round(float(observation[5])))
    time_bin = int(np.digitize(observation[7], [0.20, 0.45, 0.70]))
    current_alert = int(round(float(observation[8]) * (len(ALERT_LEVELS) - 1)))
    cancel_flag = int(round(float(observation[9])))
    delta_mag_bin = int(np.digitize(observation[10], [-0.03, 0.03]))
    delta_wave_bin = int(np.digitize(observation[11], [-0.01, 0.02]))
    delta_uncertainty_bin = int(np.digitize(observation[12], [-0.04, 0.04]))
    buoy_time_bin = int(np.digitize(observation[13], [-0.5, 0.25, 0.60]))
    tide_time_bin = int(np.digitize(observation[14], [-0.5, 0.25, 0.60]))
    return (
        magnitude_bin,
        depth_bin,
        coastal_bin,
        wave_bin,
        uncertainty_bin,
        buoy,
        tide,
        time_bin,
        current_alert,
        cancel_flag,
        delta_mag_bin,
        delta_wave_bin,
        delta_uncertainty_bin,
        buoy_time_bin,
        tide_time_bin,
    )
