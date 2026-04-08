from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

try:
    from option_a_tsunami_rl.src.environment import CANCEL, TsunamiWarningEnv, discretize_observation, scenario_from_row
except ModuleNotFoundError:
    from src.environment import CANCEL, TsunamiWarningEnv, discretize_observation, scenario_from_row


def _single_event_catalog(danger_tier: int, target_alert_level: int, sea_level_confirmed: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_group_id": f"event_{danger_tier}",
                "origin_time_utc": "2025-01-01T00:00:00+00:00",
                "location_name": "Off West Coast of Northern Sumatra",
                "latitude": 2.0,
                "longitude": 93.0,
                "initial_magnitude": 8.0 if danger_tier == 2 else 7.2,
                "max_magnitude": 8.1 if danger_tier == 2 else 7.3,
                "initial_depth_km": 12.0,
                "final_depth_km": 18.0,
                "coastal_proximity_index": 0.9,
                "first_bulletin_delay_min": 6.0,
                "final_bulletin_delay_min": 24.0,
                "bulletin_count": 4 if danger_tier == 2 else 2,
                "danger_tier": danger_tier,
                "danger_label": "confirmed_threat" if danger_tier == 2 else "no_threat",
                "target_alert_level": target_alert_level,
                "confirmed_threat_flag": danger_tier == 2,
                "potential_threat_flag": danger_tier == 1,
                "no_threat_flag": danger_tier == 0,
                "sea_level_confirmed_flag": sea_level_confirmed,
                "observed_max_wave_m": 0.22 if danger_tier == 2 else 0.01,
                "wave_height_source": "observed",
                "has_threat_assessment": danger_tier >= 1,
                "training_weight": 1,
            }
        ]
    )


class EnvironmentTests(unittest.TestCase):
    def test_cancel_is_invalid_at_start_and_not_counted_as_warning(self) -> None:
        env = TsunamiWarningEnv(_single_event_catalog(2, 4, True), seed=7)
        observation = env.reset()

        self.assertNotIn(CANCEL, env.valid_actions())
        _, _, _, info = env.step(CANCEL)

        self.assertTrue(info["invalid_action"])
        self.assertIsNone(info["warning_step"])
        self.assertEqual(env.max_alert, 0)
        self.assertEqual(observation.tolist(), info["observation_used"].tolist())

    def test_always_cancel_or_hold_on_severe_event_still_misses_severe(self) -> None:
        env = TsunamiWarningEnv(_single_event_catalog(2, 4, True), seed=11)
        env.reset()
        done = False
        info = {}
        while not done:
            action = CANCEL if CANCEL in env.valid_actions() else 0
            _, _, done, info = env.step(action)

        self.assertTrue(info["missed_severe"])
        self.assertIsNone(info["warning_step"])

    def test_sensor_confirmations_are_monotonic_within_episode(self) -> None:
        env = TsunamiWarningEnv(_single_event_catalog(2, 4, True), seed=13)
        observation = env.reset()
        buoy_values = [int(round(float(observation[4])))]
        tide_values = [int(round(float(observation[5])))]

        done = False
        while not done:
            observation, _, done, _ = env.step(0)
            if not done:
                buoy_values.append(int(round(float(observation[4]))))
                tide_values.append(int(round(float(observation[5]))))

        self.assertEqual(buoy_values, sorted(buoy_values))
        self.assertEqual(tide_values, sorted(tide_values))

    def test_hidden_trace_is_built_once_with_extended_horizon(self) -> None:
        env = TsunamiWarningEnv(_single_event_catalog(1, 2, False), seed=17)
        observation = env.reset()

        self.assertEqual(env.horizon, 12)
        self.assertEqual(len(env.hidden_trace["wave_estimates"]), env.horizon)
        self.assertEqual(len(observation), env.observation_size)

    def test_discretization_changes_with_depth_and_coastal_features(self) -> None:
        base = TsunamiWarningEnv(_single_event_catalog(1, 2, False), seed=19).reset()
        deeper = base.copy()
        deeper[1] = 180.0
        farther = base.copy()
        farther[2] = 0.20

        self.assertNotEqual(discretize_observation(base), discretize_observation(deeper))
        self.assertNotEqual(discretize_observation(base), discretize_observation(farther))

    def test_action_mask_matches_valid_actions(self) -> None:
        env = TsunamiWarningEnv(_single_event_catalog(2, 4, True), seed=23)
        env.reset()
        mask = env.action_mask()
        valid_actions = env.valid_actions()

        self.assertEqual(mask.shape, (env.action_size,))
        self.assertEqual(sorted(np.flatnonzero(mask).tolist()), valid_actions)

    def test_external_wave_proxy_is_used_when_bmkg_wave_missing(self) -> None:
        row = _single_event_catalog(1, 2, False).iloc[0].copy()
        row["observed_max_wave_m"] = None
        row["external_wave_proxy_m"] = 0.34

        scenario = scenario_from_row(row)

        self.assertAlmostEqual(scenario.observed_max_wave_m, 0.25)
        self.assertFalse(scenario.wave_imputed_flag)

    def test_no_threat_external_wave_proxy_is_capped_and_stays_imputed(self) -> None:
        row = _single_event_catalog(0, 0, False).iloc[0].copy()
        row["observed_max_wave_m"] = None
        row["external_wave_proxy_m"] = 1.75

        scenario = scenario_from_row(row)

        self.assertAlmostEqual(scenario.observed_max_wave_m, 0.05)
        self.assertTrue(scenario.wave_imputed_flag)

    def test_no_threat_wave_imputation_stays_small_without_proxy(self) -> None:
        row = _single_event_catalog(0, 0, False).iloc[0].copy()
        row["observed_max_wave_m"] = None
        row["external_wave_proxy_m"] = None

        scenario = scenario_from_row(row)

        self.assertLessEqual(scenario.observed_max_wave_m, 0.06)
        self.assertTrue(scenario.wave_imputed_flag)


if __name__ == "__main__":
    unittest.main()
