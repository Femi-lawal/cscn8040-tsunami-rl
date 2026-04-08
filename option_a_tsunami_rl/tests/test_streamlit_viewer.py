from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from streamlit.testing.v1 import AppTest
except ModuleNotFoundError:  # pragma: no cover - installed in test environment
    AppTest = None

try:
    from option_a_tsunami_rl.src.viewer import (
        available_checkpoint_paths,
        build_runtime,
        commit_ppo_decision,
        default_checkpoint_path,
        hidden_trace_frame,
        preview_ppo_decision,
        rule_decision,
        scenario_metadata,
    )
    from option_a_tsunami_rl.src.gym_env import TsunamiGymEnv
except ModuleNotFoundError:
    from src.viewer import (
        available_checkpoint_paths,
        build_runtime,
        commit_ppo_decision,
        default_checkpoint_path,
        hidden_trace_frame,
        preview_ppo_decision,
        rule_decision,
        scenario_metadata,
    )
    from src.gym_env import TsunamiGymEnv


def _catalog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_group_id": "evt_0",
                "origin_time_utc": "2025-01-01T00:00:00+00:00",
                "location_name": "Near Coast",
                "latitude": 2.0,
                "longitude": 93.0,
                "initial_magnitude": 7.0,
                "max_magnitude": 7.0,
                "initial_depth_km": 40.0,
                "final_depth_km": 45.0,
                "coastal_proximity_index": 0.45,
                "first_bulletin_delay_min": 8.0,
                "final_bulletin_delay_min": 24.0,
                "bulletin_count": 1,
                "danger_tier": 0,
                "danger_label": "no_threat",
                "target_alert_level": 0,
                "confirmed_threat_flag": False,
                "potential_threat_flag": False,
                "no_threat_flag": True,
                "sea_level_confirmed_flag": False,
                "observed_max_wave_m": None,
                "wave_height_source": "missing",
                "has_threat_assessment": False,
                "training_weight": 1.0,
            },
            {
                "event_group_id": "evt_1",
                "origin_time_utc": "2025-01-02T00:00:00+00:00",
                "location_name": "Offshore",
                "latitude": 1.2,
                "longitude": 94.2,
                "initial_magnitude": 7.3,
                "max_magnitude": 7.4,
                "initial_depth_km": 28.0,
                "final_depth_km": 30.0,
                "coastal_proximity_index": 0.50,
                "first_bulletin_delay_min": 6.0,
                "final_bulletin_delay_min": 22.0,
                "bulletin_count": 2,
                "danger_tier": 1,
                "danger_label": "potential_threat",
                "target_alert_level": 2,
                "confirmed_threat_flag": False,
                "potential_threat_flag": True,
                "no_threat_flag": False,
                "sea_level_confirmed_flag": False,
                "observed_max_wave_m": None,
                "wave_height_source": "missing",
                "has_threat_assessment": True,
                "training_weight": 2.0,
                "external_wave_proxy_m": 0.14,
            },
            {
                "event_group_id": "evt_2",
                "origin_time_utc": "2025-01-03T00:00:00+00:00",
                "location_name": "Subduction Zone",
                "latitude": 0.5,
                "longitude": 95.0,
                "initial_magnitude": 7.9,
                "max_magnitude": 8.0,
                "initial_depth_km": 16.0,
                "final_depth_km": 18.0,
                "coastal_proximity_index": 0.65,
                "first_bulletin_delay_min": 4.0,
                "final_bulletin_delay_min": 18.0,
                "bulletin_count": 4,
                "danger_tier": 2,
                "danger_label": "confirmed_threat",
                "target_alert_level": 4,
                "confirmed_threat_flag": True,
                "potential_threat_flag": False,
                "no_threat_flag": False,
                "sea_level_confirmed_flag": True,
                "observed_max_wave_m": 0.35,
                "wave_height_source": "observed",
                "has_threat_assessment": True,
                "training_weight": 4.0,
            },
        ]
    )


class StreamlitViewerTests(unittest.TestCase):
    def test_checkpoint_listing_and_default_checkpoint_exist(self) -> None:
        checkpoints = available_checkpoint_paths()
        self.assertTrue(checkpoints)
        self.assertTrue(default_checkpoint_path().exists())
        self.assertLessEqual(len(checkpoints), 3)
        self.assertIn(next(iter(checkpoints.keys())), {
            "Recommended PPO Policy",
            "Stable PPO Policy",
            "Baseline PPO Policy",
            "PPO Checkpoint 1",
        })

    def test_ppo_runtime_and_rule_policy_choose_valid_actions(self) -> None:
        env = TsunamiGymEnv(_catalog(), seed=9, weight_column="training_weight")
        observation, info = env.reset(seed=19)
        runtime = build_runtime(default_checkpoint_path(), device="cpu")
        preview = preview_ppo_decision(runtime, observation, np.asarray(info["action_mask"], dtype=np.float32))
        self.assertIn(preview.action, info["valid_actions"])
        self.assertAlmostEqual(float(preview.probabilities.sum()), 1.0, places=4)

        rule = rule_decision(observation, info["valid_actions"], env)
        self.assertIn(rule.action, info["valid_actions"])

        committed = commit_ppo_decision(runtime, observation, np.asarray(info["action_mask"], dtype=np.float32))
        self.assertEqual(committed.action_name, preview.action_name)

    def test_hidden_trace_and_scenario_metadata_are_available(self) -> None:
        env = TsunamiGymEnv(_catalog(), seed=7, weight_column="training_weight")
        env.reset(seed=7)
        trace = hidden_trace_frame(env)
        metadata = scenario_metadata(env)
        self.assertEqual(len(trace), env.env.horizon)
        self.assertIn("wave_estimate_m", trace.columns)
        self.assertIn(metadata["danger_tier"], {0, 1, 2})
        self.assertIn("location_name", metadata)

    @unittest.skipIf(AppTest is None, "streamlit testing module is unavailable")
    def test_streamlit_app_smoke(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
        at = AppTest.from_file(str(app_path))
        at.run(timeout=60)
        self.assertEqual(len(at.exception), 0)
        self.assertGreaterEqual(len(at.title), 1)
        self.assertEqual(at.title[0].value, "Tsunami Warning Agent Viewer")


if __name__ == "__main__":
    unittest.main()
