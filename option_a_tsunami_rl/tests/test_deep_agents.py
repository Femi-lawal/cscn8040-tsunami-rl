from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch

try:
    from option_a_tsunami_rl.src.deep_agents import (
        PPOConfig,
        RecurrentActorCritic,
        RecurrentRolloutBuffer,
        evaluate_ppo_lstm_on_catalog,
        score_weights_from_config,
        train_ppo_lstm,
    )
    from option_a_tsunami_rl.src.gym_env import TsunamiGymEnv
    from option_a_tsunami_rl.src.metrics import compute_operational_score
    from option_a_tsunami_rl.training.build_repro_config import build_repro_config
    from option_a_tsunami_rl.training.launch_local_run import build_local_run_manifest, load_run_config
    from option_a_tsunami_rl.training.train_ppo_lstm import (
        _checkpoint_model_shape_overrides,
        _resolve_init_checkpoint_path,
    )
except ModuleNotFoundError:
    from src.deep_agents import (
        PPOConfig,
        RecurrentActorCritic,
        RecurrentRolloutBuffer,
        evaluate_ppo_lstm_on_catalog,
        score_weights_from_config,
        train_ppo_lstm,
    )
    from src.gym_env import TsunamiGymEnv
    from src.metrics import compute_operational_score
    from training.build_repro_config import build_repro_config
    from training.launch_local_run import build_local_run_manifest, load_run_config
    from training.train_ppo_lstm import _checkpoint_model_shape_overrides, _resolve_init_checkpoint_path


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


class DeepAgentTests(unittest.TestCase):
    def test_gym_wrapper_random_episodes_have_valid_masks(self) -> None:
        env = TsunamiGymEnv(_catalog(), seed=5, weight_column="training_weight")
        for episode in range(5):
            observation, info = env.reset(seed=100 + episode)
            self.assertEqual(observation.shape, (15,))
            self.assertGreaterEqual(int(np.asarray(info["action_mask"]).sum()), 1)
            done = False
            while not done:
                valid_actions = np.flatnonzero(info["action_mask"]).tolist()
                action = int(valid_actions[0])
                observation, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                if not done:
                    self.assertGreaterEqual(int(np.asarray(info["action_mask"]).sum()), 1)

    def test_masked_policy_assigns_zero_probability_to_invalid_actions(self) -> None:
        model = RecurrentActorCritic(15, 6, hidden_size=64, lstm_size=32)
        device = torch.device("cpu")
        state = model.initial_state(2, device)
        obs = torch.zeros(2, 15, dtype=torch.float32)
        done_mask = torch.ones(2, dtype=torch.float32)
        action_mask = torch.tensor([[1, 0, 0, 1, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.float32)
        _, _, _, _, _, probs = model.act(obs, state, done_mask, action_mask, deterministic=False)
        self.assertTrue(torch.allclose(probs[action_mask == 0], torch.zeros_like(probs[action_mask == 0])))

    def test_forward_smoke_and_buffer_shapes(self) -> None:
        model = RecurrentActorCritic(15, 6, hidden_size=64, lstm_size=32)
        device = torch.device("cpu")
        state = model.initial_state(3, device)
        model.update_observation_stats(np.random.randn(5, 15).astype(np.float32))
        self.assertGreater(float(model.obs_count.item()), 1.0)
        obs = torch.randn(4, 3, 15)
        done_mask = torch.zeros(4, 3)
        action_mask = torch.ones(4, 3, 6)
        logits, values, next_state = model.forward(obs, state, done_mask, action_mask)
        self.assertEqual(logits.shape, (4, 3, 6))
        self.assertEqual(values.shape, (4, 3))
        self.assertEqual(next_state[0].shape, (1, 3, 32))

        buffer = RecurrentRolloutBuffer(rollout_steps=4, n_envs=3, observation_dim=15, action_dim=6, lstm_size=32)
        buffer.set_initial_state(state)
        for step in range(4):
            buffer.add(
                step,
                observations=np.zeros((3, 15), dtype=np.float32),
                action_masks=np.ones((3, 6), dtype=np.float32),
                actions=np.zeros(3, dtype=np.int64),
                rewards=np.ones(3, dtype=np.float32),
                start_dones=np.zeros(3, dtype=np.float32),
                next_dones=np.zeros(3, dtype=np.float32),
                values=np.zeros(3, dtype=np.float32),
                log_probs=np.zeros(3, dtype=np.float32),
            )
        buffer.compute_returns_and_advantages(np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 0.99, 0.95)
        batches = buffer.iter_minibatches(2, torch.device("cpu"), np.random.default_rng(3))
        self.assertTrue(batches)
        self.assertEqual(batches[0]["observations"].shape[-1], 15)
        self.assertEqual(buffer.returns.shape, (4, 3))
        self.assertEqual(buffer.advantages.shape, (4, 3))

    def test_tiny_ppo_run_completes(self) -> None:
        catalog = _catalog()
        config = PPOConfig(
            total_steps=128,
            n_envs=2,
            rollout_steps=16,
            update_epochs=1,
            minibatches=1,
            hidden_size=64,
            lstm_size=32,
            eval_interval=64,
            device="cpu",
            seed=9,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            model, history = train_ppo_lstm(
                catalog,
                config,
                validation_catalog=catalog,
                output_dir=Path(temp_dir),
                weight_column="training_weight",
            )
            self.assertIsInstance(model, RecurrentActorCritic)
            self.assertFalse(history.empty)
            self.assertTrue(np.isfinite(history["policy_loss"]).all())
            self.assertTrue(np.isfinite(history["value_loss"]).all())
            self.assertIn("validation_safety_score", history.columns)
            self.assertIn("curriculum_stage", history.columns)
            self.assertIn("normalize_observations", history.columns)

    def test_operational_score_penalizes_safety_failures(self) -> None:
        safe_score = compute_operational_score(20.0, 0.0, 0.0)
        risky_score = compute_operational_score(35.0, 0.2, 0.2)
        self.assertGreater(safe_score, risky_score)

    def test_score_weights_can_be_built_from_config(self) -> None:
        config = PPOConfig(severe_miss_penalty=400.0, false_warning_penalty=75.0)
        weights = score_weights_from_config(config)
        score = compute_operational_score(10.0, 0.1, 0.1, weights=weights)
        self.assertAlmostEqual(score, 10.0 - 40.0 - 7.5)

    def test_ppo_evaluation_schema_matches_expected_columns(self) -> None:
        catalog = _catalog()
        model = RecurrentActorCritic(15, 6, hidden_size=64, lstm_size=32)
        episodes, summary = evaluate_ppo_lstm_on_catalog(
            catalog,
            model,
            algorithm_name="ppo_lstm_pure",
            split_name="test",
            run_seed=1,
            seed_base=500,
            device="cpu",
        )
        expected_episode_columns = {
            "episode",
            "algorithm",
            "split",
            "run_seed",
            "event_group_id",
            "return",
            "danger_tier",
            "missed_severe",
            "false_warning",
            "warning_step",
            "watch_step",
            "invalid_actions",
        }
        expected_summary_columns = {
            "algorithm",
            "split",
            "run_seed",
            "episode_count",
            "avg_return",
            "severe_miss_rate",
            "false_warning_rate",
            "avg_warning_step_on_severe",
            "avg_watch_step_on_potential",
        }
        self.assertTrue(expected_episode_columns.issubset(set(episodes.columns)))
        self.assertTrue(expected_summary_columns.issubset(set(summary.columns)))

    def test_build_repro_config_uses_validation_safety_score(self) -> None:
        config = build_repro_config()
        self.assertEqual(config["selection_metric_name"], "validation_safety_score")
        self.assertEqual(config["selected_hyperparameters"]["total-steps"], 1000000)
        self.assertEqual(config["selected_hyperparameters"]["use-curriculum"], 1)
        self.assertEqual(config["selected_hyperparameters"]["curriculum-stage-fractions"], "0.05,0.15,0.80")
        self.assertEqual(config["selected_hyperparameters"]["init-checkpoint-path"], "outputs/models/ppo_lstm_warmstart_reference.pt")
        self.assertIsNotNone(re.search(r"validation_safety_score", config["selection_metric_name"]))

    def test_checkpoint_model_shape_overrides_reads_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "ppo_lstm_best.pt"
            torch.save(
                {
                    "state_dict": {},
                    "config": {"hidden_size": 253, "lstm_size": 117},
                },
                checkpoint_path,
            )
            overrides = _checkpoint_model_shape_overrides(checkpoint_path)
            self.assertEqual(overrides["hidden_size"], 253)
            self.assertEqual(overrides["lstm_size"], 117)

    def test_resolve_init_checkpoint_path_handles_local_path_string(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            checkpoint_path = repo_root / "outputs" / "models" / "ppo_lstm_warmstart_reference.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"placeholder")
            resolved = _resolve_init_checkpoint_path(
                "outputs/models/ppo_lstm_warmstart_reference.pt",
                repo_root,
            )
            self.assertEqual(resolved, checkpoint_path.resolve())

    def test_local_manifest_uses_repro_config(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "training" / "local_repro_config.json"
        config = load_run_config(config_path)
        manifest = build_local_run_manifest(
            config,
            repo_root=Path(__file__).resolve().parents[1],
            python_executable="python",
        )
        self.assertEqual(manifest["selection_metric_name"], "validation_safety_score")
        self.assertIn("option_a_tsunami_rl.training.train_ppo_lstm", manifest["command"])
        self.assertIn("--total-steps", manifest["command"])
        self.assertIn("1000000", manifest["command"])


if __name__ == "__main__":
    unittest.main()
