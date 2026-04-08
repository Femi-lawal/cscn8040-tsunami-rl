from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

try:
    from option_a_tsunami_rl.src.agents import evaluate_policy_on_catalog
    from option_a_tsunami_rl.src.data_pipeline import ProjectPaths, ensure_dirs, load_or_scrape, setup_logger
    from option_a_tsunami_rl.src.experiment import _with_time_splits
except ModuleNotFoundError:
    from src.agents import evaluate_policy_on_catalog
    from src.data_pipeline import ProjectPaths, ensure_dirs, load_or_scrape, setup_logger
    from src.experiment import _with_time_splits


class DataPipelineTests(unittest.TestCase):
    def test_load_or_scrape_rebuilds_summary_from_cached_intermediates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            paths = ProjectPaths(root)
            ensure_dirs(paths)
            logger = setup_logger(paths)

            bulletins = pd.DataFrame(
                [
                    {
                        "origin_date": "2025-01-01",
                        "origin_clock_utc": "00:00:00",
                        "origin_time_utc": "2025-01-01T00:00:00+00:00",
                        "magnitude": 7.8,
                        "depth_km": 15.0,
                        "latitude_raw": "2.0N",
                        "longitude_raw": "93.0E",
                        "latitude": 2.0,
                        "longitude": 93.0,
                        "location_name": "Off West Coast of Northern Sumatra",
                        "event_type": "REAL EVENT",
                        "bulletin_number_raw": "1",
                        "bulletin_number": 1,
                        "bulletin_revision": None,
                        "bulletin_type": "EARTHQUAKE BULLETIN",
                        "event_group_id": "g1",
                        "timeline_group_id": "g1",
                        "detail_event_id": "20250101000500",
                        "issue_time_utc": "2025-01-01T00:05:00+00:00",
                        "detail_url": "https://example.com",
                    }
                ]
            )
            details = pd.DataFrame(
                [
                    {
                        "detail_event_id": "20250101000500",
                        "detail_text": "example",
                        "bulletin_header": "header",
                        "evaluation_text": "2. EVALUATION example",
                        "no_threat_flag": False,
                        "potential_threat_flag": True,
                        "confirmed_threat_flag": False,
                        "monitoring_flag": True,
                        "sea_level_confirmed_flag": False,
                        "observed_max_wave_m": None,
                        "threat_class": "potential",
                    }
                ]
            )

            bulletins.to_csv(paths.data_processed / "bmkg_public_bulletins.csv", index=False)
            details.to_csv(paths.data_processed / "bmkg_detail_pages.csv", index=False)

            loaded_bulletins, loaded_details, events = load_or_scrape(paths, logger)

            self.assertEqual(len(loaded_bulletins), 1)
            self.assertEqual(len(loaded_details), 1)
            self.assertEqual(len(events), 1)
            self.assertEqual(int(events.iloc[0]["target_alert_level"]), 2)

            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)

    def test_time_splits_keep_each_danger_class_in_validation_and_test(self) -> None:
        rows: list[dict] = []
        for label, tier, count in [
            ("no_threat", 0, 9),
            ("potential_threat", 1, 9),
            ("confirmed_threat", 2, 9),
        ]:
            for index in range(count):
                rows.append(
                    {
                        "event_group_id": f"{label}_{index}",
                        "origin_time_utc": f"2025-01-{index + 1:02d}T00:00:00+00:00",
                        "danger_label": label,
                        "danger_tier": tier,
                    }
                )

        split_df = _with_time_splits(pd.DataFrame(rows))
        counts = (
            split_df.groupby(["time_split", "danger_label"])
            .size()
            .to_dict()
        )

        self.assertGreater(counts.get(("validation", "no_threat"), 0), 0)
        self.assertGreater(counts.get(("validation", "potential_threat"), 0), 0)
        self.assertGreater(counts.get(("validation", "confirmed_threat"), 0), 0)
        self.assertGreater(counts.get(("test", "no_threat"), 0), 0)
        self.assertGreater(counts.get(("test", "potential_threat"), 0), 0)
        self.assertGreater(counts.get(("test", "confirmed_threat"), 0), 0)

    def test_catalog_evaluation_uses_each_held_out_event_once(self) -> None:
        catalog = pd.DataFrame(
            [
                {
                    "event_group_id": f"event_{index}",
                    "origin_time_utc": f"2025-02-0{index + 1}T00:00:00+00:00",
                    "location_name": "Off West Coast of Northern Sumatra",
                    "latitude": 2.0,
                    "longitude": 93.0,
                    "initial_magnitude": 7.2 + 0.1 * index,
                    "max_magnitude": 7.3 + 0.1 * index,
                    "initial_depth_km": 12.0 + index,
                    "final_depth_km": 18.0 + index,
                    "coastal_proximity_index": 0.8,
                    "first_bulletin_delay_min": 6.0,
                    "final_bulletin_delay_min": 24.0,
                    "bulletin_count": 2,
                    "danger_tier": 1 if index == 1 else 0,
                    "danger_label": "potential_threat" if index == 1 else "no_threat",
                    "target_alert_level": 2 if index == 1 else 0,
                    "confirmed_threat_flag": False,
                    "potential_threat_flag": index == 1,
                    "no_threat_flag": index != 1,
                    "sea_level_confirmed_flag": False,
                    "observed_max_wave_m": 0.12 if index == 1 else 0.01,
                    "wave_height_source": "observed",
                    "has_threat_assessment": index == 1,
                    "training_weight": 1,
                }
                for index in range(3)
            ]
        )

        def hold_policy(_observation, _valid_actions, _env) -> int:
            return 0

        episodes, summary = evaluate_policy_on_catalog(
            catalog,
            hold_policy,
            algorithm_name="hold",
            split_name="test",
            run_seed=1,
            seed_base=123,
        )

        self.assertEqual(len(episodes), 3)
        self.assertEqual(episodes["event_group_id"].nunique(), 3)
        self.assertEqual(int(summary.iloc[0]["episode_count"]), 3)


if __name__ == "__main__":
    unittest.main()
