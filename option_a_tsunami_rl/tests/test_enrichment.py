from __future__ import annotations

import unittest

import pandas as pd

try:
    from option_a_tsunami_rl.src.enrichment import generate_synthetic_training_catalog
except ModuleNotFoundError:
    from src.enrichment import generate_synthetic_training_catalog


class EnrichmentTests(unittest.TestCase):
    def test_synthetic_generator_preserves_schema_and_marks_rows(self) -> None:
        train_catalog = pd.DataFrame(
            [
                {
                    "event_group_id": f"real_{index}",
                    "origin_time_utc": f"2024-01-{index + 1:02d}T00:00:00+00:00",
                    "location_name": "Off West Coast of Northern Sumatra",
                    "latitude": 2.0,
                    "longitude": 93.0,
                    "initial_magnitude": 7.0 + 0.1 * index,
                    "max_magnitude": 7.1 + 0.1 * index,
                    "initial_depth_km": 12.0,
                    "final_depth_km": 18.0,
                    "coastal_proximity_index": 0.8,
                    "first_bulletin_delay_min": 5.0,
                    "final_bulletin_delay_min": 20.0,
                    "bulletin_count": 2,
                    "max_bulletin_number": 2,
                    "danger_tier": index % 3,
                    "danger_label": ["no_threat", "potential_threat", "confirmed_threat"][index % 3],
                    "target_alert_level": [0, 2, 4][index % 3],
                    "confirmed_threat_flag": index % 3 == 2,
                    "potential_threat_flag": index % 3 == 1,
                    "no_threat_flag": index % 3 == 0,
                    "sea_level_confirmed_flag": index % 3 == 2,
                    "observed_max_wave_m": 0.01 if index % 3 == 0 else 0.25,
                    "external_wave_proxy_m": 0.02 if index % 3 == 0 else 0.30,
                    "wave_height_source": "observed",
                    "has_threat_assessment": index % 3 > 0,
                    "training_weight": 1 + index,
                    "data_source": "bmkg",
                    "is_synthetic": 0,
                }
                for index in range(6)
            ]
        )

        synthetic = generate_synthetic_training_catalog(train_catalog, synthetic_multiplier=0.5, seed=7)

        self.assertEqual(list(synthetic.columns), list(train_catalog.columns))
        self.assertTrue((synthetic["is_synthetic"] == 1).all())
        self.assertTrue(synthetic["event_group_id"].str.startswith("synthetic_").all())
        self.assertGreater(len(synthetic), 0)


if __name__ == "__main__":
    unittest.main()
