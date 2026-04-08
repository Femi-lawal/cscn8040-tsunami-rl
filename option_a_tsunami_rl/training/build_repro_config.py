from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_repro_config() -> dict[str, Any]:
    return {
        "selection_metric_name": "validation_safety_score",
        "selected_run_name": "ppo_lstm_local_repro_v1",
        "notes": [
            "Local reproducibility configuration matching the strongest bundled PPO-LSTM checkpoint.",
            "Uses observation normalization and a light curriculum schedule during warm-start fine-tuning.",
            "Warm start is taken from a local checkpoint bundled in outputs/models rather than any remote object store.",
        ],
        "event_summary_path": "data/processed/bmkg_event_summary_enriched.csv",
        "output_dir": "outputs/local_repro",
        "model_dir": "outputs/models",
        "search_space": {
            "learning-rate": {"min": 1e-4, "max": 5e-4, "scaling": "log"},
            "entropy-coef": {"min": 1e-3, "max": 3e-2, "scaling": "log"},
            "clip-coef": {"min": 0.1, "max": 0.3, "scaling": "linear"},
            "gamma": {"min": 0.985, "max": 0.999, "scaling": "linear"},
            "gae-lambda": {"min": 0.90, "max": 0.99, "scaling": "linear"},
            "hidden-size": {"min": 128, "max": 256, "scaling": "linear"},
            "lstm-size": {"min": 64, "max": 128, "scaling": "linear"},
            "rollout-steps": {"min": 128, "max": 512, "scaling": "linear"},
            "n-envs": {"min": 8, "max": 16, "scaling": "linear"},
            "minibatches": ["4", "8"],
            "update-epochs": ["3", "4", "5"],
        },
        "selected_hyperparameters": {
            "total-steps": 1000000,
            "n-envs": 9,
            "rollout-steps": 349,
            "update-epochs": 4,
            "minibatches": 8,
            "hidden-size": 253,
            "lstm-size": 117,
            "clip-coef": 0.26565443649206644,
            "learning-rate": 0.0001961873863004801,
            "entropy-coef": 0.02398199811509386,
            "gamma": 0.9897047115635784,
            "gae-lambda": 0.9149191327911267,
            "eval-interval": 50000,
            "seeds": "77,88,99",
            "device": "cuda",
            "synthetic-multiplier": 1.0,
            "severe-miss-penalty": 300.0,
            "false-warning-penalty": 50.0,
            "normalize-observations": 1,
            "use-curriculum": 1,
            "curriculum-stage-fractions": "0.05,0.15,0.80",
            "init-checkpoint-path": "outputs/models/ppo_lstm_warmstart_reference.pt",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the local PPO-LSTM reproducibility configuration.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = build_repro_config()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"wrote_repro_config={args.output}")


if __name__ == "__main__":
    main()
