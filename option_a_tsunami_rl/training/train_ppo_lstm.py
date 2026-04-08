from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import pandas as pd
import torch

from option_a_tsunami_rl.src.deep_agents import (
    PPOConfig,
    evaluate_ppo_lstm_on_catalog,
    resolve_device,
    train_ppo_lstm,
)
from option_a_tsunami_rl.src.enrichment import generate_synthetic_training_catalog
from option_a_tsunami_rl.src.evaluation_utils import aggregate_eval, with_time_splits


LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _emit_metric_lines(metrics: dict[str, object], output_dir: Path) -> None:
    emitted_lines: list[str] = []
    for key, value in metrics.items():
        if isinstance(value, list):
            continue
        line = f"{key}={value}"
        emitted_lines.append(line)
        print(line, flush=True)
        LOGGER.info(line)
    (output_dir / "metric_lines.txt").write_text("\n".join(emitted_lines) + "\n", encoding="utf-8")


def _parse_seed_list(raw: str) -> list[int]:
    values = [value.strip() for value in raw.split(",")]
    seeds = [int(value) for value in values if value]
    return seeds or [77]


def _parse_fraction_list(raw: str) -> tuple[float, float, float]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    parsed = [float(value) for value in values]
    if len(parsed) != 3:
        raise ValueError("Expected exactly three curriculum fractions, for example: 0.2,0.3,0.5")
    return tuple(parsed)  # type: ignore[return-value]


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_event_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Event summary not found: {path}")
    return pd.read_csv(path)


def _copy_best_seed_artifacts(model_dir: Path, best_seed: int) -> None:
    shutil.copy2(model_dir / f"ppo_lstm_seed{best_seed}_best.pt", model_dir / "ppo_lstm_best.pt")
    shutil.copy2(model_dir / f"ppo_lstm_seed{best_seed}_last.pt", model_dir / "ppo_lstm_last.pt")


def _safe_copy_if_different(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        return
    shutil.copy2(src, dst)


def _resolve_init_checkpoint_path(init_checkpoint_path: str | None, repo_root: Path | None = None) -> Path | None:
    if init_checkpoint_path is None:
        return None
    base = repo_root or _default_repo_root()
    candidate = Path(str(init_checkpoint_path))
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _checkpoint_model_shape_overrides(init_checkpoint_path: Path | None) -> dict[str, int]:
    if init_checkpoint_path is None or not init_checkpoint_path.exists():
        return {}
    checkpoint = torch.load(init_checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    overrides: dict[str, int] = {}
    for key in ("hidden_size", "lstm_size"):
        value = config.get(key)
        if value is None:
            continue
        overrides[key] = int(value)
    return overrides


def main() -> None:
    _configure_logging()
    repo_root = _default_repo_root()
    default_event_summary = repo_root / "data" / "processed" / "bmkg_event_summary_enriched.csv"
    default_output_dir = repo_root / "outputs" / "local_repro"
    default_model_dir = repo_root / "outputs" / "models"

    parser = argparse.ArgumentParser(description="Train PPO-LSTM for Option A using local artifacts only.")
    parser.add_argument("--event-summary-path", type=Path, default=default_event_summary)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--model-dir", type=Path, default=default_model_dir)
    parser.add_argument("--seeds", type=str, default="77,88,99")
    parser.add_argument("--synthetic-multiplier", type=float, default=1.0)
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatches", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--lstm-size", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=50_000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--severe-miss-penalty", type=float, default=300.0)
    parser.add_argument("--false-warning-penalty", type=float, default=50.0)
    parser.add_argument("--warning-delay-penalty", type=float, default=0.0)
    parser.add_argument("--potential-delay-penalty", type=float, default=0.0)
    parser.add_argument("--normalize-observations", type=int, default=1)
    parser.add_argument("--use-curriculum", type=int, default=1)
    parser.add_argument("--curriculum-stage-fractions", type=str, default="0.2,0.3,0.5")
    parser.add_argument("--init-checkpoint-path", type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir
    model_dir = args.model_dir
    generated_model_dir = output_dir / "models"
    table_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    generated_model_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    resolved_init_checkpoint_path = _resolve_init_checkpoint_path(args.init_checkpoint_path, repo_root)
    if resolved_init_checkpoint_path is not None and not resolved_init_checkpoint_path.exists():
        raise FileNotFoundError(f"Warm-start checkpoint not found: {resolved_init_checkpoint_path}")

    architecture_overrides = _checkpoint_model_shape_overrides(resolved_init_checkpoint_path)
    hidden_size = architecture_overrides.get("hidden_size", args.hidden_size)
    lstm_size = architecture_overrides.get("lstm_size", args.lstm_size)
    if architecture_overrides:
        LOGGER.info(
            "Using warm-start architecture overrides hidden_size=%s lstm_size=%s from %s",
            hidden_size,
            lstm_size,
            resolved_init_checkpoint_path,
        )

    event_summary = _load_event_summary(args.event_summary_path)
    split_df = with_time_splits(event_summary)
    train_catalog = split_df[split_df["time_split"] == "train"].copy()
    validation_catalog = split_df[split_df["time_split"] == "validation"].copy()
    test_catalog = split_df[split_df["time_split"] == "test"].copy()
    synthetic_train_catalog = generate_synthetic_training_catalog(
        train_catalog,
        synthetic_multiplier=args.synthetic_multiplier,
        seed=20260327,
    )
    augmented_train_catalog = pd.concat([train_catalog, synthetic_train_catalog], ignore_index=True)

    seeds = _parse_seed_list(args.seeds)
    resolved_device = resolve_device(args.device)
    evaluation_by_seed_all: list[pd.DataFrame] = []
    evaluation_episodes_all: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []
    best_seed = None
    best_validation_score = -float("inf")

    for seed in seeds:
        config = PPOConfig(
            total_steps=args.total_steps,
            n_envs=args.n_envs,
            rollout_steps=args.rollout_steps,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_coef=args.clip_coef,
            learning_rate=args.learning_rate,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            max_grad_norm=args.max_grad_norm,
            update_epochs=args.update_epochs,
            minibatches=args.minibatches,
            hidden_size=hidden_size,
            lstm_size=lstm_size,
            seed=seed,
            eval_interval=args.eval_interval,
            device=str(resolved_device),
            severe_miss_penalty=args.severe_miss_penalty,
            false_warning_penalty=args.false_warning_penalty,
            warning_delay_penalty=args.warning_delay_penalty,
            potential_delay_penalty=args.potential_delay_penalty,
            normalize_observations=bool(args.normalize_observations),
            use_curriculum=bool(args.use_curriculum),
            curriculum_stage_fractions=_parse_fraction_list(args.curriculum_stage_fractions),
            init_checkpoint_path=str(resolved_init_checkpoint_path) if resolved_init_checkpoint_path else None,
        )
        artifact_prefix = f"ppo_lstm_seed{seed}"
        model, history_df = train_ppo_lstm(
            augmented_train_catalog,
            config,
            validation_catalog=validation_catalog,
            output_dir=output_dir,
            weight_column="training_weight",
            artifact_prefix=artifact_prefix,
        )
        history_df["run_seed"] = seed
        history_frames.append(history_df)

        validation_episodes, validation_summary = evaluate_ppo_lstm_on_catalog(
            validation_catalog,
            model,
            algorithm_name="ppo_lstm_pure",
            split_name="validation",
            run_seed=seed,
            seed_base=101000 + seed * 100,
            device=resolved_device,
        )
        test_episodes, test_summary = evaluate_ppo_lstm_on_catalog(
            test_catalog,
            model,
            algorithm_name="ppo_lstm_pure",
            split_name="test",
            run_seed=seed,
            seed_base=102000 + seed * 100,
            device=resolved_device,
        )
        evaluation_episodes_all.extend([validation_episodes, test_episodes])
        evaluation_by_seed_all.extend([validation_summary, test_summary])

        validation_score = float(validation_summary.iloc[0]["safety_score"])
        if validation_score > best_validation_score:
            best_validation_score = validation_score
            best_seed = seed

    if best_seed is not None:
        _copy_best_seed_artifacts(generated_model_dir, best_seed)
        _safe_copy_if_different(generated_model_dir / "ppo_lstm_best.pt", model_dir / "ppo_lstm_best.pt")
        _safe_copy_if_different(generated_model_dir / "ppo_lstm_last.pt", model_dir / "ppo_lstm_last.pt")

    history_df = pd.concat(history_frames, ignore_index=True)
    evaluation_by_seed_df = pd.concat(evaluation_by_seed_all, ignore_index=True)
    evaluation_episode_df = pd.concat(evaluation_episodes_all, ignore_index=True)
    evaluation_summary_df = aggregate_eval(evaluation_by_seed_df)

    history_df.to_csv(table_dir / "ppo_lstm_training_history.csv", index=False)
    evaluation_by_seed_df.to_csv(table_dir / "ppo_lstm_evaluation_by_seed.csv", index=False)
    evaluation_episode_df.to_csv(table_dir / "ppo_lstm_evaluation_episode_metrics.csv", index=False)
    evaluation_summary_df.to_csv(table_dir / "ppo_lstm_evaluation_summary.csv", index=False)

    validation_mean = evaluation_by_seed_df[evaluation_by_seed_df["split"] == "validation"].mean(numeric_only=True)
    test_mean = evaluation_by_seed_df[evaluation_by_seed_df["split"] == "test"].mean(numeric_only=True)
    metrics = {
        "validation_avg_return": float(validation_mean["avg_return"]),
        "validation_safety_score": float(validation_mean["safety_score"]),
        "validation_severe_miss_rate": float(validation_mean["severe_miss_rate"]),
        "validation_false_warning_rate": float(validation_mean["false_warning_rate"]),
        "test_avg_return": float(test_mean["avg_return"]),
        "test_safety_score": float(test_mean["safety_score"]),
        "test_severe_miss_rate": float(test_mean["severe_miss_rate"]),
        "test_false_warning_rate": float(test_mean["false_warning_rate"]),
        "resolved_device": str(resolved_device),
        "seeds": seeds,
        "best_seed": best_seed,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _emit_metric_lines(metrics, output_dir)


if __name__ == "__main__":
    main()
