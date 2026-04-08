from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from .agents import (
    TrainConfig,
    evaluate_policy_on_catalog,
    pure_greedy_policy,
    rule_based_policy,
    safe_greedy_policy,
    train_q_learning,
    train_sarsa,
)
from .data_pipeline import ProjectPaths, setup_logger
from .deep_agents import PPOConfig, evaluate_ppo_lstm_on_catalog, resolve_device, train_ppo_lstm
from .dp_baseline import ToyTsunamiMDP, policy_table, value_iteration
from .evaluation_utils import aggregate_eval, with_time_splits
from .environment import TsunamiWarningEnv
from .enrichment import generate_synthetic_training_catalog

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")


def _ensure_output_dirs(root: Path) -> tuple[Path, Path]:
    figures_dir = root / "outputs" / "figures"
    tables_dir = root / "outputs" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    (root / "outputs" / "models").mkdir(parents=True, exist_ok=True)
    (root / "outputs" / "tensorboard").mkdir(parents=True, exist_ok=True)
    return figures_dir, tables_dir


def _parse_int_list_env(name: str, default: list[int]) -> list[int]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    values = [value.strip() for value in raw.split(",")]
    parsed = [int(value) for value in values if value]
    return parsed or default


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


def _save_data_summary(event_summary: pd.DataFrame, figures_dir: Path, tables_dir: Path) -> None:
    counts = (
        event_summary.groupby(["time_split", "danger_label"])
        .size()
        .reset_index(name="count")
    )
    counts.to_csv(tables_dir / "danger_label_counts.csv", index=False)
    split_counts = event_summary["time_split"].value_counts().rename_axis("split").reset_index(name="count")
    split_counts.to_csv(tables_dir / "time_split_counts.csv", index=False)

    plt.figure(figsize=(8, 4.5))
    sns.barplot(
        data=counts,
        x="danger_label",
        y="count",
        hue="time_split",
        palette="crest",
    )
    plt.title("BMKG Event Summary by Danger Label and Time Split")
    plt.xlabel("Danger Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(figures_dir / "danger_label_distribution.png", dpi=160)
    plt.close()


def _save_training_curve(training_history: pd.DataFrame, figures_dir: Path) -> None:
    plot_df = training_history.copy()
    plot_df["rolling_return"] = (
        plot_df.groupby(["algorithm", "run_seed"])["return"]
        .transform(lambda series: series.rolling(100, min_periods=25).mean())
    )

    plt.figure(figsize=(8, 4.5))
    sns.lineplot(
        data=plot_df,
        x="episode",
        y="rolling_return",
        hue="algorithm",
        style="run_seed",
        linewidth=1.6,
    )
    plt.title("Training Return (100-Episode Rolling Mean)")
    plt.xlabel("Episode")
    plt.ylabel("Rolling Return")
    plt.tight_layout()
    plt.savefig(figures_dir / "training_returns.png", dpi=160)
    plt.close()


def _with_time_splits(event_summary: pd.DataFrame) -> pd.DataFrame:
    return with_time_splits(event_summary)


def _aggregate_eval(by_seed: pd.DataFrame) -> pd.DataFrame:
    return aggregate_eval(by_seed)


def _freeze_baseline_snapshot(
    root: Path,
    config_df: pd.DataFrame,
    split_df: pd.DataFrame,
    evaluation_summary_df: pd.DataFrame,
    run_seeds: list[int],
) -> None:
    baseline_dir = root / "baseline_results"
    marker_path = baseline_dir / "baseline_frozen.json"
    if marker_path.exists():
        return

    baseline_dir.mkdir(parents=True, exist_ok=True)
    config_df.to_csv(baseline_dir / "config_snapshot.csv", index=False)
    (baseline_dir / "config_snapshot.json").write_text(
        json.dumps(
            {row["parameter"]: row["value"] for row in config_df.to_dict(orient="records")},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    split_df.to_csv(baseline_dir / "event_summary_with_splits.csv", index=False)
    split_df[split_df["time_split"] == "train"].to_csv(baseline_dir / "train_split.csv", index=False)
    split_df[split_df["time_split"] == "validation"].to_csv(baseline_dir / "validation_split.csv", index=False)
    split_df[split_df["time_split"] == "test"].to_csv(baseline_dir / "test_split.csv", index=False)
    evaluation_summary_df.to_csv(baseline_dir / "metrics_snapshot.csv", index=False)
    (baseline_dir / "seed_list.json").write_text(
        json.dumps({"run_seeds": run_seeds}, indent=2),
        encoding="utf-8",
    )
    test_rows: list[str] = []
    for algorithm in ["q_learning_pure", "sarsa_pure", "q_learning_safe", "sarsa_safe", "rule_based"]:
        subset = evaluation_summary_df[
            (evaluation_summary_df["algorithm"] == algorithm)
            & (evaluation_summary_df["split"] == "test")
        ]
        if subset.empty:
            continue
        row = subset.iloc[0]
        test_rows.append(
            f"| {algorithm} | {row['avg_return_mean']:.2f} | {row['severe_miss_rate_mean']:.3f} | {row['false_warning_rate_mean']:.3f} |"
        )
    report_text = (
        "# Frozen Baseline\n\n"
        "This folder freezes the baseline before the outcome-driven reward and longer hidden-trace refactor.\n\n"
        "## Test Snapshot\n\n"
        "| Policy | Avg Return | Severe Miss Rate | False Warning Rate |\n"
        "|---|---:|---:|---:|\n"
        + "\n".join(test_rows)
        + "\n"
    )
    (baseline_dir / "baseline_report.md").write_text(report_text, encoding="utf-8")
    marker_path.write_text(
        json.dumps({"frozen": True, "run_seeds": run_seeds}, indent=2),
        encoding="utf-8",
    )


def _train_algorithm(
    algorithm: str,
    train_catalog: pd.DataFrame,
    validation_catalog: pd.DataFrame,
    test_catalog: pd.DataFrame,
    run_seed: int,
    config: TrainConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_env = TsunamiWarningEnv(train_catalog, seed=100 + run_seed, weight_column="training_weight")

    if algorithm == "q_learning":
        q_table, training_history = train_q_learning(
            train_env,
            config=config,
            seed=200 + run_seed,
            run_seed=run_seed,
        )
    else:
        q_table, training_history = train_sarsa(
            train_env,
            config=config,
            seed=300 + run_seed,
            run_seed=run_seed,
        )

    policies = [
        (f"{algorithm}_pure", pure_greedy_policy(q_table), 4000 + run_seed * 1000),
        (f"{algorithm}_safe", safe_greedy_policy(q_table, fallback_margin=2.5), 5000 + run_seed * 1000),
    ]

    evaluation_episodes_all: list[pd.DataFrame] = []
    evaluation_summaries_all: list[pd.DataFrame] = []
    for policy_name, policy, seed_base in policies:
        validation_episodes, validation_summary = evaluate_policy_on_catalog(
            validation_catalog,
            policy,
            algorithm_name=policy_name,
            split_name="validation",
            run_seed=run_seed,
            seed_base=seed_base,
        )
        test_episodes, test_summary = evaluate_policy_on_catalog(
            test_catalog,
            policy,
            algorithm_name=policy_name,
            split_name="test",
            run_seed=run_seed,
            seed_base=seed_base + 500,
        )
        evaluation_episodes_all.append(pd.concat([validation_episodes, test_episodes], ignore_index=True))
        evaluation_summaries_all.append(pd.concat([validation_summary, test_summary], ignore_index=True))

    evaluation_episodes = pd.concat(evaluation_episodes_all, ignore_index=True)
    evaluation_summary = pd.concat(evaluation_summaries_all, ignore_index=True)
    return training_history, evaluation_episodes, evaluation_summary


def _evaluate_rule_baseline(
    validation_catalog: pd.DataFrame,
    test_catalog: pd.DataFrame,
    run_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_episodes, validation_summary = evaluate_policy_on_catalog(
        validation_catalog,
        rule_based_policy,
        algorithm_name="rule_based",
        split_name="validation",
        run_seed=run_seed,
        seed_base=7000 + run_seed * 1000,
    )
    test_episodes, test_summary = evaluate_policy_on_catalog(
        test_catalog,
        rule_based_policy,
        algorithm_name="rule_based",
        split_name="test",
        run_seed=run_seed,
        seed_base=8000 + run_seed * 1000,
    )
    return (
        pd.concat([validation_episodes, test_episodes], ignore_index=True),
        pd.concat([validation_summary, test_summary], ignore_index=True),
    )


def run_experiments(root: Path, event_summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    logger = setup_logger(ProjectPaths(root))
    figures_dir, tables_dir = _ensure_output_dirs(root)

    split_df = _with_time_splits(event_summary)
    split_df.to_csv(tables_dir / "event_summary_with_splits.csv", index=False)
    _save_data_summary(split_df, figures_dir, tables_dir)

    train_catalog = split_df[split_df["time_split"] == "train"].copy()
    validation_catalog = split_df[split_df["time_split"] == "validation"].copy()
    test_catalog = split_df[split_df["time_split"] == "test"].copy()
    synthetic_train_catalog = generate_synthetic_training_catalog(
        train_catalog,
        synthetic_multiplier=1.0,
        seed=20260327,
    )
    synthetic_train_catalog.to_csv(root / "data" / "processed" / "synthetic_train_events.csv", index=False)
    synthetic_counts = (
        synthetic_train_catalog.groupby("danger_label")
        .size()
        .reset_index(name="count")
    )
    synthetic_counts.to_csv(tables_dir / "synthetic_train_label_counts.csv", index=False)
    augmented_train_catalog = pd.concat([train_catalog, synthetic_train_catalog], ignore_index=True)

    config = TrainConfig()
    run_seeds = [11, 22, 33]
    ppo_device = "cuda" if torch.cuda.is_available() else "cpu"
    ppo_config = PPOConfig(
        total_steps=_parse_int_env("PPO_TOTAL_STEPS", 200_000 if ppo_device == "cuda" else 40_000),
        n_envs=_parse_int_env("PPO_N_ENVS", 16 if ppo_device == "cuda" else 4),
        rollout_steps=_parse_int_env("PPO_ROLLOUT_STEPS", 256 if ppo_device == "cuda" else 64),
        update_epochs=4,
        minibatches=_parse_int_env("PPO_MINIBATCHES", 8 if ppo_device == "cuda" else 4),
        seed=77,
        eval_interval=_parse_int_env("PPO_EVAL_INTERVAL", 50_000 if ppo_device == "cuda" else 10_000),
        device=ppo_device,
        normalize_observations=os.getenv("PPO_NORMALIZE_OBSERVATIONS", "1") != "0",
        use_curriculum=os.getenv("PPO_USE_CURRICULUM", "1") != "0",
    )
    ppo_run_seeds = _parse_int_list_env("PPO_RUN_SEEDS", [77])
    resolved_ppo_device = resolve_device(ppo_config.device)

    config_df = pd.DataFrame(
        [
            {"parameter": "episodes", "value": config.episodes},
            {"parameter": "alpha", "value": config.alpha},
            {"parameter": "gamma", "value": config.gamma},
            {"parameter": "epsilon_start", "value": config.epsilon_start},
            {"parameter": "epsilon_end", "value": config.epsilon_end},
            {"parameter": "rule_warm_start_episodes", "value": config.rule_warm_start_episodes},
            {"parameter": "warm_start_alpha", "value": config.warm_start_alpha},
            {"parameter": "synthetic_training_multiplier", "value": 1.0},
            {"parameter": "real_train_event_count", "value": len(train_catalog)},
            {"parameter": "synthetic_train_event_count", "value": len(synthetic_train_catalog)},
            {"parameter": "augmented_train_event_count", "value": len(augmented_train_catalog)},
            {"parameter": "policy_variants", "value": "pure_greedy, safe_rule_fallback"},
            {"parameter": "safe_policy_fallback_margin", "value": 2.5},
            {"parameter": "reward_schedule_shaping_note", "value": "schedule shaping removed; reward is outcome- and evidence-driven"},
            {"parameter": "episode_horizon_steps", "value": 12},
            {"parameter": "dp_baseline_note", "value": "fully observed oracle reference, not operational comparator"},
            {"parameter": "ppo_total_steps", "value": ppo_config.total_steps},
            {"parameter": "ppo_n_envs", "value": ppo_config.n_envs},
            {"parameter": "ppo_rollout_steps", "value": ppo_config.rollout_steps},
            {"parameter": "ppo_update_epochs", "value": ppo_config.update_epochs},
            {"parameter": "ppo_minibatches", "value": ppo_config.minibatches},
            {"parameter": "ppo_hidden_size", "value": ppo_config.hidden_size},
            {"parameter": "ppo_lstm_size", "value": ppo_config.lstm_size},
            {"parameter": "ppo_eval_interval", "value": ppo_config.eval_interval},
            {"parameter": "ppo_requested_device", "value": ppo_config.device},
            {"parameter": "ppo_resolved_device", "value": str(resolved_ppo_device)},
            {"parameter": "ppo_run_seeds", "value": ",".join(str(seed) for seed in ppo_run_seeds)},
            {"parameter": "ppo_severe_miss_penalty", "value": ppo_config.severe_miss_penalty},
            {"parameter": "ppo_false_warning_penalty", "value": ppo_config.false_warning_penalty},
            {"parameter": "ppo_selection_metric", "value": "validation_safety_score"},
            {"parameter": "ppo_normalize_observations", "value": int(ppo_config.normalize_observations)},
            {"parameter": "ppo_use_curriculum", "value": int(ppo_config.use_curriculum)},
            {"parameter": "ppo_curriculum_stage_fractions", "value": ",".join(str(value) for value in ppo_config.curriculum_stage_fractions)},
        ]
    )
    config_df.to_csv(tables_dir / "experiment_configuration.csv", index=False)

    training_histories: list[pd.DataFrame] = []
    evaluation_episodes_all: list[pd.DataFrame] = []
    evaluation_by_seed_all: list[pd.DataFrame] = []

    for algorithm in ["q_learning", "sarsa"]:
        for run_seed in run_seeds:
            training_history, evaluation_episodes, evaluation_summary = _train_algorithm(
                algorithm,
                augmented_train_catalog,
                validation_catalog,
                test_catalog,
                run_seed,
                config,
            )
            training_histories.append(training_history)
            evaluation_episodes_all.append(evaluation_episodes)
            evaluation_by_seed_all.append(evaluation_summary)

    for run_seed in run_seeds:
        rule_episodes, rule_summary = _evaluate_rule_baseline(
            validation_catalog,
            test_catalog,
            run_seed,
        )
        evaluation_episodes_all.append(rule_episodes)
        evaluation_by_seed_all.append(rule_summary)

    training_history_df = pd.concat(training_histories, ignore_index=True)
    evaluation_episodes_df = pd.concat(evaluation_episodes_all, ignore_index=True)
    evaluation_by_seed_df = pd.concat(evaluation_by_seed_all, ignore_index=True)

    ppo_histories: list[pd.DataFrame] = []
    best_ppo_seed: int | None = None
    best_ppo_validation_score = -float("inf")
    for ppo_seed in ppo_run_seeds:
        run_config = PPOConfig(**{**ppo_config.__dict__, "seed": ppo_seed})
        artifact_prefix = f"ppo_lstm_seed{ppo_seed}"
        ppo_model, ppo_history_df = train_ppo_lstm(
            augmented_train_catalog,
            run_config,
            validation_catalog=validation_catalog,
            output_dir=root / "outputs",
            weight_column="training_weight",
            artifact_prefix=artifact_prefix,
        )
        ppo_history_df["run_seed"] = ppo_seed
        ppo_histories.append(ppo_history_df)

        ppo_validation_episodes, ppo_validation_summary = evaluate_ppo_lstm_on_catalog(
            validation_catalog,
            ppo_model,
            algorithm_name="ppo_lstm_pure",
            split_name="validation",
            run_seed=ppo_seed,
            seed_base=92000 + ppo_seed * 100,
            device=resolved_ppo_device,
        )
        ppo_test_episodes, ppo_test_summary = evaluate_ppo_lstm_on_catalog(
            test_catalog,
            ppo_model,
            algorithm_name="ppo_lstm_pure",
            split_name="test",
            run_seed=ppo_seed,
            seed_base=93000 + ppo_seed * 100,
            device=resolved_ppo_device,
        )
        evaluation_episodes_df = pd.concat(
            [evaluation_episodes_df, ppo_validation_episodes, ppo_test_episodes],
            ignore_index=True,
        )
        evaluation_by_seed_df = pd.concat(
            [evaluation_by_seed_df, ppo_validation_summary, ppo_test_summary],
            ignore_index=True,
        )

        validation_score = float(ppo_validation_summary.iloc[0]["safety_score"])
        if validation_score > best_ppo_validation_score:
            best_ppo_validation_score = validation_score
            best_ppo_seed = ppo_seed

    if best_ppo_seed is not None:
        outputs_models = root / "outputs" / "models"
        shutil.copy2(
            outputs_models / f"ppo_lstm_seed{best_ppo_seed}_best.pt",
            outputs_models / "ppo_lstm_best.pt",
        )
        shutil.copy2(
            outputs_models / f"ppo_lstm_seed{best_ppo_seed}_last.pt",
            outputs_models / "ppo_lstm_last.pt",
        )

    ppo_history_df = pd.concat(ppo_histories, ignore_index=True) if ppo_histories else pd.DataFrame()
    evaluation_summary_df = _aggregate_eval(evaluation_by_seed_df)

    training_history_df.to_csv(tables_dir / "training_history.csv", index=False)
    evaluation_episodes_df.to_csv(tables_dir / "agent_evaluation_episodes.csv", index=False)
    evaluation_by_seed_df.to_csv(tables_dir / "agent_evaluation_by_seed.csv", index=False)
    evaluation_summary_df.to_csv(tables_dir / "agent_evaluation_summary.csv", index=False)
    evaluation_episodes_df.to_csv(tables_dir / "evaluation_episode_metrics.csv", index=False)
    evaluation_summary_df.to_csv(tables_dir / "evaluation_summary.csv", index=False)
    if not ppo_history_df.empty:
        ppo_history_df.to_csv(tables_dir / "ppo_lstm_training_history.csv", index=False)
    _freeze_baseline_snapshot(root, config_df, split_df, evaluation_summary_df, run_seeds)

    mdp = ToyTsunamiMDP()
    _, optimal_policy = value_iteration(mdp)
    dp_policy_df = policy_table(optimal_policy)
    dp_policy_df.to_csv(tables_dir / "dp_optimal_policy.csv", index=False)

    _save_training_curve(training_history_df, figures_dir)
    logger.info("Saved training and evaluation artifacts")
    return {
        "training_history": training_history_df,
        "ppo_training_history": ppo_history_df,
        "evaluation_summary": evaluation_summary_df,
        "evaluation_by_seed": evaluation_by_seed_df,
        "evaluation_episodes": evaluation_episodes_df,
        "experiment_configuration": config_df,
        "dp_policy": dp_policy_df,
        "split_df": split_df,
    }
