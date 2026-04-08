from __future__ import annotations

from math import sqrt

import pandas as pd


def with_time_splits(event_summary: pd.DataFrame) -> pd.DataFrame:
    df = event_summary.copy()
    df["origin_time_utc"] = pd.to_datetime(df["origin_time_utc"], utc=True, errors="coerce")
    split_frames: list[pd.DataFrame] = []

    for _, group in df.groupby("danger_label", sort=False):
        group = group.sort_values("origin_time_utc").reset_index(drop=True).copy()
        count = len(group)
        split_labels = ["train"] * count

        if count >= 3:
            train_count = max(1, min(count - 2, int(count * 0.70)))
            validation_count = max(1, min(count - train_count - 1, int(count * 0.15)))
            test_count = count - train_count - validation_count
            if test_count <= 0:
                validation_count = max(1, validation_count - 1)
                test_count = count - train_count - validation_count

            for index in range(train_count, train_count + validation_count):
                split_labels[index] = "validation"
            for index in range(train_count + validation_count, count):
                split_labels[index] = "test"
        elif count == 2:
            split_labels = ["train", "test"]

        group["time_split"] = split_labels
        split_frames.append(group)

    return pd.concat(split_frames, ignore_index=True).sort_values("origin_time_utc").reset_index(drop=True)


def metric_ci95(values: pd.Series) -> float:
    clean = values.dropna()
    if len(clean) <= 1:
        return 0.0
    return float(1.96 * clean.std(ddof=1) / sqrt(len(clean)))


def aggregate_eval(by_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    metrics = [
        "avg_return",
        "median_return",
        "severe_miss_rate",
        "false_warning_rate",
        "avg_alert_changes",
        "avg_invalid_actions",
        "avg_warning_step_on_severe",
        "avg_watch_step_on_potential",
        "safety_score",
    ]
    for (algorithm, split_name), group in by_seed.groupby(["algorithm", "split"], sort=False):
        row = {
            "algorithm": algorithm,
            "split": split_name,
            "runs": len(group),
            "episode_count": int(group["episode_count"].iloc[0]),
            "no_threat_event_count": int(group["no_threat_event_count"].iloc[0]),
            "potential_event_count": int(group["potential_event_count"].iloc[0]),
            "severe_event_count": int(group["severe_event_count"].iloc[0]),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_ci95"] = metric_ci95(group[metric])
        rows.append(row)
    return pd.DataFrame(rows)
