from __future__ import annotations

from dataclasses import dataclass
from math import isnan


@dataclass(frozen=True)
class OperationalScoreWeights:
    severe_miss_penalty: float = 300.0
    false_warning_penalty: float = 50.0
    warning_delay_penalty: float = 0.0
    potential_delay_penalty: float = 0.0


def _safe_delay(value: float | int | None) -> float:
    if value is None:
        return 0.0
    value = float(value)
    if isnan(value):
        return 0.0
    return value


def compute_operational_score(
    avg_return: float,
    severe_miss_rate: float,
    false_warning_rate: float,
    avg_warning_step_on_severe: float | int | None = None,
    avg_watch_step_on_potential: float | int | None = None,
    *,
    weights: OperationalScoreWeights | None = None,
) -> float:
    active_weights = weights or OperationalScoreWeights()
    return float(
        avg_return
        - active_weights.severe_miss_penalty * severe_miss_rate
        - active_weights.false_warning_penalty * false_warning_rate
        - active_weights.warning_delay_penalty * _safe_delay(avg_warning_step_on_severe)
        - active_weights.potential_delay_penalty * _safe_delay(avg_watch_step_on_potential)
    )
