from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalibrationBin:
    bin_start: float
    bin_end: float
    count: int
    avg_prediction: float
    actual_rate: float

    @property
    def gap(self) -> float:
        return abs(self.avg_prediction - self.actual_rate)


def brier_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = labels.astype(float)
    probabilities = np.clip(probabilities.astype(float), 0.0, 1.0)
    return float(np.mean((probabilities - labels) ** 2))


def calibration_bins(
    labels: np.ndarray,
    probabilities: np.ndarray,
    num_bins: int = 10,
) -> list[CalibrationBin]:
    labels = labels.astype(float)
    probabilities = np.clip(probabilities.astype(float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    bins: list[CalibrationBin] = []

    for index in range(num_bins):
        start = edges[index]
        end = edges[index + 1]
        if index == num_bins - 1:
            mask = (probabilities >= start) & (probabilities <= end)
        else:
            mask = (probabilities >= start) & (probabilities < end)

        count = int(mask.sum())
        if count == 0:
            bins.append(
                CalibrationBin(
                    bin_start=float(start),
                    bin_end=float(end),
                    count=0,
                    avg_prediction=0.0,
                    actual_rate=0.0,
                )
            )
            continue

        bins.append(
            CalibrationBin(
                bin_start=float(start),
                bin_end=float(end),
                count=count,
                avg_prediction=float(probabilities[mask].mean()),
                actual_rate=float(labels[mask].mean()),
            )
        )

    return bins


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    num_bins: int = 10,
) -> float:
    bins = calibration_bins(labels, probabilities, num_bins)
    total = sum(bin_result.count for bin_result in bins)
    if total == 0:
        return 0.0
    return float(
        sum((bin_result.count / total) * bin_result.gap for bin_result in bins)
    )


def calibration_table(
    labels: np.ndarray,
    probabilities: np.ndarray,
    num_bins: int = 10,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "bin_start": bin_result.bin_start,
                "bin_end": bin_result.bin_end,
                "count": bin_result.count,
                "avg_prediction": bin_result.avg_prediction,
                "actual_rate": bin_result.actual_rate,
                "gap": bin_result.gap,
            }
            for bin_result in calibration_bins(labels, probabilities, num_bins)
        ]
    )

