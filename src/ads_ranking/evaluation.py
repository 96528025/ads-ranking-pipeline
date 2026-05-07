from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return {
        "auc": float(roc_auc_score(labels, clipped)),
        "logloss": float(log_loss(labels, clipped)),
    }


def dcg_at_k(relevance: np.ndarray, k: int) -> float:
    values = relevance[:k]
    discounts = np.log2(np.arange(2, len(values) + 2))
    return float(np.sum(values / discounts))


def ndcg_at_k(frame: pd.DataFrame, group_col: str, label_col: str, score_col: str, k: int) -> float:
    scores: list[float] = []
    for _, group in frame.groupby(group_col):
        ranked = group.sort_values(score_col, ascending=False)
        ideal = group.sort_values(label_col, ascending=False)
        ideal_dcg = dcg_at_k(ideal[label_col].to_numpy(), k)
        if ideal_dcg == 0.0:
            continue
        scores.append(dcg_at_k(ranked[label_col].to_numpy(), k) / ideal_dcg)
    return float(np.mean(scores)) if scores else 0.0


def topk_average(frame: pd.DataFrame, group_col: str, score_col: str, value_col: str, k: int) -> float:
    selected = (
        frame.sort_values(score_col, ascending=False)
        .groupby(group_col)
        .head(k)
    )
    return float(selected[value_col].mean())

