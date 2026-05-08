from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ads_ranking.data import FeatureSpec


@dataclass(frozen=True)
class LoadedAliCcpData:
    frame: pd.DataFrame
    feature_spec: FeatureSpec


def load_aliccp_tabular_sample(
    path: str | Path,
    num_rows: int | None = None,
    categorical_columns: list[str] | None = None,
    numerical_columns: list[str] | None = None,
) -> LoadedAliCcpData:
    """Load a preprocessed Ali-CCP sample in CSV or Parquet format.

    Expected labels:
    - click: impression-level click label.
    - conversion: conversion label.

    If explicit feature columns are not provided, object/category columns are treated as
    categorical and numeric non-label columns are treated as numerical.
    """
    data_path = Path(path)
    frame = read_table(data_path, num_rows=num_rows)
    frame = normalize_labels(frame)

    label_columns = {"click", "conversion", "post_click_conversion"}
    ignored_columns = label_columns | {"request_id", "impression_id", "raw_bid"}

    if categorical_columns is None:
        categorical_columns = [
            column
            for column in frame.columns
            if column not in ignored_columns
            and (
                pd.api.types.is_object_dtype(frame[column])
                or pd.api.types.is_categorical_dtype(frame[column])
                or pd.api.types.is_string_dtype(frame[column])
            )
        ]

    if numerical_columns is None:
        numerical_columns = [
            column
            for column in frame.columns
            if column not in ignored_columns
            and column not in categorical_columns
            and pd.api.types.is_numeric_dtype(frame[column])
        ]

    if not categorical_columns and not numerical_columns:
        raise ValueError("Ali-CCP sample must contain at least one feature column.")

    frame[categorical_columns] = frame[categorical_columns].astype(str).fillna("__missing__")
    frame[numerical_columns] = frame[numerical_columns].fillna(0.0)

    if "request_id" not in frame.columns:
        frame["request_id"] = range(len(frame))

    return LoadedAliCcpData(
        frame=frame,
        feature_spec=FeatureSpec(
            categorical=list(categorical_columns),
            numerical=list(numerical_columns),
        ),
    )


def read_table(path: Path, num_rows: int | None) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, nrows=num_rows)
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
        return frame.head(num_rows) if num_rows is not None else frame
    raise ValueError(f"Unsupported Ali-CCP sample format: {path.suffix}. Use CSV or Parquet.")


def normalize_labels(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "click" not in normalized.columns:
        raise ValueError("Ali-CCP sample must include a 'click' label column.")
    if "conversion" not in normalized.columns:
        raise ValueError("Ali-CCP sample must include a 'conversion' label column.")

    normalized["click"] = normalized["click"].astype(int)
    normalized["conversion"] = normalized["conversion"].astype(int)
    normalized["post_click_conversion"] = normalized["conversion"]
    return normalized
