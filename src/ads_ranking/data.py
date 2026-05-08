from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from ads_ranking.features import HistoricalPriorFeatureEngineer, build_default_prior_specs


@dataclass(frozen=True)
class FeatureSpec:
    categorical: list[str]
    numerical: list[str]


@dataclass
class EncodedData:
    train: pd.DataFrame
    test: pd.DataFrame
    feature_spec: FeatureSpec
    category_sizes: dict[str, int]
    scaler: StandardScaler


class AdsDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, feature_spec: FeatureSpec, label_col: str):
        self.categorical = torch.tensor(
            frame[feature_spec.categorical].to_numpy(dtype=np.int64),
            dtype=torch.long,
        )
        self.numerical = torch.tensor(
            frame[feature_spec.numerical].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        self.labels = torch.tensor(frame[label_col].to_numpy(dtype=np.float32), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.categorical[index], self.numerical[index], self.labels[index]


def encode_and_split(
    frame: pd.DataFrame,
    feature_spec: FeatureSpec,
    test_size: float,
    seed: int,
    add_engineered_features: bool = False,
) -> EncodedData:
    encoded = frame.copy()
    encoded["raw_bid"] = encoded["bid"]
    category_sizes: dict[str, int] = {}

    for column in feature_spec.categorical:
        categories = sorted(encoded[column].astype(str).unique())
        mapping = {value: index for index, value in enumerate(categories)}
        encoded[column] = encoded[column].astype(str).map(mapping).astype(int)
        category_sizes[column] = len(categories)

    train, test = train_test_split(
        encoded,
        test_size=test_size,
        random_state=seed,
        stratify=encoded["click"],
    )

    scaler = StandardScaler()
    train = train.copy()
    test = test.copy()

    if add_engineered_features:
        engineer = HistoricalPriorFeatureEngineer(build_default_prior_specs())
        train = engineer.fit_transform(train)
        test = engineer.transform(test)

    train[feature_spec.numerical] = scaler.fit_transform(train[feature_spec.numerical])
    test[feature_spec.numerical] = scaler.transform(test[feature_spec.numerical])

    return EncodedData(
        train=train.reset_index(drop=True),
        test=test.reset_index(drop=True),
        feature_spec=feature_spec,
        category_sizes=category_sizes,
        scaler=scaler,
    )
