from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


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


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def generate_synthetic_ads_data(num_samples: int, seed: int) -> pd.DataFrame:
    """Generate impression-level ads data with click and conversion labels."""
    rng = np.random.default_rng(seed)

    user_segment = rng.integers(0, 8, size=num_samples)
    device = rng.choice(["ios", "android", "web"], size=num_samples, p=[0.45, 0.45, 0.10])
    hour_bucket = rng.integers(0, 6, size=num_samples)
    ad_category = rng.integers(0, 12, size=num_samples)
    creative_type = rng.choice(["video", "image", "carousel"], size=num_samples, p=[0.55, 0.35, 0.10])

    user_activity = rng.gamma(shape=2.0, scale=1.0, size=num_samples)
    ad_quality = rng.beta(a=3.0, b=2.0, size=num_samples)
    bid = rng.lognormal(mean=0.0, sigma=0.45, size=num_samples)

    segment_pref = rng.normal(0.0, 0.5, size=8)
    category_quality = rng.normal(0.0, 0.45, size=12)
    segment_category_match = ((user_segment * 3 + ad_category) % 5 == 0).astype(float)

    device_effect = np.select(
        [device == "ios", device == "android", device == "web"],
        [0.18, 0.08, -0.08],
    )
    creative_effect = np.select(
        [creative_type == "video", creative_type == "image", creative_type == "carousel"],
        [0.25, 0.05, 0.12],
    )
    hour_effect = np.sin(hour_bucket / 6.0 * 2.0 * np.pi) * 0.15

    ctr_logit = (
        -2.4
        + segment_pref[user_segment]
        + category_quality[ad_category]
        + 0.35 * segment_category_match
        + device_effect
        + creative_effect
        + 0.18 * user_activity
        + 0.85 * ad_quality
        - 0.08 * bid
        + hour_effect
    )
    click_probability = sigmoid(ctr_logit)
    click = rng.binomial(1, click_probability)

    cvr_logit = (
        -2.6
        + 0.35 * segment_pref[user_segment]
        + 0.40 * category_quality[ad_category]
        + 1.15 * segment_category_match
        + 1.05 * ad_quality
        + 0.30 * np.log1p(user_activity)
        + 0.10 * bid
        + 0.28 * (creative_type == "video")
        + 0.16 * (device == "ios")
    )
    conversion_probability = sigmoid(cvr_logit)
    post_click_conversion = rng.binomial(1, conversion_probability)
    conversion = post_click_conversion * click

    return pd.DataFrame(
        {
            "impression_id": np.arange(num_samples),
            "request_id": np.arange(num_samples) // 10,
            "user_segment": user_segment.astype(str),
            "device": device,
            "hour_bucket": hour_bucket.astype(str),
            "ad_category": ad_category.astype(str),
            "creative_type": creative_type,
            "user_activity": user_activity,
            "ad_quality": ad_quality,
            "bid": bid,
            "click": click,
            "post_click_conversion": post_click_conversion,
            "conversion": conversion,
            "true_ctr": click_probability,
            "true_cvr": conversion_probability,
        }
    )


def encode_and_split(
    frame: pd.DataFrame,
    feature_spec: FeatureSpec,
    test_size: float,
    seed: int,
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
    train[feature_spec.numerical] = scaler.fit_transform(train[feature_spec.numerical])
    test[feature_spec.numerical] = scaler.transform(test[feature_spec.numerical])

    return EncodedData(
        train=train.reset_index(drop=True),
        test=test.reset_index(drop=True),
        feature_spec=feature_spec,
        category_sizes=category_sizes,
        scaler=scaler,
    )
