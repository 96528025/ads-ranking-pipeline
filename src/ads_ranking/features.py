from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PriorSpec:
    name: str
    columns: list[str]
    target: str


class HistoricalPriorFeatureEngineer:
    """Fit smoothed historical rate features on train data and apply them to any split."""

    def __init__(self, specs: list[PriorSpec], smoothing: float = 20.0):
        self.specs = specs
        self.smoothing = smoothing
        self.global_rates: dict[str, float] = {}
        self.tables: dict[str, pd.DataFrame] = {}

    def fit(self, frame: pd.DataFrame) -> "HistoricalPriorFeatureEngineer":
        for spec in self.specs:
            global_rate = float(frame[spec.target].mean())
            grouped = (
                frame.groupby(spec.columns, dropna=False)[spec.target]
                .agg(["sum", "count"])
                .reset_index()
            )
            grouped[spec.name] = (
                grouped["sum"] + self.smoothing * global_rate
            ) / (grouped["count"] + self.smoothing)
            self.global_rates[spec.name] = global_rate
            self.tables[spec.name] = grouped[[*spec.columns, spec.name]]
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed = frame.copy()
        for spec in self.specs:
            transformed = transformed.merge(
                self.tables[spec.name],
                on=spec.columns,
                how="left",
            )
            transformed[spec.name] = transformed[spec.name].fillna(self.global_rates[spec.name])

        if {"user_ad_ctr_prior", "category_ctr_prior"}.issubset(transformed.columns):
            transformed["user_ad_ctr_lift"] = (
                transformed["user_ad_ctr_prior"] - transformed["category_ctr_prior"]
            )
        if {"user_ad_cvr_prior", "category_cvr_prior"}.issubset(transformed.columns):
            transformed["user_ad_cvr_lift"] = (
                transformed["user_ad_cvr_prior"] - transformed["category_cvr_prior"]
            )

        return transformed

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)


def build_default_prior_specs() -> list[PriorSpec]:
    return [
        PriorSpec("category_ctr_prior", ["ad_category"], "click"),
        PriorSpec("creative_ctr_prior", ["creative_type"], "click"),
        PriorSpec("hour_ctr_prior", ["hour_bucket"], "click"),
        PriorSpec("user_segment_ctr_prior", ["user_segment"], "click"),
        PriorSpec("user_ad_ctr_prior", ["user_segment", "ad_category"], "click"),
        PriorSpec("category_cvr_prior", ["ad_category"], "post_click_conversion"),
        PriorSpec("creative_cvr_prior", ["creative_type"], "post_click_conversion"),
        PriorSpec("user_ad_cvr_prior", ["user_segment", "ad_category"], "post_click_conversion"),
    ]


def default_engineered_feature_names() -> list[str]:
    return [
        "category_ctr_prior",
        "creative_ctr_prior",
        "hour_ctr_prior",
        "user_segment_ctr_prior",
        "user_ad_ctr_prior",
        "user_ad_ctr_lift",
        "category_cvr_prior",
        "creative_cvr_prior",
        "user_ad_cvr_prior",
        "user_ad_cvr_lift",
    ]

