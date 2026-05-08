from __future__ import annotations

from pathlib import Path

import pandas as pd

from ads_ranking.datasets.aliccp import load_aliccp_tabular_sample


def load_real_world_ads_data(path: str | Path, dataset_name: str, num_rows: int | None = None) -> pd.DataFrame:
    """Placeholder entry point for public ads datasets such as Ali-CCP, Avazu, or iPinYou."""
    if dataset_name.lower() in {"aliccp", "ali-ccp"}:
        return load_aliccp_tabular_sample(path, num_rows=num_rows).frame
    raise NotImplementedError(
        f"Real-world loader for '{dataset_name}' is not implemented yet. "
        "Add a dataset-specific loader before running real-world experiments."
    )
