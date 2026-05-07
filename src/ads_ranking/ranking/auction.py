from __future__ import annotations

import pandas as pd


def add_auction_scores(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    bid_col = "raw_bid" if "raw_bid" in ranked.columns else "bid"
    ranked["ctr_score"] = ranked["pred_ctr"]
    ranked["conversion_value_score"] = ranked["pred_ctr"] * ranked["pred_cvr"]
    ranked["expected_value_score"] = ranked["pred_ctr"] * ranked["pred_cvr"] * ranked[bid_col]
    return ranked
