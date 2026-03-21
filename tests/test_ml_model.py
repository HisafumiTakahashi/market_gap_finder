from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analyze.ml_model import compare_with_v3, compute_market_gap, prepare_features, train_cv


@pytest.fixture()
def ml_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "population": [10000, 15000, 20000, 25000, 30000, 35000],
            "genre_diversity": [2, 3, 2, 4, 3, 5],
            "genre_hhi": [0.6, 0.4, 0.5, 0.3, 0.35, 0.25],
            "neighbor_avg_restaurants": [5, 6, 7, 8, 9, 10],
            "saturation_index": [1.2, 1.5, 1.7, 2.0, 2.2, 2.5],
            "nearest_station_distance": [0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
            "unified_genre": ["cafe", "ramen", "cafe", "izakaya", "sushi", "cafe"],
            "restaurant_count": [2, 3, 4, 5, 6, 7],
            "opportunity_score": [0.2, 0.5, 0.4, 0.7, 0.6, 0.9],
        }
    )


class TestPrepareFeatures:
    def test_feature_columns_and_log_target(self, ml_df: pd.DataFrame) -> None:
        features, target = prepare_features(ml_df)
        assert {
            "population",
            "genre_diversity",
            "genre_hhi",
            "neighbor_avg_restaurants",
            "saturation_index",
            "nearest_station_distance",
            "genre_encoded",
        }.issubset(features.columns)
        assert target.iloc[0] == pytest.approx(np.log1p(2))


class TestTrainCv:
    def test_train_cv_result_keys(self, ml_df: pd.DataFrame) -> None:
        result = train_cv(ml_df, n_splits=2, num_rounds=20)
        assert {"fold_metrics", "avg_rmse", "avg_r2", "oof_predictions", "feature_importance"} <= result.keys()
        assert len(result["fold_metrics"]) == 2
        assert len(result["oof_predictions"]) == len(ml_df)


class TestComputeMarketGap:
    def test_compute_market_gap(self, ml_df: pd.DataFrame) -> None:
        predictions = np.log1p(np.array([3, 4, 5, 6, 7, 8], dtype=float))
        result = compute_market_gap(ml_df, predictions)
        assert result.loc[0, "market_gap"] == pytest.approx(np.log1p(3) - np.log1p(2))
        assert result.loc[0, "predicted_count"] == pytest.approx(3.0)
        assert result.loc[0, "gap_count"] == pytest.approx(1.0)


class TestCompareWithV3:
    def test_compare_with_v3_adds_rank_columns(self, ml_df: pd.DataFrame) -> None:
        df = ml_df.copy()
        df["market_gap"] = [0.5, 0.2, 0.8, 0.1, 0.3, 0.4]
        result = compare_with_v3(df)
        assert {"rank_ml", "rank_v3", "rank_diff"} <= set(result.columns)
