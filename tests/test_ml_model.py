from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analyze.ml_model import (
    compare_with_v3,
    compute_market_gap,
    filter_recommendations,
    prepare_features,
    train_cv,
    tune_hyperparams,
)


@pytest.fixture()
def ml_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "jis_mesh": ["m1", "m1", "m2", "m2", "m3", "m3"],
            "population": [10000, 15000, 20000, 25000, 30000, 35000],
            "genre_diversity": [2, 3, 2, 4, 3, 5],
            "genre_hhi": [0.6, 0.4, 0.5, 0.3, 0.35, 0.25],
            "other_genre_count": [3, 2, 5, 4, 7, 6],
            "neighbor_avg_restaurants": [5, 6, 7, 8, 9, 10],
            "neighbor_avg_population": [1000, 1100, 1200, 1300, 1400, 1500],
            "saturation_index": [1.2, 1.5, 1.7, 2.0, 2.2, 2.5],
            "nearest_station_distance": [0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
            "nearest_station_passengers": [10000, 12000, 14000, 16000, 18000, 20000],
            "land_price": [100, 110, 120, 130, 140, 150],
            "google_match_count": [1, 0, 1, 1, 0, 1],
            "google_avg_rating": [4.2, 0.0, 4.0, 3.8, 0.0, 4.5],
            "google_total_reviews": [20, 0, 10, 5, 0, 30],
            "reviews_per_shop": [20, 0, 10, 5, 0, 30],
            "google_density": [0.8, 0.9, 1.1, 1.2, 1.4, 1.5],
            "unified_genre": ["cafe", "ramen", "cafe", "izakaya", "washoku", "cafe"],
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
            "other_genre_count",
            "neighbor_avg_restaurants",
            "neighbor_avg_population",
            "saturation_index",
            "nearest_station_distance",
            "nearest_station_passengers",
            "google_avg_rating",
            "reviews_per_shop",
            "google_density",
            "genre_encoded",
        }.issubset(features.columns)
        assert "pop_x_genre" not in features.columns
        assert "neighbor_pop_x_genre" not in features.columns
        assert "price_x_saturation" not in features.columns
        assert "pop_x_station_dist" not in features.columns
        assert features.loc[0, "other_genre_count"] == pytest.approx(np.log1p(3))
        assert features.loc[1, "google_avg_rating"] == pytest.approx(0.0)
        assert target.iloc[0] == pytest.approx(np.log1p(2))

    def test_interaction_features_are_not_generated(self, ml_df: pd.DataFrame) -> None:
        """交互作用特徴量が生成されないことを検証する。"""
        features, _ = prepare_features(ml_df)
        assert "price_x_saturation" not in features.columns
        assert "pop_x_station_dist" not in features.columns

    def test_prepare_features_residual_mode(self, ml_df: pd.DataFrame) -> None:
        features, target = prepare_features(ml_df, target_mode="residual")
        assert not features.empty
        assert target.mean() == pytest.approx(0.0, abs=1.0)

    def test_genre_encoded_stable_across_subsets(self, ml_df: pd.DataFrame) -> None:
        """異なるサブセットでもgenre_encodedの値が同じことを検証。"""
        full_features, _ = prepare_features(ml_df)
        subset = ml_df[ml_df["unified_genre"].isin(["cafe", "ramen"])].reset_index(drop=True)
        subset_features, _ = prepare_features(subset)

        cafe_code_full = full_features.loc[ml_df["unified_genre"] == "cafe", "genre_encoded"].iloc[0]
        cafe_code_subset = subset_features.loc[subset["unified_genre"] == "cafe", "genre_encoded"].iloc[0]

        assert cafe_code_full == cafe_code_subset


class TestTrainCv:
    def test_train_cv_result_keys(self, ml_df: pd.DataFrame) -> None:
        result = train_cv(ml_df, n_splits=2, num_rounds=20)
        assert {"fold_metrics", "avg_rmse", "avg_r2", "oof_predictions", "feature_importance", "n_filtered"} <= result.keys()
        assert len(result["fold_metrics"]) == 2
        assert len(result["oof_predictions"]) == len(ml_df)

    def test_train_cv_returns_fold_predictions(self, ml_df: pd.DataFrame) -> None:
        result = train_cv(ml_df, n_splits=2, num_rounds=20)
        assert "fold_predictions" in result
        assert result["fold_predictions"].shape == (len(ml_df), 2)
        assert np.any(result["fold_predictions"] != 0)

    def test_train_cv_uses_stratified_kfold_bins(self, ml_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
        """restaurant_count の層化ビンを使って分割することを検証する。"""
        called = {"bins": None}

        class DummyStratifiedKFold:
            """StratifiedKFold の呼び出し引数を記録するダミー。"""

            def __init__(self, n_splits: int, shuffle: bool, random_state: int) -> None:
                self.n_splits = n_splits
                self.shuffle = shuffle
                self.random_state = random_state

            def split(self, X: pd.DataFrame, y: pd.Series | None = None):
                called["bins"] = y
                yield np.array([0, 1, 2, 3]), np.array([4, 5])
                yield np.array([2, 3, 4, 5]), np.array([0, 1])

        monkeypatch.setattr("src.analyze.ml_model.StratifiedKFold", DummyStratifiedKFold)
        result = train_cv(ml_df, n_splits=2, num_rounds=5)
        assert called["bins"] is not None
        assert set(pd.Series(called["bins"]).dropna().unique()) <= {0, 1}
        assert len(result["oof_predictions"]) == len(ml_df)

    def test_train_cv_uses_stratified_kfold_without_mesh_col(
        self, ml_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """メッシュ列がなくても層化分割を使うことを検証する。"""
        called = {"used": False}

        class DummyStratifiedKFold:
            """StratifiedKFold の利用有無を記録するダミー。"""

            def __init__(self, n_splits: int, shuffle: bool, random_state: int) -> None:
                self.n_splits = n_splits
                self.shuffle = shuffle
                self.random_state = random_state

            def split(self, X: pd.DataFrame, y: pd.Series | None = None):
                called["used"] = True
                yield np.array([0, 1, 2, 3]), np.array([4, 5])
                yield np.array([2, 3, 4, 5]), np.array([0, 1])

        monkeypatch.setattr("src.analyze.ml_model.StratifiedKFold", DummyStratifiedKFold)
        result = train_cv(ml_df.drop(columns=["jis_mesh"]), n_splits=2, num_rounds=5)
        assert called["used"] is True
        assert len(result["oof_predictions"]) == len(ml_df)


class TestComputeMarketGap:
    def test_compute_market_gap(self, ml_df: pd.DataFrame) -> None:
        predictions = np.log1p(np.array([3, 4, 5, 6, 7, 8], dtype=float))
        result = compute_market_gap(ml_df, predictions)
        assert result.loc[0, "market_gap"] == pytest.approx(np.log1p(3) - np.log1p(2))
        assert result.loc[0, "predicted_count"] == pytest.approx(3.0)
        assert result.loc[0, "gap_count"] == pytest.approx(1.0)

    def test_compute_market_gap_with_ci(self, ml_df: pd.DataFrame) -> None:
        predictions = np.array(
            [1.80, 1.55, 1.50, np.log1p(6), np.log1p(7), np.log1p(8)],
            dtype=float,
        )
        fold_predictions = np.array(
            [
                [1.00, 1.20],
                [1.40, 1.60],
                [1.40, 1.60],
                [1.80, 2.00],
                [1.90, 2.10],
                [2.00, 2.20],
            ],
            dtype=float,
        )
        result = compute_market_gap(ml_df, predictions, fold_predictions=fold_predictions)
        assert {"gap_std", "gap_ci_lower", "gap_ci_upper", "gap_reliability"} <= set(result.columns)
        assert (result["gap_ci_lower"] < result["market_gap"]).all()
        assert (result["market_gap"] < result["gap_ci_upper"]).all()
        assert result.loc[0, "market_gap"] > 0
        assert result.loc[0, "gap_ci_lower"] > 0
        assert result.loc[0, "gap_reliability"] == "high"
        assert result.loc[1, "market_gap"] > 0
        assert result.loc[1, "gap_ci_lower"] <= 0
        assert result.loc[1, "gap_reliability"] == "medium"
        assert result.loc[2, "market_gap"] <= 0
        assert result.loc[2, "gap_reliability"] == "low"

    def test_reliability_boundary_ci_lower_zero(self, ml_df: pd.DataFrame) -> None:
        actual = np.log1p(2)
        predictions = np.array(
            [actual + 0.196, np.log1p(4), np.log1p(5), np.log1p(6), np.log1p(7), np.log1p(8)],
            dtype=float,
        )
        fold_predictions = np.array(
            [
                [0.90, 1.10],
                [1.10, 1.30],
                [1.30, 1.50],
                [1.50, 1.70],
                [1.70, 1.90],
                [1.90, 2.10],
            ],
            dtype=float,
        )
        result = compute_market_gap(ml_df, predictions, fold_predictions=fold_predictions)
        assert result.loc[0, "gap_ci_lower"] == pytest.approx(0.0, abs=1e-12)
        assert result.loc[0, "gap_reliability"] == "medium"

    def test_reliability_boundary_gap_zero(self, ml_df: pd.DataFrame) -> None:
        predictions = np.log1p(ml_df["restaurant_count"].to_numpy(dtype=float))
        fold_predictions = np.array(
            [
                [0.90, 1.10],
                [1.10, 1.30],
                [1.30, 1.50],
                [1.50, 1.70],
                [1.70, 1.90],
                [1.90, 2.10],
            ],
            dtype=float,
        )
        result = compute_market_gap(ml_df, predictions, fold_predictions=fold_predictions)
        assert result.loc[0, "market_gap"] == pytest.approx(0.0)
        assert result.loc[0, "gap_reliability"] == "low"

    def test_compute_market_gap_without_ci(self, ml_df: pd.DataFrame) -> None:
        predictions = np.log1p(np.array([3, 4, 5, 6, 7, 8], dtype=float))
        result = compute_market_gap(ml_df, predictions)
        assert "gap_std" not in result.columns
        assert "gap_ci_lower" not in result.columns
        assert "gap_ci_upper" not in result.columns
        assert "gap_reliability" not in result.columns


class TestFilterRecommendations:
    def test_filter_by_rc(self, ml_df: pd.DataFrame) -> None:
        df = ml_df.assign(gap_count=1.5, market_gap=0.5, gap_reliability="high")
        result = filter_recommendations(df, min_rc=5, min_gap_count=1.0, min_reliability="medium")
        assert not result.empty
        assert (result["restaurant_count"] >= 5).all()

    def test_filter_by_gap_count(self, ml_df: pd.DataFrame) -> None:
        df = ml_df.assign(gap_count=[0.5, 1.0, 1.5, 0.9, 2.0, 3.0], market_gap=0.5, gap_reliability="high")
        result = filter_recommendations(df, min_rc=2, min_gap_count=1.0, min_reliability="medium")
        assert not result.empty
        assert (result["gap_count"] >= 1.0).all()

    def test_filter_by_reliability(self, ml_df: pd.DataFrame) -> None:
        df = ml_df.assign(
            gap_count=1.5,
            market_gap=0.5,
            gap_reliability=["high", "medium", "low", "medium", "high", "low"],
        )
        result = filter_recommendations(df, min_rc=2, min_gap_count=1.0, min_reliability="medium")
        assert not result.empty
        assert set(result["gap_reliability"]) <= {"high", "medium"}
        assert "low" not in set(result["gap_reliability"])

    def test_filter_empty_df(self) -> None:
        df = pd.DataFrame(columns=["restaurant_count", "gap_count", "gap_reliability", "market_gap"])
        result = filter_recommendations(df, min_rc=2, min_gap_count=1.0, min_reliability="medium")
        assert result.empty

    def test_filter_without_reliability_column(self, ml_df: pd.DataFrame) -> None:
        df = ml_df.assign(gap_count=1.5, market_gap=0.5).drop(columns=["opportunity_score"])
        result = filter_recommendations(df, min_rc=2, min_gap_count=1.0, min_reliability="medium")
        assert not result.empty
        assert "gap_reliability" not in result.columns


class TestCompareWithV3:
    def test_compare_with_v3_adds_rank_columns(self, ml_df: pd.DataFrame) -> None:
        df = ml_df.copy()
        df["market_gap"] = [0.5, 0.2, 0.8, 0.1, 0.3, 0.4]
        result = compare_with_v3(df)
        assert {"rank_ml", "rank_v3", "rank_diff"} <= set(result.columns)


class TestTuneHyperparams:
    def test_tune_hyperparams_returns_best_config(self, ml_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
        class DummyTrial:
            def __init__(self) -> None:
                self.user_attrs: dict[str, float] = {}

            def suggest_int(self, name: str, low: int, high: int) -> int:
                return {"num_leaves": 31, "min_child_samples": 10, "bagging_freq": 3, "num_boost_round": 120}[name]

            def suggest_float(
                self,
                name: str,
                low: float,
                high: float,
                log: bool = False,
            ) -> float:
                return {
                    "learning_rate": 0.05,
                    "feature_fraction": 0.9,
                    "bagging_fraction": 0.85,
                    "lambda_l1": 1.5,
                    "lambda_l2": 2.5,
                }[name]

            def set_user_attr(self, name: str, value: float) -> None:
                self.user_attrs[name] = value

        class DummyStudy:
            def __init__(self) -> None:
                self.best_params: dict[str, float] = {}

            def optimize(self, objective, n_trials: int) -> None:
                assert n_trials == 3
                trial = DummyTrial()
                score = objective(trial)
                assert score == pytest.approx(0.456)
                self.best_params = {
                    "num_leaves": 31,
                    "learning_rate": 0.05,
                    "min_child_samples": 10,
                    "feature_fraction": 0.9,
                    "bagging_fraction": 0.85,
                    "bagging_freq": 3,
                    "lambda_l1": 1.5,
                    "lambda_l2": 2.5,
                    "num_boost_round": 120,
                }

        class DummyLogging:
            WARNING = "warning"

            def __init__(self) -> None:
                self.level = None

            def set_verbosity(self, level) -> None:
                self.level = level

        class DummyOptuna:
            def __init__(self) -> None:
                self.logging = DummyLogging()

            def create_study(self, direction: str) -> DummyStudy:
                assert direction == "minimize"
                return DummyStudy()

        calls: list[dict] = []

        def fake_train_cv(
            df: pd.DataFrame,
            n_splits: int = 5,
            params: dict | None = None,
            num_rounds: int = 300,
            target_mode: str = "raw",
        ) -> dict:
            calls.append(
                {
                    "n_splits": n_splits,
                    "params": params,
                    "num_rounds": num_rounds,
                    "target_mode": target_mode,
                }
            )
            return {
                "fold_metrics": [],
                "avg_rmse": 0.456,
                "avg_r2": 0.789,
                "oof_predictions": np.zeros(len(df)),
                "feature_importance": pd.DataFrame({"feature": [], "importance": []}),
                "n_filtered": 0,
            }

        monkeypatch.setattr("src.analyze.ml_model.optuna", DummyOptuna())
        monkeypatch.setattr("src.analyze.ml_model.train_cv", fake_train_cv)

        result = tune_hyperparams(ml_df, n_trials=3, n_splits=2)

        assert result["best_params"] == {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "min_child_samples": 10,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.85,
            "bagging_freq": 3,
            "lambda_l1": 1.5,
            "lambda_l2": 2.5,
        }
        assert result["best_num_rounds"] == 120
        assert result["best_rmse"] == pytest.approx(0.456)
        assert result["best_r2"] == pytest.approx(0.789)
        assert len(calls) == 2
        assert calls[0]["n_splits"] == 2
        assert calls[0]["num_rounds"] == 120
