from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analyze.ml_model import compare_with_v3, compute_market_gap, prepare_features, train_cv, tune_hyperparams


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
            "price_x_saturation",
            "pop_x_station_dist",
        }.issubset(features.columns)
        assert "pop_x_genre" not in features.columns
        assert "neighbor_pop_x_genre" not in features.columns
        assert features.loc[0, "other_genre_count"] == pytest.approx(np.log1p(3))
        assert features.loc[1, "google_avg_rating"] == pytest.approx(0.0)
        assert target.iloc[0] == pytest.approx(np.log1p(2))

    def test_interaction_feature_values(self, ml_df: pd.DataFrame) -> None:
        features, _ = prepare_features(ml_df)
        assert features.loc[0, "price_x_saturation"] == pytest.approx(
            features.loc[0, "land_price"] * features.loc[0, "saturation_index"]
        )
        assert features.loc[0, "pop_x_station_dist"] == pytest.approx(
            features.loc[0, "population"] * features.loc[0, "nearest_station_distance"]
        )

    def test_prepare_features_residual_mode(self, ml_df: pd.DataFrame) -> None:
        features, target = prepare_features(ml_df, target_mode="residual")
        assert not features.empty
        assert target.mean() == pytest.approx(0.0, abs=1.0)


class TestTrainCv:
    def test_train_cv_result_keys(self, ml_df: pd.DataFrame) -> None:
        result = train_cv(ml_df, n_splits=2, num_rounds=20)
        assert {"fold_metrics", "avg_rmse", "avg_r2", "oof_predictions", "feature_importance", "n_filtered"} <= result.keys()
        assert len(result["fold_metrics"]) == 2
        assert len(result["oof_predictions"]) == len(ml_df)

    def test_train_cv_uses_group_kfold_when_group_col_exists(self, ml_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
        called = {"groups": None}

        class DummyGroupKFold:
            def __init__(self, n_splits: int) -> None:
                self.n_splits = n_splits

            def split(self, X: pd.DataFrame, y: pd.Series | None = None, groups: pd.Series | None = None):
                called["groups"] = groups
                yield np.array([0, 1, 2, 3]), np.array([4, 5])
                yield np.array([2, 3, 4, 5]), np.array([0, 1])

        monkeypatch.setattr("src.analyze.ml_model.GroupKFold", DummyGroupKFold)
        result = train_cv(ml_df, n_splits=2, num_rounds=5)
        assert called["groups"] is not None
        assert set(called["groups"].astype(str)) == {"m1", "m2", "m3"}
        assert len(result["oof_predictions"]) == len(ml_df)

    def test_train_cv_falls_back_to_kfold_without_group_col(self, ml_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
        called = {"used": False}

        class DummyKFold:
            def __init__(self, n_splits: int, shuffle: bool, random_state: int) -> None:
                self.n_splits = n_splits

            def split(self, X: pd.DataFrame, y: pd.Series | None = None, groups: pd.Series | None = None):
                called["used"] = True
                yield np.array([0, 1, 2, 3]), np.array([4, 5])
                yield np.array([2, 3, 4, 5]), np.array([0, 1])

        monkeypatch.setattr("src.analyze.ml_model.KFold", DummyKFold)
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
            group_col: str = "jis_mesh",
            target_mode: str = "raw",
        ) -> dict:
            calls.append(
                {
                    "n_splits": n_splits,
                    "params": params,
                    "num_rounds": num_rounds,
                    "group_col": group_col,
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

        result = tune_hyperparams(ml_df, n_trials=3, n_splits=2, group_col="jis_mesh")

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
        assert calls[0]["group_col"] == "jis_mesh"
        assert calls[0]["num_rounds"] == 120
