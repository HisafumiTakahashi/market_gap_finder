"""Machine learning utilities for estimating restaurant count gaps."""

from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
try:
    import optuna
except ImportError:  # pragma: no cover - exercised in runtime environments without Optuna installed
    optuna = None
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

from config import settings

logger = logging.getLogger(__name__)

DEMAND_FEATURES = [
    "population",
    "pop_working",
    "households",
    "neighbor_avg_population",
    "nearest_station_passengers",
    "land_price",
    "nearest_station_distance",
]
NUMERIC_FEATURES = [
    "population",
    "pop_working",
    "pop_adult",
    "pop_elderly",
    "households",
    "single_households",
    "young_single",
    "working_ratio",
    "elderly_ratio",
    "single_ratio",
    "young_single_ratio",
    "genre_diversity",
    "genre_hhi",
    "other_genre_count",
    "commercial_density_rank",
    "neighbor_avg_restaurants",
    "neighbor_avg_population",
    "saturation_index",
    "nearest_station_distance",
    "nearest_station_passengers",
    "station_accessibility",
    "land_price",
    "google_avg_rating",
    "reviews_per_shop",
    "google_density",
]
LOG_TRANSFORM_FEATURES = {
    "other_genre_count",
    "neighbor_avg_restaurants",
    "google_total_reviews",
    "land_price",
    "station_accessibility",
}
CATEGORICAL_FEATURE = "unified_genre"
TARGET_COL = "restaurant_count"
GENRE_ORDER = ["cafe", "chinese", "curry", "italian", "izakaya", "other", "ramen", "washoku", "yakiniku"]
DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 45,
    "learning_rate": 0.075,
    "min_child_samples": 5,
    "feature_fraction": 0.535,
    "bagging_fraction": 0.996,
    "bagging_freq": 3,
    "lambda_l1": 0.786,
    "lambda_l2": 0.796,
    "verbose": -1,
    "seed": 42,
}
DEFAULT_NUM_ROUNDS = 457
DEFAULT_EARLY_STOPPING = 30


def compute_expected_count(df: pd.DataFrame) -> np.ndarray:
    """需要側特徴量のみでlog1p(restaurant_count)の期待値を線形回帰で推定する。"""
    from sklearn.linear_model import Ridge

    work = df.copy()
    target = np.log1p(pd.to_numeric(work[TARGET_COL], errors="coerce").fillna(0))

    feature_cols = [c for c in DEMAND_FEATURES if c in work.columns]
    if not feature_cols:
        return np.full(len(df), target.mean())

    X = work[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    model = Ridge(alpha=1.0)
    model.fit(X, target)
    return model.predict(X)


def prepare_features(df: pd.DataFrame, target_mode: str = "raw") -> tuple[pd.DataFrame, pd.Series]:
    """Prepare model features and log1p target."""
    work = df.copy()
    work[TARGET_COL] = pd.to_numeric(work[TARGET_COL], errors="coerce").fillna(0)
    target = np.log1p(work[TARGET_COL])
    if target_mode == "residual":
        expected = compute_expected_count(work)
        target = target - expected

    feature_cols = []
    for col in NUMERIC_FEATURES:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
            if col in LOG_TRANSFORM_FEATURES:
                work[col] = np.log1p(work[col])
            feature_cols.append(col)

    if CATEGORICAL_FEATURE in work.columns:
        dtype = pd.CategoricalDtype(categories=GENRE_ORDER, ordered=True)
        work["genre_encoded"] = work[CATEGORICAL_FEATURE].astype(dtype).cat.codes
        feature_cols.append("genre_encoded")

    if "land_price" in feature_cols and "saturation_index" in feature_cols:
        work["price_x_saturation"] = work["land_price"] * work["saturation_index"]
        feature_cols.append("price_x_saturation")

    if "population" in feature_cols and "nearest_station_distance" in feature_cols:
        work["pop_x_station_dist"] = work["population"] * work["nearest_station_distance"]
        feature_cols.append("pop_x_station_dist")

    return work[feature_cols].copy(), target


def train_cv(
    df: pd.DataFrame,
    n_splits: int = 5,
    params: dict | None = None,
    num_rounds: int = DEFAULT_NUM_ROUNDS,
    # Optuna検証: フィルタなしが R2 最良 (0.8028)。閾値 pop<4000&rc>200 は実質0件除外
    filter_outliers: bool = False,
    target_mode: str = "raw",
) -> dict:
    """Train with cross validation, using grouped folds when available."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    clean_mask = pd.Series(True, index=df.index)
    if filter_outliers and {"population", "restaurant_count"} <= set(df.columns):
        pop = pd.to_numeric(df["population"], errors="coerce").fillna(0)
        rc = pd.to_numeric(df["restaurant_count"], errors="coerce").fillna(0)
        clean_mask = ~((pop < 2000) & (rc > 100))
        filtered_count = int((~clean_mask).sum())
        if filtered_count > 0:
            logger.info("Filtered %d outlier rows", filtered_count)

    clean_df = df.loc[clean_mask].reset_index(drop=True)
    work = clean_df.sample(frac=1.0, random_state=42).reset_index().rename(columns={"index": "_original_index"})
    features, target = prepare_features(work, target_mode=target_mode)
    logger.info("Training CV model with %d rows and %d features", features.shape[0], features.shape[1])

    # restaurant_count をビン分割して層化
    rc = pd.to_numeric(work[TARGET_COL], errors="coerce").fillna(0)
    bins = pd.qcut(rc, q=min(10, n_splits), labels=False, duplicates="drop")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    shuffled_oof_predictions = np.zeros(len(work))
    shuffled_fold_predictions = np.zeros((len(work), n_splits))
    fold_metrics = []
    feature_importance_list = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(features, bins), 1):
        X_train = features.iloc[train_idx]
        y_train = target.iloc[train_idx]
        X_val = features.iloc[val_idx]
        y_val = target.iloc[val_idx]

        categorical_features = ["genre_encoded"] if "genre_encoded" in X_train.columns else []
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=categorical_features)
        model = lgb.train(
            params,
            train_data,
            num_boost_round=num_rounds,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(DEFAULT_EARLY_STOPPING, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        val_pred = model.predict(X_val)
        full_pred = model.predict(features)
        shuffled_oof_predictions[val_idx] = val_pred
        shuffled_fold_predictions[:, fold_idx - 1] = full_pred

        fold_metrics.append(
            {
                "fold": fold_idx,
                "rmse": root_mean_squared_error(y_val, val_pred),
                "r2": r2_score(y_val, val_pred),
            }
        )
        feature_importance_list.append(
            pd.DataFrame(
                {
                    "feature": features.columns,
                    "importance": model.feature_importance(importance_type="gain"),
                }
            )
        )

    clean_oof_predictions = np.zeros(len(clean_df))
    clean_fold_predictions = np.zeros((len(clean_df), n_splits))
    clean_oof_predictions[work["_original_index"].to_numpy()] = shuffled_oof_predictions
    clean_fold_predictions[work["_original_index"].to_numpy()] = shuffled_fold_predictions

    oof_predictions = np.zeros(len(df))
    fold_predictions = np.zeros((len(df), n_splits))
    oof_predictions[np.flatnonzero(clean_mask.to_numpy())] = clean_oof_predictions
    fold_predictions[np.flatnonzero(clean_mask.to_numpy())] = clean_fold_predictions

    importance_df = (
        pd.concat(feature_importance_list)
        .groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    return {
        "fold_metrics": fold_metrics,
        "avg_rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
        "avg_r2": float(np.mean([m["r2"] for m in fold_metrics])),
        "oof_predictions": oof_predictions,
        "fold_predictions": fold_predictions,
        "feature_importance": importance_df,
        "n_filtered": int((~clean_mask).sum()),
    }


def train_full_model(
    df: pd.DataFrame,
    params: dict | None = None,
    num_rounds: int = DEFAULT_NUM_ROUNDS,
    target_mode: str = "raw",
) -> lgb.Booster:
    """Train a model on the full dataset."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    features, target = prepare_features(df, target_mode=target_mode)
    categorical_features = ["genre_encoded"] if "genre_encoded" in features.columns else []
    model = lgb.train(
        params,
        lgb.Dataset(features, label=target, categorical_feature=categorical_features),
        num_boost_round=num_rounds,
    )
    logger.info("Trained full model with %d rounds", num_rounds)
    return model


def tune_hyperparams(
    df: pd.DataFrame,
    n_trials: int = 50,
    n_splits: int = 5,
) -> dict:
    """Optunaでハイパーパラメータを最適化する。"""
    if optuna is None:
        raise ImportError("optuna is required for hyperparameter tuning")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 10.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
        }
        num_rounds = trial.suggest_int("num_boost_round", 100, 500)
        cv_results = train_cv(
            df,
            n_splits=n_splits,
            params=params,
            num_rounds=num_rounds,
        )
        trial.set_user_attr("avg_r2", cv_results["avg_r2"])
        return float(cv_results["avg_rmse"])

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params.copy()
    best_num_rounds = int(best_params.pop("num_boost_round"))
    best_cv = train_cv(
        df,
        n_splits=n_splits,
        params=best_params,
        num_rounds=best_num_rounds,
    )

    return {
        "best_params": best_params,
        "best_num_rounds": best_num_rounds,
        "best_rmse": float(best_cv["avg_rmse"]),
        "best_r2": float(best_cv["avg_r2"]),
        "study": study,
    }


def compute_market_gap(
    df: pd.DataFrame,
    oof_predictions: np.ndarray,
    target_mode: str = "raw",
    fold_predictions: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute model predicted gap vs actual restaurant count."""
    out = df.copy()
    actual = np.log1p(pd.to_numeric(out[TARGET_COL], errors="coerce").fillna(0).values)
    if target_mode == "residual":
        actual = actual - compute_expected_count(out)
    out["predicted_log_count"] = oof_predictions
    out["actual_log_count"] = actual
    out["market_gap"] = oof_predictions - actual
    if target_mode == "residual":
        expected = compute_expected_count(out)
        predicted_log_count = oof_predictions + expected
        out["predicted_count"] = np.expm1(predicted_log_count)
        out["gap_count"] = out["predicted_count"] - pd.to_numeric(out[TARGET_COL], errors="coerce").fillna(0).values
    else:
        out["predicted_count"] = np.expm1(oof_predictions)
        out["gap_count"] = out["predicted_count"] - pd.to_numeric(out[TARGET_COL], errors="coerce").fillna(0).values

    if fold_predictions is not None:
        gap_std = np.std(fold_predictions, axis=1)
        out["gap_std"] = gap_std
        ci_lower = out["market_gap"].to_numpy() - 1.96 * gap_std
        ci_upper = out["market_gap"].to_numpy() + 1.96 * gap_std
        out["gap_ci_lower"] = ci_lower
        out["gap_ci_upper"] = ci_upper

        reliability = np.full(len(out), "low", dtype=object)
        high_mask = ci_lower > 0
        medium_mask = (out["market_gap"].to_numpy() > 0) & ~high_mask
        reliability[high_mask] = "high"
        reliability[medium_mask] = "medium"
        out["gap_reliability"] = reliability

    return out


def filter_recommendations(
    df: pd.DataFrame,
    min_rc: int = 2,
    min_gap_count: float = 1.0,
    min_reliability: str | None = "medium",
) -> pd.DataFrame:
    """Filter and sort recommendation candidates."""
    out = df.copy()

    if "restaurant_count" in out.columns:
        out = out.loc[pd.to_numeric(out["restaurant_count"], errors="coerce").fillna(0) >= min_rc]
    if "gap_count" in out.columns:
        out = out.loc[pd.to_numeric(out["gap_count"], errors="coerce").fillna(0) >= min_gap_count]

    if min_reliability in {"high", "medium"} and "gap_reliability" in out.columns:
        allowed = {"high"} if min_reliability == "high" else {"high", "medium"}
        out = out.loc[out["gap_reliability"].isin(allowed)]

    if "market_gap" in out.columns:
        out = out.sort_values("market_gap", ascending=False)

    return out


def compute_shap_values(model: lgb.Booster, df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Compute SHAP values for a trained model."""
    import shap

    features, _ = prepare_features(df)
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(features), features


def compare_with_v3(df: pd.DataFrame) -> pd.DataFrame:
    """Compare ML gap ranking with v3 opportunity ranking."""
    if "market_gap" not in df.columns or "opportunity_score" not in df.columns:
        raise KeyError("market_gap and opportunity_score columns are required")

    out = df.copy()
    out["rank_ml"] = out["market_gap"].rank(ascending=False, method="min").astype(int)
    out["rank_v3"] = out["opportunity_score"].rank(ascending=False, method="min").astype(int)
    out["rank_diff"] = out["rank_v3"] - out["rank_ml"]
    return out



def save_model(model: lgb.Booster, tag: str) -> Path:
    """Save a trained model."""
    models_dir = settings.PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{tag}_lightgbm.txt"
    model.save_model(str(path))
    logger.info("Saved model to %s", path)
    return path


def load_model(tag: str) -> lgb.Booster | None:
    """Load a saved model if present."""
    path = settings.PROJECT_ROOT / "models" / f"{tag}_lightgbm.txt"
    if not path.exists():
        return None
    model = lgb.Booster(model_file=str(path))
    logger.info("Loaded model from %s", path)
    return model
