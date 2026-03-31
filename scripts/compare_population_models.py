#!/usr/bin/env python3
"""Compare ML performance across population handling approaches."""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from scripts.integrate_estat import (  # noqa: E402
    _merge_google_ratings,
    aggregate_hotpepper_by_mesh,
    load_or_fetch_population,
    normalize_population_df,
)
from src.analyze.features import add_all_features  # noqa: E402
from src.analyze.ml_model import (  # noqa: E402
    DEFAULT_PARAMS,
    prepare_features,
    train_cv,
    tune_hyperparams,
)
from src.analyze.scoring import compute_opportunity_score_v3b  # noqa: E402
from src.collect.estat import POPULATION_CAT01_CODES as POPULATION_METRICS  # noqa: E402
from src.collect.land_price import load_land_price_cache  # noqa: E402
from src.collect.station import load_station_cache  # noqa: E402
from src.collect.station_passengers import load_passenger_cache  # noqa: E402
from src.preprocess.cleaner import load_hotpepper  # noqa: E402
from src.preprocess.google_matcher import match_google_to_hotpepper  # noqa: E402
from src.preprocess.mesh_converter import mesh_quarter_to_mesh1  # noqa: E402

logger = logging.getLogger(__name__)

ALL_TAGS = ["tokyo", "osaka", "nagoya", "fukuoka"]
NO_POP_COLUMNS = [
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
    "neighbor_avg_population",
    "saturation_index",
    "genre_saturation",
]


@dataclass
class ApproachResult:
    name: str
    cv_rmse: float
    cv_r2: float
    loao_rmse: float
    loao_r2: float
    best_params: dict
    best_num_rounds: int
    feature_importance: pd.DataFrame


def discover_tags() -> list[str]:
    """Return tags that have the required baseline and raw input files."""
    available: list[str] = []
    for tag in ALL_TAGS:
        integrated_path = settings.PROCESSED_DATA_DIR / f"{tag}_integrated.csv"
        hotpepper_path = settings.RAW_DATA_DIR / f"{tag}_hotpepper.csv"
        estat_path = settings.RAW_DATA_DIR / f"{tag}_estat_population.csv"
        if integrated_path.exists() and hotpepper_path.exists() and estat_path.exists():
            available.append(tag)
    return available


def load_integrated_df(tag: str) -> pd.DataFrame:
    path = settings.PROCESSED_DATA_DIR / f"{tag}_integrated.csv"
    logger.info("Loading integrated dataset: %s", path)
    return pd.read_csv(path)


def build_fillna_zero_df(tag: str) -> pd.DataFrame:
    """Rebuild one area's integrated data without fallback fill, using fillna(0)."""
    logger.info("Rebuilding fillna_zero dataset for tag=%s", tag)
    hotpepper_df = load_hotpepper(tag)

    google_path = settings.RAW_DATA_DIR / f"{tag}_google_places.csv"
    google_df: pd.DataFrame | None = None
    if google_path.exists():
        google_df = pd.read_csv(google_path)
        hotpepper_df = match_google_to_hotpepper(hotpepper_df, google_df)

    mesh_agg_df = aggregate_hotpepper_by_mesh(hotpepper_df)

    mesh1_codes = sorted({mesh_quarter_to_mesh1(str(code)) for code in mesh_agg_df["jis_mesh"].dropna()})
    population_raw_df = load_or_fetch_population(tag, mesh1_codes=mesh1_codes, skip_fetch=True)
    population_df, _fallback_residuals = normalize_population_df(population_raw_df)
    del _fallback_residuals

    integrated_df = mesh_agg_df.merge(population_df, on="jis_mesh", how="left")
    for column in POPULATION_METRICS.values():
        if column not in integrated_df.columns:
            integrated_df[column] = 0.0
        integrated_df[column] = pd.to_numeric(integrated_df[column], errors="coerce").fillna(0.0)

    station_df = load_station_cache(tag)
    passenger_df = load_passenger_cache(tag)
    price_df = load_land_price_cache(tag)

    featured_df = add_all_features(
        integrated_df,
        station_df=station_df,
        passenger_df=passenger_df,
        price_df=price_df,
    )

    featured_df = _merge_google_ratings(featured_df, google_df)

    return compute_opportunity_score_v3b(featured_df)


def drop_no_pop_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[col for col in NO_POP_COLUMNS if col in df.columns], errors="ignore").copy()


@contextmanager
def patched_ml_features():
    """Temporarily remove population-derived columns from ml_model feature lists."""
    import src.analyze.ml_model as ml_model

    original_numeric = list(ml_model.NUMERIC_FEATURES)
    original_demand = list(ml_model.DEMAND_FEATURES)
    ml_model.NUMERIC_FEATURES = [col for col in original_numeric if col not in NO_POP_COLUMNS]
    ml_model.DEMAND_FEATURES = [col for col in original_demand if col not in NO_POP_COLUMNS]
    try:
        yield
    finally:
        ml_model.NUMERIC_FEATURES = original_numeric
        ml_model.DEMAND_FEATURES = original_demand


def concat_frames(data_by_tag: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat([data_by_tag[tag] for tag in data_by_tag], ignore_index=True)


def train_single_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: dict,
    num_rounds: int,
) -> tuple[np.ndarray, pd.Series, pd.DataFrame]:
    """Train on train_df and predict test_df using existing feature preparation."""
    train_features, train_target = prepare_features(train_df)
    test_features, test_target = prepare_features(test_df)
    categorical_features = ["genre_encoded"] if "genre_encoded" in train_features.columns else []
    model = lgb.train(
        {**DEFAULT_PARAMS, **params},
        lgb.Dataset(train_features, label=train_target, categorical_feature=categorical_features),
        num_boost_round=num_rounds,
    )
    predictions = model.predict(test_features)
    importance_df = (
        pd.DataFrame(
            {
                "feature": train_features.columns,
                "importance": model.feature_importance(importance_type="gain"),
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return predictions, test_target, importance_df


def evaluate_loao(
    data_by_tag: dict[str, pd.DataFrame],
    params: dict,
    num_rounds: int,
    no_pop: bool = False,
) -> tuple[float, float]:
    """Evaluate leave-one-area-out performance."""
    if len(data_by_tag) < 2:
        raise ValueError("LOAO requires at least 2 tags")

    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []

    for test_tag in data_by_tag:
        train_frames = [data_by_tag[tag] for tag in data_by_tag if tag != test_tag]
        train_df = pd.concat(train_frames, ignore_index=True)
        test_df = data_by_tag[test_tag].reset_index(drop=True)
        logger.info(
            "LOAO split: test=%s train_rows=%d test_rows=%d",
            test_tag,
            len(train_df),
            len(test_df),
        )
        if no_pop:
            with patched_ml_features():
                preds, target, _ = train_single_split(train_df, test_df, params=params, num_rounds=num_rounds)
        else:
            preds, target, _ = train_single_split(train_df, test_df, params=params, num_rounds=num_rounds)
        all_predictions.append(preds)
        all_targets.append(target.to_numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_predictions)
    return float(root_mean_squared_error(y_true, y_pred)), float(r2_score(y_true, y_pred))


def train_full_importance(
    df: pd.DataFrame,
    params: dict,
    num_rounds: int,
    no_pop: bool = False,
) -> pd.DataFrame:
    if no_pop:
        with patched_ml_features():
            _, _, importance_df = train_single_split(df, df, params=params, num_rounds=num_rounds)
            return importance_df
    _, _, importance_df = train_single_split(df, df, params=params, num_rounds=num_rounds)
    return importance_df


def evaluate_approach(name: str, data_by_tag: dict[str, pd.DataFrame], no_pop: bool = False) -> ApproachResult:
    combined_df = concat_frames(data_by_tag)
    logger.info("Evaluating approach=%s rows=%d", name, len(combined_df))

    if no_pop:
        with patched_ml_features():
            tuning = tune_hyperparams(combined_df, n_trials=50, n_splits=5)
            cv_result = train_cv(
                combined_df,
                n_splits=5,
                params=tuning["best_params"],
                num_rounds=tuning["best_num_rounds"],
            )
    else:
        tuning = tune_hyperparams(combined_df, n_trials=50, n_splits=5)
        cv_result = train_cv(
            combined_df,
            n_splits=5,
            params=tuning["best_params"],
            num_rounds=tuning["best_num_rounds"],
        )

    loao_rmse, loao_r2 = evaluate_loao(
        data_by_tag,
        params=tuning["best_params"],
        num_rounds=tuning["best_num_rounds"],
        no_pop=no_pop,
    )
    importance_df = train_full_importance(
        combined_df,
        params=tuning["best_params"],
        num_rounds=tuning["best_num_rounds"],
        no_pop=no_pop,
    )

    return ApproachResult(
        name=name,
        cv_rmse=float(cv_result["avg_rmse"]),
        cv_r2=float(cv_result["avg_r2"]),
        loao_rmse=loao_rmse,
        loao_r2=loao_r2,
        best_params=tuning["best_params"],
        best_num_rounds=int(tuning["best_num_rounds"]),
        feature_importance=importance_df,
    )


def print_results(results: list[ApproachResult]) -> None:
    print("=" * 60)
    print("Model Comparison Results")
    print("=" * 60)
    print()
    print("Approach        | CV RMSE  | CV R2   | LOAO RMSE | LOAO R2")
    print("----------------|----------|---------|-----------|--------")
    for result in results:
        print(
            f"{result.name:<15} | "
            f"{result.cv_rmse:>7.4f} | "
            f"{result.cv_r2:>7.4f} | "
            f"{result.loao_rmse:>9.4f} | "
            f"{result.loao_r2:>7.4f}"
        )

    best = min(results, key=lambda item: item.loao_rmse)
    print()
    print(f"Best approach: {best.name}")
    print()

    for result in results:
        print(f"Top 10 Feature Importances: {result.name}")
        top10 = result.feature_importance.head(10)
        for row in top10.itertuples(index=False):
            print(f"  {row.feature:<24} {float(row.importance):.4f}")
        print()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tags = discover_tags()
    if not tags:
        logger.error("No valid tags found under %s and %s", settings.PROCESSED_DATA_DIR, settings.RAW_DATA_DIR)
        return 1

    logger.info("Using tags: %s", ", ".join(tags))

    baseline_by_tag = {tag: load_integrated_df(tag) for tag in tags}
    fillna_zero_by_tag = {tag: build_fillna_zero_df(tag) for tag in tags}
    no_pop_by_tag = {tag: drop_no_pop_columns(df) for tag, df in baseline_by_tag.items()}

    results = [
        evaluate_approach("baseline", baseline_by_tag),
        evaluate_approach("fillna_zero", fillna_zero_by_tag),
        evaluate_approach("no_pop", no_pop_by_tag, no_pop=True),
    ]
    print_results(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
