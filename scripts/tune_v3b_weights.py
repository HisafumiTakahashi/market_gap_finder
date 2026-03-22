"""Optimize v3b weights to maximize Spearman correlation with ML gap rankings."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from src.analyze.ml_model import compute_market_gap, prepare_features, train_cv
from src.analyze.scoring import _normalize, _optional_normalized, compute_demand_score_v2
from src.analyze.constants import _COMPETITOR_OFFSET, _POP_UNIT


def load_combined() -> pd.DataFrame:
    tags = ["tokyo", "osaka", "nagoya", "fukuoka"]
    frames = []
    for tag in tags:
        path = settings.PROCESSED_DATA_DIR / f"{tag}_integrated.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def score_v3b_with_weights(df: pd.DataFrame, weights: dict) -> pd.Series:
    """Compute v3b score with given weights (returns raw score, not normalized)."""
    restaurant_count = pd.to_numeric(df.get("restaurant_count"), errors="coerce").fillna(0.0)

    # Genre log gap
    if "unified_genre" in df.columns:
        log_count = np.log1p(restaurant_count)
        genre_log_mean = log_count.groupby(df["unified_genre"]).transform("mean")
        genre_log_gap = genre_log_mean - log_count
    else:
        genre_log_gap = pd.Series(0.0, index=df.index)

    other_genre_norm = _optional_normalized(df, "other_genre_count")
    google_rating_norm = _optional_normalized(df, "google_avg_rating")
    genre_gap_bonus = _optional_normalized(df, "genre_hhi")
    saturation = _optional_normalized(df, "saturation_index")
    neighbor_pressure = _optional_normalized(df, "neighbor_avg_restaurants")

    raw_score = (
        genre_log_gap * weights["genre_base"]
        + other_genre_norm * weights["other_genre"]
        + google_rating_norm * weights["rating"]
        + genre_gap_bonus * weights["genre_gap"]
        - saturation * weights["saturation"]
        - neighbor_pressure * weights["neighbor"]
    )
    return raw_score


def main() -> None:
    print("Loading data...")
    df = load_combined()
    print(f"  {len(df)} rows loaded")

    # Train ML model and get gap predictions
    print("Training ML model...")
    cv_result = train_cv(df)
    gap_df = compute_market_gap(df, cv_result["oof_predictions"])

    # Extract ML rankings
    ml_rank = gap_df["market_gap"].rank(ascending=False, method="min")

    def objective(trial: optuna.Trial) -> float:
        weights = {
            "genre_base": trial.suggest_float("genre_base", 0.1, 5.0),
            "other_genre": trial.suggest_float("other_genre", 0.0, 3.0),
            "rating": trial.suggest_float("rating", 0.0, 2.0),
            "genre_gap": trial.suggest_float("genre_gap", 0.0, 2.0),
            "saturation": trial.suggest_float("saturation", 0.0, 3.0),
            "neighbor": trial.suggest_float("neighbor", 0.0, 2.0),
        }
        raw_score = score_v3b_with_weights(gap_df, weights)
        v3b_rank = raw_score.rank(ascending=False, method="min")
        corr = ml_rank.corr(v3b_rank, method="spearman")
        return corr

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200)

    print(f"\n{'='*60}")
    print(f"Best Spearman: {study.best_value:.4f}")
    print(f"Best weights:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v:.4f}")
    print(f"{'='*60}")

    # Show top 10 trials
    print("\nTop 10 trials:")
    trials = sorted(study.trials, key=lambda t: t.value if t.value else -1, reverse=True)
    for t in trials[:10]:
        print(f"  Spearman={t.value:.4f} | " + " | ".join(f"{k}={v:.3f}" for k, v in t.params.items()))


if __name__ == "__main__":
    main()
