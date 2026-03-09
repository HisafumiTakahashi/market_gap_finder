from __future__ import annotations

import logging

import pandas as pd

from src.analyze.scoring import compute_opportunity_score
from src.preprocess.cleaner import assign_mesh_code, map_genre

logger = logging.getLogger(__name__)

_DATE_COLUMN_CANDIDATES = ("opening_date", "open_date", "date", "opened_at")
_RATING_COLUMN_CANDIDATES = ("rating", "avg_rating")
_REVIEW_COLUMN_CANDIDATES = ("total_reviews", "review_count", "user_ratings_total", "reviews")


def _find_first_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    return next((column for column in candidates if column in df.columns), None)


def _normalize_lat_lng(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()

    if "lat" not in normalized.columns and "latitude" in normalized.columns:
        normalized = normalized.rename(columns={"latitude": "lat"})

    if "lng" not in normalized.columns:
        if "longitude" in normalized.columns:
            normalized = normalized.rename(columns={"longitude": "lng"})
        elif "lon" in normalized.columns:
            normalized = normalized.rename(columns={"lon": "lng"})

    return normalized


def _get_numeric_series(
    df: pd.DataFrame,
    candidates: tuple[str, ...],
    default: float = 0.0,
) -> pd.Series:
    column = _find_first_column(df, candidates)
    if column is None:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def _aggregate_training_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "mesh_code",
                "unified_genre",
                "restaurant_count",
                "avg_rating",
                "total_reviews",
                "lat",
                "lng",
            ]
        )

    aggregated_input = df.copy()
    aggregated_input["avg_rating_source"] = _get_numeric_series(
        aggregated_input,
        _RATING_COLUMN_CANDIDATES,
        default=0.0,
    )
    aggregated_input["total_reviews_source"] = _get_numeric_series(
        aggregated_input,
        _REVIEW_COLUMN_CANDIDATES,
        default=0.0,
    )
    aggregated_input["lat"] = pd.to_numeric(aggregated_input["lat"], errors="coerce")
    aggregated_input["lng"] = pd.to_numeric(aggregated_input["lng"], errors="coerce")

    aggregated = (
        aggregated_input.groupby(["mesh_code", "unified_genre"], dropna=False)
        .agg(
            restaurant_count=("mesh_code", "size"),
            avg_rating=("avg_rating_source", "mean"),
            total_reviews=("total_reviews_source", "sum"),
            lat=("lat", "mean"),
            lng=("lng", "mean"),
        )
        .reset_index()
    )

    return aggregated


def load_historical_openings(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    date_column = _find_first_column(df, _DATE_COLUMN_CANDIDATES)
    if date_column is None:
        raise KeyError("opening_date/open_date/date/opened_at column is required")

    loaded = _normalize_lat_lng(df)
    loaded["opening_date"] = pd.to_datetime(loaded[date_column], errors="coerce")
    loaded = loaded.dropna(subset=["opening_date"]).reset_index(drop=True)

    logger.info("Loaded %s historical rows after dropping invalid dates", len(loaded))

    loaded = map_genre(loaded)
    loaded = assign_mesh_code(loaded)
    return loaded


def simulate_scoring_at_date(target_date: str, df_aggregated: pd.DataFrame) -> pd.DataFrame:
    target_ts = pd.to_datetime(target_date).normalize()
    scoring_input = df_aggregated.copy()

    if "snapshot_date" in scoring_input.columns:
        scoring_input["snapshot_date"] = pd.to_datetime(
            scoring_input["snapshot_date"],
            errors="coerce",
        )
        scoring_input = scoring_input[scoring_input["snapshot_date"] <= target_ts].copy()

    scored = compute_opportunity_score(scoring_input)
    scored["target_date"] = target_ts
    return scored


def evaluate_accuracy(scored_df: pd.DataFrame, actuals_df: pd.DataFrame) -> dict[str, float]:
    if scored_df.empty:
        return {
            "correlation": 0.0,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "hit_count": 0.0,
            "k": 0.0,
        }

    actual_counts = (
        actuals_df.groupby(["mesh_code", "unified_genre"], dropna=False)
        .size()
        .rename("actual_openings")
        .reset_index()
    )

    merged = scored_df.merge(
        actual_counts,
        on=["mesh_code", "unified_genre"],
        how="left",
    )
    merged["actual_openings"] = pd.to_numeric(
        merged["actual_openings"],
        errors="coerce",
    ).fillna(0.0)

    score_series = pd.to_numeric(merged["opportunity_score"], errors="coerce").fillna(0.0)
    actual_series = merged["actual_openings"]

    correlation = 0.0
    if score_series.nunique(dropna=False) > 1 and actual_series.nunique(dropna=False) > 1:
        correlation_value = score_series.corr(actual_series)
        correlation = 0.0 if pd.isna(correlation_value) else float(correlation_value)

    k = min(20, len(scored_df))
    if k == 0:
        return {
            "correlation": correlation,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "hit_count": 0.0,
            "k": 0.0,
        }

    top_k = merged.nlargest(k, "opportunity_score")
    hit_count = float((top_k["actual_openings"] > 0).sum())
    positive_count = float((merged["actual_openings"] > 0).sum())

    precision_at_k = hit_count / float(k)
    recall_at_k = hit_count / positive_count if positive_count > 0 else 0.0

    return {
        "correlation": correlation,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "hit_count": hit_count,
        "k": float(k),
    }


def run_backtest(
    historical_filepath: str,
    test_dates: list[str] | None = None,
) -> pd.DataFrame:
    historical = load_historical_openings(historical_filepath)
    result_columns = [
        "test_date",
        "correlation",
        "precision_at_k",
        "recall_at_k",
        "hit_count",
        "k",
        "train_rows",
        "scored_rows",
    ]
    if historical.empty:
        return pd.DataFrame(columns=result_columns)

    if test_dates is None:
        normalized_dates = pd.to_datetime(historical["opening_date"], errors="coerce").dt.normalize()
        test_dates = [timestamp.strftime("%Y-%m-%d") for timestamp in sorted(normalized_dates.dropna().unique())]

    results: list[dict[str, float | str | int]] = []

    for test_date in test_dates:
        target_ts = pd.to_datetime(test_date).normalize()
        train_df = historical[historical["opening_date"] < target_ts].copy()

        aggregated_train = _aggregate_training_data(train_df)
        scored_df = simulate_scoring_at_date(
            target_date=target_ts.strftime("%Y-%m-%d"),
            df_aggregated=aggregated_train,
        )

        window_end = target_ts + pd.Timedelta(days=30)
        actuals_df = historical[
            (historical["opening_date"] >= target_ts) & (historical["opening_date"] < window_end)
        ].copy()

        metrics = evaluate_accuracy(scored_df=scored_df, actuals_df=actuals_df)
        metrics["test_date"] = target_ts.strftime("%Y-%m-%d")
        metrics["train_rows"] = int(len(train_df))
        metrics["scored_rows"] = int(len(scored_df))
        results.append(metrics)

        logger.info(
            "Backtest %s: train_rows=%s scored_rows=%s actual_rows=%s",
            target_ts.strftime("%Y-%m-%d"),
            len(train_df),
            len(scored_df),
            len(actuals_df),
        )

    if not results:
        return pd.DataFrame(columns=result_columns)

    return pd.DataFrame(results, columns=result_columns)
