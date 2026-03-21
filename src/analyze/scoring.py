"""Scoring utilities for identifying market gaps."""

from __future__ import annotations

import logging

import pandas as pd

from config.settings import WEIGHT_COMPETITOR, WEIGHT_DEMAND, WEIGHT_LAND_PRICE, WEIGHT_POPULATION

logger = logging.getLogger(__name__)

_COMPETITOR_OFFSET = 0.1


def _normalize(series: pd.Series) -> pd.Series:
    """Normalize numeric values to 0-1."""
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if values.empty:
        return pd.Series(dtype=float, index=series.index)

    min_value = float(values.min())
    max_value = float(values.max())
    if min_value == max_value:
        return pd.Series(0.0, index=series.index, dtype=float)

    return ((values - min_value) / (max_value - min_value)).astype(float)


def _rank_normalize(series: pd.Series) -> pd.Series:
    """Normalize descending rank to 0-1."""
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if values.empty:
        return pd.Series(dtype=float, index=series.index)
    if len(values) == 1:
        return pd.Series(1.0, index=series.index, dtype=float)

    ranks = values.rank(ascending=False, method="average")
    return ((len(values) - ranks) / (len(values) - 1)).astype(float)


def compute_demand_score(df: pd.DataFrame) -> pd.Series:
    """Compute demand score from mesh restaurant counts."""
    logger.info("Computing demand score for %s rows", len(df))

    if df.empty:
        return pd.Series(dtype=float, index=df.index, name="demand_score")

    mesh_total_count = (
        pd.to_numeric(df["restaurant_count"], errors="coerce")
        .fillna(0.0)
        .groupby(df["mesh_code"])
        .transform("sum")
    )
    return _normalize(mesh_total_count).rename("demand_score")


def compute_opportunity_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the baseline opportunity score."""
    logger.info("Computing opportunity score for %s rows", len(df))

    out = df.copy()
    if out.empty:
        out["demand_score"] = pd.Series(dtype=float, index=out.index)
        out["competitor_density"] = pd.Series(dtype=float, index=out.index)
        out["opportunity_score"] = pd.Series(dtype=float, index=out.index)
        return out

    demand_score = compute_demand_score(out)
    competitor_density = pd.to_numeric(out["restaurant_count"], errors="coerce").fillna(0.0)
    opportunity_raw = demand_score / (competitor_density + _COMPETITOR_OFFSET)

    out["demand_score"] = demand_score
    out["competitor_density"] = competitor_density
    out["opportunity_score"] = _normalize(opportunity_raw)
    return out


def rank_opportunities(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Return the top opportunities sorted by score."""
    if "opportunity_score" not in df.columns:
        raise KeyError("opportunity_score column is required")
    if top_n <= 0:
        return df.iloc[0:0].copy()

    logger.info("Ranking top %s opportunities from %s rows", top_n, len(df))
    return df.sort_values("opportunity_score", ascending=False).head(top_n).reset_index(drop=True)


def generate_reason(row: pd.Series) -> str:
    """Generate a simple reason string for one recommendation row."""
    mesh_code = str(row.get("mesh_code", "unknown_mesh"))
    unified_genre = str(row.get("unified_genre", "unknown_genre"))
    opportunity_score = float(pd.to_numeric(pd.Series([row.get("opportunity_score", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
    demand_score = float(pd.to_numeric(pd.Series([row.get("demand_score", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
    competitor_density = int(
        pd.to_numeric(pd.Series([row.get("competitor_density", row.get("restaurant_count", 0))]), errors="coerce")
        .fillna(0)
        .iloc[0]
    )

    if demand_score >= 0.7:
        demand_label = "high"
    elif demand_score >= 0.4:
        demand_label = "medium"
    else:
        demand_label = "low"

    if competitor_density <= 3:
        competitor_label = "low"
    elif competitor_density <= 10:
        competitor_label = "medium"
    else:
        competitor_label = "high"

    return (
        f"{mesh_code} / {unified_genre} "
        f"opportunity {opportunity_score:.3f} / "
        f"demand {demand_label}({demand_score:.2f}) / "
        f"competition {competitor_label}({competitor_density})"
    )


def get_top_recommendations(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Rank rows and attach reason strings."""
    logger.info("Generating top %s recommendations", top_n)

    ranked = rank_opportunities(df, top_n=top_n)
    if ranked.empty:
        ranked["reason"] = pd.Series(dtype=str, index=ranked.index)
        return ranked

    ranked["reason"] = ranked.apply(generate_reason, axis=1)
    return ranked


def run_scoring(aggregated_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Run the baseline scoring pipeline."""
    logger.info("Running scoring pipeline for %s rows", len(aggregated_df))
    return get_top_recommendations(compute_opportunity_score(aggregated_df), top_n=top_n)


def compute_demand_score_v2(df: pd.DataFrame) -> pd.Series:
    """Compute demand score using population when available."""
    logger.info("Computing demand score v2 for %s rows", len(df))

    if df.empty:
        return pd.Series(dtype=float, index=df.index, name="demand_score")

    if "daytime_population" in df.columns:
        demand_raw = pd.to_numeric(df["daytime_population"], errors="coerce").fillna(0.0)
    elif "population" in df.columns:
        demand_raw = pd.to_numeric(df["population"], errors="coerce").fillna(0.0)
    else:
        logger.warning("Population columns missing; falling back to restaurant_count for demand score v2")
        demand_raw = pd.to_numeric(df["restaurant_count"], errors="coerce").fillna(0.0)

    return _normalize(demand_raw).rename("demand_score")


def compute_opportunity_score_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the v2 opportunity score."""
    logger.info("Computing opportunity score v2 for %s rows", len(df))

    out = df.copy()
    if out.empty:
        out["demand_score"] = pd.Series(dtype=float, index=out.index)
        out["competitor_density"] = pd.Series(dtype=float, index=out.index)
        out["opportunity_score"] = pd.Series(dtype=float, index=out.index)
        return out

    demand_score = compute_demand_score_v2(out)
    restaurant_count = pd.to_numeric(out["restaurant_count"], errors="coerce").fillna(0.0)

    if "population" in out.columns:
        population = pd.to_numeric(out["population"], errors="coerce").fillna(0.0)
        competitor_density = restaurant_count / (population / 10000.0 + _COMPETITOR_OFFSET)
        population_signal = _normalize(population)
    else:
        competitor_density = restaurant_count
        if "daytime_population" in out.columns:
            population_signal = _normalize(pd.to_numeric(out["daytime_population"], errors="coerce").fillna(0.0))
        else:
            population_signal = pd.Series(1.0, index=out.index, dtype=float)

    opportunity_raw = (
        demand_score * WEIGHT_DEMAND + population_signal * WEIGHT_POPULATION
    ) / (competitor_density * WEIGHT_COMPETITOR + _COMPETITOR_OFFSET)

    out["demand_score"] = demand_score
    out["competitor_density"] = competitor_density.rename("competitor_density")
    out["opportunity_score"] = _normalize(opportunity_raw)
    return out


def run_scoring_v2(aggregated_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Run the v2 scoring pipeline."""
    logger.info("Running scoring v2 pipeline for %s rows", len(aggregated_df))
    return get_top_recommendations(compute_opportunity_score_v2(aggregated_df), top_n=top_n)


def compute_opportunity_score_v3(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the v3 opportunity score with engineered features."""
    logger.info("Computing opportunity score v3 for %d rows", len(df))

    out = df.copy()
    if out.empty:
        for col in ("demand_score", "competitor_density", "opportunity_score"):
            out[col] = pd.Series(dtype=float, index=out.index)
        return out

    demand_score = compute_demand_score_v2(out)
    restaurant_count = pd.to_numeric(out["restaurant_count"], errors="coerce").fillna(0.0)
    if "population" in out.columns:
        population = pd.to_numeric(out["population"], errors="coerce").fillna(0.0)
        competitor_density = restaurant_count / (population / 10000.0 + _COMPETITOR_OFFSET)
        population_signal = _normalize(population)
    else:
        competitor_density = restaurant_count
        population_signal = pd.Series(1.0, index=out.index, dtype=float)

    diversity = _normalize(pd.to_numeric(out.get("genre_diversity"), errors="coerce").fillna(0.0)) if "genre_diversity" in out.columns else pd.Series(0.0, index=out.index)
    genre_hhi = pd.to_numeric(out.get("genre_hhi"), errors="coerce").fillna(0.0) if "genre_hhi" in out.columns else pd.Series(0.0, index=out.index)
    genre_gap_bonus = _normalize(genre_hhi)
    neighbor_pressure = _normalize(pd.to_numeric(out.get("neighbor_avg_restaurants"), errors="coerce").fillna(0.0)) if "neighbor_avg_restaurants" in out.columns else pd.Series(0.0, index=out.index)
    saturation = _normalize(pd.to_numeric(out.get("saturation_index"), errors="coerce").fillna(0.0)) if "saturation_index" in out.columns else pd.Series(0.0, index=out.index)

    if "nearest_station_distance" in out.columns:
        station_dist = pd.to_numeric(out["nearest_station_distance"], errors="coerce").fillna(0.0)
        station_proximity = _normalize(1.0 / (station_dist + 0.01))
    else:
        station_proximity = pd.Series(0.0, index=out.index)

    if "land_price" in out.columns:
        land_price_norm = _normalize(pd.to_numeric(out["land_price"], errors="coerce").fillna(0.0))
    else:
        land_price_norm = pd.Series(0.0, index=out.index)

    numerator = (
        demand_score * WEIGHT_DEMAND
        + population_signal * WEIGHT_POPULATION
        + diversity * 0.3
        + genre_gap_bonus * 0.2
        + station_proximity * 0.3
    )
    denominator = (
        competitor_density * WEIGHT_COMPETITOR
        + neighbor_pressure * 0.2
        + saturation * 0.3
        + land_price_norm * WEIGHT_LAND_PRICE * 0.2
        + _COMPETITOR_OFFSET
    )

    out["demand_score"] = demand_score
    out["competitor_density"] = competitor_density
    out["opportunity_score"] = _normalize(numerator / denominator)
    return out


def run_scoring_v3(aggregated_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Run the v3 scoring pipeline."""
    logger.info("Running scoring v3 pipeline for %d rows", len(aggregated_df))
    return get_top_recommendations(compute_opportunity_score_v3(aggregated_df), top_n=top_n)


def compute_opportunity_score_v4(df: pd.DataFrame, ml_gap: pd.Series | None = None) -> pd.DataFrame:
    """Blend v3 score rank and ML gap rank into an ensemble score."""
    out = compute_opportunity_score_v3(df)
    if ml_gap is None:
        return out

    ml_gap_series = pd.to_numeric(ml_gap.reindex(out.index), errors="coerce").fillna(0.0)
    v3_rank = _rank_normalize(out["opportunity_score"])
    ml_rank = _rank_normalize(ml_gap_series)

    out["v3_opportunity_score"] = out["opportunity_score"]
    out["market_gap"] = ml_gap_series
    out["v3_rank_score"] = v3_rank
    out["ml_gap_rank_score"] = ml_rank
    out["opportunity_score"] = 0.6 * v3_rank + 0.4 * ml_rank
    return out
