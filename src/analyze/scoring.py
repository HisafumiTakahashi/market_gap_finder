"""Scoring utilities for identifying market gaps."""

from __future__ import annotations

import logging

import pandas as pd

from src.analyze.constants import _COMPETITOR_OFFSET, _POP_UNIT  # noqa: F401 - re-export
from src.analyze.utils import mesh_col as _mesh_col
from config.settings import (
    V3B_W_DIVERSITY,
    V3B_W_GENRE_BASE,
    V3B_W_GENRE_GAP,
    V3B_W_LAND_PRICE,
    V3B_W_NEIGHBOR,
    V3B_W_OTHER_GENRE,
    V3B_W_RATING,
    V3B_W_SATURATION,
    V3B_W_STATION,
    WEIGHT_COMPETITOR,
    WEIGHT_DEMAND,
    WEIGHT_LAND_PRICE,
    WEIGHT_POPULATION,
)

logger = logging.getLogger(__name__)


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


def _optional_normalized(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a normalized numeric series, or zeros when the column is absent."""
    if column not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    return _normalize(pd.to_numeric(df[column], errors="coerce").fillna(0.0))


def compute_demand_score(df: pd.DataFrame) -> pd.Series:
    """Compute demand score from mesh restaurant counts."""
    logger.info("Computing demand score for %s rows", len(df))

    if df.empty:
        return pd.Series(dtype=float, index=df.index, name="demand_score")

    mesh_col = _mesh_col(df)
    if mesh_col not in df.columns:
        return _normalize(pd.to_numeric(df["restaurant_count"], errors="coerce").fillna(0.0)).rename("demand_score")
    mesh_total_count = (
        pd.to_numeric(df["restaurant_count"], errors="coerce")
        .fillna(0.0)
        .groupby(df[mesh_col])
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


_GENRE_JA: dict[str, str] = {
    "izakaya": "居酒屋",
    "italian": "イタリアン",
    "chinese": "中華",
    "yakiniku": "焼肉",
    "cafe": "カフェ",
    "ramen": "ラーメン",
    "washoku": "和食",
    "curry": "カレー",
    "other": "その他",
}


def generate_reason(row: pd.Series) -> str:
    """Generate a human-readable Japanese reason string for one recommendation row."""
    restaurant_count = int(
        pd.to_numeric(pd.Series([row.get("restaurant_count", 0)]), errors="coerce")
        .fillna(0)
        .iloc[0]
    )
    unified_genre_raw = str(row.get("unified_genre", "") or "")
    unified_genre = _GENRE_JA.get(unified_genre_raw, unified_genre_raw)
    population = int(pd.to_numeric(pd.Series([row.get("population", 0)]), errors="coerce").fillna(0).iloc[0])
    gap = float(pd.to_numeric(pd.Series([row.get("market_gap", 0)]), errors="coerce").fillna(0.0).iloc[0])
    station_name = str(row.get("nearest_station_name", "") or "")
    station_dist = float(pd.to_numeric(pd.Series([row.get("nearest_station_distance", 0)]), errors="coerce").fillna(0.0).iloc[0])
    saturation = float(pd.to_numeric(pd.Series([row.get("saturation_index", 0)]), errors="coerce").fillna(0.0).iloc[0])

    parts: list[str] = []

    # 競合状況
    if unified_genre:
        if restaurant_count <= 1:
            parts.append(f"{unified_genre}の競合がほぼなし（{restaurant_count}店舗）")
        elif restaurant_count <= 3:
            parts.append(f"{unified_genre}の競合が少ない（{restaurant_count}店舗）")
        elif restaurant_count <= 10:
            parts.append(f"{unified_genre}は{restaurant_count}店舗で中程度の競合")
        else:
            parts.append(f"{unified_genre}は{restaurant_count}店舗で競合多い")

    # 人口
    if population > 0:
        parts.append(f"商圏人口{population:,}人")

    # 飽和度
    if saturation > 0:
        if saturation < 5:
            parts.append("飽和度が低く参入余地あり")
        elif saturation > 20:
            parts.append("飽和度が高い")

    # 駅アクセス
    if station_name and station_dist > 0:
        if station_dist <= 0.3:
            parts.append(f"{station_name}駅から{station_dist * 1000:.0f}m（駅近）")
        elif station_dist <= 0.8:
            parts.append(f"{station_name}駅から{station_dist * 1000:.0f}m（徒歩圏）")

    # MLギャップ
    if gap > 1.0:
        parts.append(f"ML予測で大きな出店余地（gap={gap:+.2f}）")
    elif gap > 0.5:
        parts.append(f"ML予測で出店余地あり（gap={gap:+.2f}）")
    elif gap < -0.5:
        parts.append(f"ML予測で供給過多の可能性（gap={gap:+.2f}）")

    if not parts:
        return "詳細データ要確認"
    return "。".join(parts)


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
        competitor_density = restaurant_count / (population / _POP_UNIT + _COMPETITOR_OFFSET)
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
        competitor_density = restaurant_count / (population / _POP_UNIT + _COMPETITOR_OFFSET)
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


def compute_opportunity_score_v3b(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the v3b opportunity score using additive log-space genre gap.

    Mirrors the ML model structure: genre-specific baseline in log space,
    additive demand/penalty signals, no ratio form.
    """
    logger.info("Computing opportunity score v3b for %d rows", len(df))

    out = df.copy()
    if out.empty:
        for col in ("demand_score", "competitor_density", "opportunity_score"):
            out[col] = pd.Series(dtype=float, index=out.index)
        return out

    restaurant_count = pd.to_numeric(out.get("restaurant_count"), errors="coerce").fillna(0.0)
    if "population" in out.columns:
        population = pd.to_numeric(out["population"], errors="coerce").fillna(0.0)
        competitor_density = restaurant_count / (population / _POP_UNIT + _COMPETITOR_OFFSET)
    else:
        competitor_density = restaurant_count

    demand_score = compute_demand_score_v2(out)

    # --- Genre gap in log space (mirrors ML genre_encoded) ---
    if "unified_genre" in out.columns and "restaurant_count" in out.columns:
        import numpy as np
        log_count = np.log1p(restaurant_count)
        genre_log_mean = log_count.groupby(out["unified_genre"]).transform("mean")
        # positive = fewer than genre average = opportunity
        genre_log_gap = genre_log_mean - log_count
    else:
        genre_log_gap = pd.Series(0.0, index=out.index, dtype=float)

    # --- Demand signals (positive = opportunity) ---
    other_genre_norm = _optional_normalized(out, "other_genre_count")
    google_rating_norm = _optional_normalized(out, "google_avg_rating")
    genre_gap_bonus = _optional_normalized(out, "genre_hhi")

    # --- Penalty signals (negative = barrier) ---
    saturation = _optional_normalized(out, "saturation_index")
    neighbor_pressure = _optional_normalized(out, "neighbor_avg_restaurants")

    # --- Additive score (not ratio) ---
    raw_score = (
        genre_log_gap * V3B_W_GENRE_BASE
        + other_genre_norm * V3B_W_OTHER_GENRE
        + google_rating_norm * V3B_W_RATING
        + genre_gap_bonus * V3B_W_GENRE_GAP
        - saturation * V3B_W_SATURATION
        - neighbor_pressure * V3B_W_NEIGHBOR
    )

    out["demand_score"] = demand_score
    out["competitor_density"] = competitor_density
    out["opportunity_score"] = _normalize(raw_score)
    return out


def run_scoring_v3b(aggregated_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Run the v3b scoring pipeline."""
    logger.info("Running scoring v3b pipeline for %d rows", len(aggregated_df))
    return get_top_recommendations(compute_opportunity_score_v3b(aggregated_df), top_n=top_n)


def compute_opportunity_score_v4(df: pd.DataFrame, ml_gap: pd.Series | None = None) -> pd.DataFrame:
    """Blend v3b score rank and ML gap rank into an ensemble score."""
    out = compute_opportunity_score_v3b(df)
    if ml_gap is None:
        return out

    ml_gap_series = pd.to_numeric(ml_gap.reindex(out.index), errors="coerce").fillna(0.0)
    v3b_rank = _rank_normalize(out["opportunity_score"])
    ml_rank = _rank_normalize(ml_gap_series)

    out["v3b_opportunity_score"] = out["opportunity_score"]
    out["market_gap"] = ml_gap_series
    out["v3b_rank_score"] = v3b_rank
    out["ml_gap_rank_score"] = ml_rank
    out["opportunity_score"] = 0.6 * v3b_rank + 0.4 * ml_rank
    return out
