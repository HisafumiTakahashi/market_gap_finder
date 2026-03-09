from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_COMPETITOR_OFFSET = 0.1


def _normalize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if values.empty:
        return pd.Series(dtype=float, index=series.index)

    min_value = float(values.min())
    max_value = float(values.max())
    if min_value == max_value:
        return pd.Series(0.0, index=series.index, dtype=float)

    return ((values - min_value) / (max_value - min_value)).astype(float)


def compute_demand_score(df: pd.DataFrame) -> pd.Series:
    logger.info("Computing demand score for %s rows", len(df))

    if df.empty:
        return pd.Series(dtype=float, index=df.index, name="demand_score")

    mesh_total_count = (
        pd.to_numeric(df["restaurant_count"], errors="coerce")
        .fillna(0.0)
        .groupby(df["mesh_code"])
        .transform("sum")
    )
    demand_raw = mesh_total_count

    return _normalize(demand_raw).rename("demand_score")


def compute_opportunity_score(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing opportunity score for %s rows", len(df))

    out = df.copy()
    if out.empty:
        out["demand_score"] = pd.Series(dtype=float, index=out.index)
        out["competitor_density"] = pd.Series(dtype=float, index=out.index)
        out["opportunity_score"] = pd.Series(dtype=float, index=out.index)
        return out

    demand_score = compute_demand_score(out)
    competitor_density = pd.to_numeric(
        out["restaurant_count"], errors="coerce"
    ).fillna(0.0)
    opportunity_raw = demand_score / (competitor_density + _COMPETITOR_OFFSET)

    out["demand_score"] = demand_score
    out["competitor_density"] = competitor_density
    out["opportunity_score"] = _normalize(opportunity_raw)
    return out


def rank_opportunities(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    if "opportunity_score" not in df.columns:
        raise KeyError("opportunity_score column is required")
    if top_n <= 0:
        return df.iloc[0:0].copy()

    logger.info("Ranking top %s opportunities from %s rows", top_n, len(df))
    return (
        df.sort_values("opportunity_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def generate_reason(row: pd.Series) -> str:
    mesh_code = str(row.get("mesh_code", "unknown_mesh"))
    unified_genre = str(row.get("unified_genre", "unknown_genre"))
    opportunity_score = float(
        pd.to_numeric(pd.Series([row.get("opportunity_score", 0.0)]), errors="coerce")
        .fillna(0.0)
        .iloc[0]
    )
    demand_score = float(
        pd.to_numeric(pd.Series([row.get("demand_score", 0.0)]), errors="coerce")
        .fillna(0.0)
        .iloc[0]
    )
    competitor_density = int(
        pd.to_numeric(
            pd.Series([row.get("competitor_density", row.get("restaurant_count", 0))]),
            errors="coerce",
        )
        .fillna(0)
        .iloc[0]
    )

    if demand_score >= 0.7:
        demand_label = "\u9ad8\u3044"
    elif demand_score >= 0.4:
        demand_label = "\u4e2d\u7a0b\u5ea6"
    else:
        demand_label = "\u4f4e\u3044"

    if competitor_density <= 3:
        competitor_label = "\u5c11\u306a\u3044"
    elif competitor_density <= 10:
        competitor_label = "\u4e2d\u7a0b\u5ea6"
    else:
        competitor_label = "\u591a\u3044"

    return (
        f"\u3010{mesh_code} \u00d7 {unified_genre}\u3011"
        f"\u6a5f\u4f1a\u30b9\u30b3\u30a2 {opportunity_score:.3f} / "
        f"\u9700\u8981: {demand_label}({demand_score:.2f}) / "
        f"\u7af6\u5408: {competitor_label}({competitor_density}\u5e97)"
    )


def get_top_recommendations(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    logger.info("Generating top %s recommendations", top_n)

    ranked = rank_opportunities(df, top_n=top_n)
    if ranked.empty:
        ranked["reason"] = pd.Series(dtype=str, index=ranked.index)
        return ranked

    ranked["reason"] = ranked.apply(generate_reason, axis=1)
    return ranked


def run_scoring(aggregated_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    logger.info("Running scoring pipeline for %s rows", len(aggregated_df))
    scored_df = compute_opportunity_score(aggregated_df)
    return get_top_recommendations(scored_df, top_n=top_n)
