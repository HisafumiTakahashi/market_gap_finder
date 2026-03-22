#!/usr/bin/env python3
"""e-Stat integration pipeline CLI."""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from config import settings
from src.analyze.features import add_all_features
from src.analyze.constants import _COMPETITOR_OFFSET, _POP_UNIT
from src.analyze.scoring import (
    compute_opportunity_score_v3,
    generate_reason,
    run_scoring_v3,
)
from src.collect.estat import (
    POPULATION_CAT01_ALIASES,
    POPULATION_CAT01_CODES as POPULATION_METRICS,
    fetch_mesh_population,
    save_raw,
)
from src.collect.land_price import load_land_price_cache
from src.collect.station import load_station_cache
from src.collect.station_passengers import load_passenger_cache
from src.preprocess.cleaner import load_hotpepper, map_genre
from src.preprocess.google_matcher import match_google_to_hotpepper
from src.preprocess.mesh_converter import (
    assign_jis_mesh_quarter,
    lat_lon_to_mesh_quarter,
    mesh_quarter_to_mesh1,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="e-Stat mesh population integration pipeline.")
    parser.add_argument("--tag", type=str, default="tokyo", help="Area tag.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top rows to display.")
    parser.add_argument(
        "--stats-id",
        type=str,
        default="0003412636",
        help="e-Stat population dataset ID.",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip e-Stat API fetch and reuse cached CSV.",
    )
    return parser.parse_args()


def aggregate_hotpepper_by_mesh(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate HotPepper rows by JIS quarter mesh and genre."""
    prepared = assign_jis_mesh_quarter(map_genre(df))
    prepared["lat"] = pd.to_numeric(prepared.get("lat"), errors="coerce")
    prepared["lng"] = pd.to_numeric(prepared.get("lng"), errors="coerce")
    prepared = prepared.dropna(subset=["jis_mesh"]).copy()
    has_google_cols = {"google_rating", "google_review_count"} <= set(prepared.columns)

    if has_google_cols:
        prepared["google_rating"] = pd.to_numeric(prepared["google_rating"], errors="coerce")
        prepared["google_review_count"] = pd.to_numeric(prepared["google_review_count"], errors="coerce")
        aggregated = (
            prepared.groupby(["jis_mesh", "unified_genre"], dropna=False)
            .agg(
                restaurant_count=("id", "size"),
                lat=("lat", "mean"),
                lng=("lng", "mean"),
                google_avg_rating=("google_rating", "mean"),
                google_total_reviews=("google_review_count", "sum"),
                google_match_count=("google_rating", lambda s: int(s.notna().sum())),
            )
            .reset_index()
        )
    else:
        aggregated = (
            prepared.groupby(["jis_mesh", "unified_genre"], dropna=False)
            .agg(
                restaurant_count=("id", "size"),
                lat=("lat", "mean"),
                lng=("lng", "mean"),
            )
            .reset_index()
        )

    aggregated["mesh_code"] = aggregated["jis_mesh"]
    return aggregated


def normalize_population_df(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalize e-Stat population and household data to quarter-mesh totals."""
    base_columns = ["jis_mesh", *POPULATION_METRICS.values()]
    if df.empty:
        return pd.DataFrame(columns=base_columns), {}

    normalized = df.copy()
    if "mesh_code" not in normalized.columns:
        raise KeyError("mesh_code column is required in e-Stat data")

    normalized["jis_mesh"] = normalized["mesh_code"].astype(str).str.extract(r"(\d{8,10})", expand=False)
    value_col = "population" if "population" in normalized.columns else "value"
    if value_col not in normalized.columns:
        return pd.DataFrame(columns=base_columns), {}

    normalized[value_col] = (
        normalized[value_col].astype(str).str.replace(",", "", regex=False).pipe(pd.to_numeric, errors="coerce")
    )

    if "cat01" not in normalized.columns:
        normalized["cat01"] = "0010"

    normalized["cat01"] = normalized["cat01"].astype(str).replace(POPULATION_CAT01_ALIASES)
    normalized = normalized.loc[normalized["cat01"].isin(POPULATION_METRICS)].copy()
    normalized = normalized.dropna(subset=["jis_mesh", value_col])
    if normalized.empty:
        return pd.DataFrame(columns=base_columns), {}

    quarter = normalized[normalized["jis_mesh"].astype(str).str.len() == 10].copy()
    if quarter.empty:
        return pd.DataFrame(columns=base_columns), {}

    quarter = quarter.assign(metric=quarter["cat01"].map(POPULATION_METRICS), metric_value=quarter[value_col])
    quarter_pop = (
        quarter.pivot_table(
            index="jis_mesh",
            columns="metric",
            values="metric_value",
            aggfunc="sum",
        )
        .reset_index()
    )

    for column in POPULATION_METRICS.values():
        if column not in quarter_pop.columns:
            quarter_pop[column] = pd.NA

    fallback_by_metric: dict[str, dict[str, float]] = {}
    quarter_pop["_mesh3"] = quarter_pop["jis_mesh"].astype(str).str[:8]
    for column in POPULATION_METRICS.values():
        quarter_pop[column] = pd.to_numeric(quarter_pop[column], errors="coerce")
        fallback_by_metric[column] = quarter_pop.groupby("_mesh3")[column].mean().to_dict()
    quarter_pop = quarter_pop.drop(columns=["_mesh3"])

    return quarter_pop[base_columns].sort_values("jis_mesh").reset_index(drop=True), fallback_by_metric


def load_or_fetch_population(tag: str, mesh1_codes: list[str], skip_fetch: bool) -> pd.DataFrame:
    """Load cached e-Stat data or fetch it."""
    cache_name = f"{tag}_estat_population.csv"
    cache_path = settings.RAW_DATA_DIR / cache_name

    if skip_fetch:
        if not cache_path.exists():
            raise FileNotFoundError(cache_path)
        logger.info("Loading cached e-Stat population data: %s", cache_path)
        return pd.read_csv(cache_path, dtype={"cat01": str, "mesh_code": str})

    logger.info("Fetching e-Stat population data for %d mesh1 codes", len(mesh1_codes))
    population_df = fetch_mesh_population(mesh1_codes=mesh1_codes)
    save_raw(population_df, cache_name)
    return population_df


def save_integrated(df: pd.DataFrame, tag: str) -> None:
    """Save integrated output data."""
    settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = settings.PROCESSED_DATA_DIR / f"{tag}_integrated.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("Saved integrated data: %s", output_path)


def _merge_google_ratings(integrated_df: pd.DataFrame, google_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute derived Google aggregate features from pre-matched mesh data and raw Google data."""
    out = integrated_df.copy()
    for col, default in (("google_avg_rating", 0.0), ("google_total_reviews", 0.0), ("google_match_count", 0)):
        if col not in out.columns:
            out[col] = default
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default)
    out["google_match_count"] = out["google_match_count"].astype(int)
    out["reviews_per_shop"] = out["google_total_reviews"] / out["google_match_count"].replace(0, 1)

    population = pd.to_numeric(out.get("population"), errors="coerce").fillna(0.0)
    if google_df is not None and not google_df.empty and "jis_mesh" in out.columns:
        gdf = google_df.copy()
        gdf["lat"] = pd.to_numeric(gdf["lat"], errors="coerce")
        gdf["lng"] = pd.to_numeric(gdf["lng"], errors="coerce")
        gdf = gdf.dropna(subset=["lat", "lng"])
        gdf["jis_mesh"] = [lat_lon_to_mesh_quarter(lat, lng) for lat, lng in zip(gdf["lat"], gdf["lng"])]
        mesh_gp_count = gdf.groupby("jis_mesh").size().rename("google_place_count")
        out = out.merge(mesh_gp_count, on="jis_mesh", how="left")
        out["google_place_count"] = out["google_place_count"].fillna(0).astype(int)
        out["google_density"] = out["google_place_count"] / (population / _POP_UNIT + _COMPETITOR_OFFSET)
        out["google_density"] = out["google_density"].fillna(0.0)
    else:
        out["google_density"] = 0.0

    return out


def main() -> int:
    """Run the e-Stat integration pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    try:
        logger.info("Loading HotPepper data for tag=%s", args.tag)
        hotpepper_df = load_hotpepper(args.tag)

        google_path = settings.RAW_DATA_DIR / f"{args.tag}_google_places.csv"
        google_df: pd.DataFrame | None = None
        google_matched_df = hotpepper_df
        if google_path.exists():
            google_df = pd.read_csv(google_path)
            logger.info("Loaded Google Places rows: %d", len(google_df))
            google_matched_df = match_google_to_hotpepper(hotpepper_df, google_df)
            matched_count = int(google_matched_df["google_rating"].notna().sum())
            logger.info(
                "Google Places matched: %d/%d shops (%.1f%%)",
                matched_count,
                len(google_matched_df),
                (matched_count / len(google_matched_df) * 100) if len(google_matched_df) else 0.0,
            )

        logger.info("Aggregating HotPepper rows to quarter mesh")
        mesh_agg_df = aggregate_hotpepper_by_mesh(google_matched_df)
        logger.info("HotPepper mesh rows: %d", len(mesh_agg_df))

        mesh1_codes = sorted({mesh_quarter_to_mesh1(code) for code in mesh_agg_df["jis_mesh"].dropna().astype(str)})
        logger.info("Unique mesh1 codes: %d", len(mesh1_codes))

        population_raw_df = load_or_fetch_population(
            tag=args.tag,
            mesh1_codes=mesh1_codes,
            skip_fetch=args.skip_fetch,
        )
        population_df, fallback_residuals = normalize_population_df(population_raw_df)
        logger.info("Normalized population mesh rows: %d", len(population_df))

        integrated_df = mesh_agg_df.merge(population_df, on="jis_mesh", how="left")
        for column in POPULATION_METRICS.values():
            if column not in integrated_df.columns:
                integrated_df[column] = pd.NA
            integrated_df[column] = pd.to_numeric(integrated_df[column], errors="coerce")

        if fallback_residuals:
            mesh3_codes = integrated_df["jis_mesh"].astype(str).str[:8]
            total_filled = 0
            total_missing = 0
            for column in POPULATION_METRICS.values():
                missing_mask = integrated_df[column].isna()
                total_missing += int(missing_mask.sum())
                if not missing_mask.any():
                    continue
                fallback_map = fallback_residuals.get(column, {})
                fallback_values = mesh3_codes.loc[missing_mask].map(fallback_map)
                integrated_df.loc[missing_mask, column] = fallback_values.values
                total_filled += int(fallback_values.notna().sum())
            logger.info(
                "Fallback demographic fill: %d/%d metric cells filled from parent mesh averages",
                total_filled,
                total_missing,
            )

        logger.info("Adding station, passenger, and land-price features")
        station_df = load_station_cache(args.tag)
        price_df = load_land_price_cache(args.tag)
        passenger_df = load_passenger_cache(args.tag)

        featured_df = add_all_features(
            integrated_df, station_df=station_df, passenger_df=passenger_df, price_df=price_df
        )
        featured_df = _merge_google_ratings(featured_df, google_df)

        full_scored_df = compute_opportunity_score_v3(featured_df).sort_values(
            "opportunity_score", ascending=False
        )
        full_scored_df["reason"] = full_scored_df.apply(generate_reason, axis=1)
        save_integrated(full_scored_df, args.tag)

        top_candidates = run_scoring_v3(featured_df, top_n=args.top_n)
        logger.info("Displaying top %d rows", min(args.top_n, len(top_candidates)))
        for rank, row in enumerate(top_candidates.itertuples(index=False), start=1):
            logger.info(
                "%d. mesh=%s genre=%s score=%.4f population=%.0f",
                rank,
                getattr(row, "jis_mesh", getattr(row, "jis_mesh3", "")),
                getattr(row, "unified_genre", ""),
                getattr(row, "opportunity_score", 0.0),
                getattr(row, "population", 0.0),
            )

        return 0
    except Exception:
        logger.exception("e-Stat integration pipeline failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
