#!/usr/bin/env python3
"""e-Stat integration pipeline CLI."""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from config import settings
from src.analyze.features import add_all_features
from src.analyze.scoring import (
    compute_opportunity_score_v2,
    compute_opportunity_score_v3,
    generate_reason,
    run_scoring_v3,
)
from src.collect.estat import fetch_mesh_population, save_raw
from src.collect.land_price import load_land_price_cache
from src.collect.station import load_station_cache
from src.collect.station_passengers import load_passenger_cache
from src.preprocess.cleaner import load_hotpepper, map_genre
from src.preprocess.google_matcher import match_google_to_hotpepper
from src.preprocess.mesh_converter import assign_jis_mesh, lat_lon_to_mesh3, mesh3_to_mesh1


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
    """Aggregate HotPepper rows by JIS mesh and genre."""
    prepared = assign_jis_mesh(map_genre(df))
    prepared["lat"] = pd.to_numeric(prepared.get("lat"), errors="coerce")
    prepared["lng"] = pd.to_numeric(prepared.get("lng"), errors="coerce")
    prepared = prepared.dropna(subset=["jis_mesh3"]).copy()
    has_google_cols = {"google_rating", "google_review_count"} <= set(prepared.columns)

    if has_google_cols:
        prepared["google_rating"] = pd.to_numeric(prepared["google_rating"], errors="coerce")
        prepared["google_review_count"] = pd.to_numeric(
            prepared["google_review_count"], errors="coerce"
        )
        aggregated = (
            prepared.groupby(["jis_mesh3", "unified_genre"], dropna=False)
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
            prepared.groupby(["jis_mesh3", "unified_genre"], dropna=False)
            .agg(
                restaurant_count=("id", "size"),
                lat=("lat", "mean"),
                lng=("lng", "mean"),
            )
            .reset_index()
        )

    aggregated["mesh_code"] = aggregated["jis_mesh3"]
    return aggregated


def normalize_population_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize e-Stat population data to JIS mesh 3 totals."""
    if df.empty:
        return pd.DataFrame(columns=["jis_mesh3", "population"])

    normalized = df.copy()

    if "mesh_code" not in normalized.columns:
        raise KeyError("mesh_code column is required in e-Stat data")

    normalized["jis_mesh3"] = normalized["mesh_code"].astype(str).str.extract(r"(\d{8})", expand=False)
    normalized["population"] = (
        normalized["population"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )

    if "cat01" in normalized.columns:
        cat01_str = normalized["cat01"].astype(str)
        total_mask = cat01_str.isin(["0010"]) | cat01_str.str.contains(
            "総数|人口|世帯",
            na=False,
        )
        if total_mask.any():
            normalized = normalized.loc[total_mask].copy()

    normalized = normalized.dropna(subset=["jis_mesh3", "population"])
    if normalized.empty:
        return pd.DataFrame(columns=["jis_mesh3", "population"])

    return (
        normalized.groupby("jis_mesh3", as_index=False)
        .agg(population=("population", "sum"))
        .sort_values("jis_mesh3")
        .reset_index(drop=True)
    )


def load_or_fetch_population(tag: str, mesh1_codes: list[str], skip_fetch: bool) -> pd.DataFrame:
    """Load cached e-Stat data or fetch it."""
    cache_name = f"{tag}_estat_population.csv"
    cache_path = settings.RAW_DATA_DIR / cache_name

    if skip_fetch:
        if not cache_path.exists():
            raise FileNotFoundError(cache_path)
        logger.info("Loading cached e-Stat population data: %s", cache_path)
        return pd.read_csv(cache_path)

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


def _merge_google_ratings(
    integrated_df: pd.DataFrame, google_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Compute derived Google aggregate features from pre-matched mesh data and raw Google data."""
    out = integrated_df.copy()
    out["google_avg_rating"] = pd.to_numeric(out.get("google_avg_rating"), errors="coerce").fillna(0.0)
    out["google_total_reviews"] = pd.to_numeric(
        out.get("google_total_reviews"), errors="coerce"
    ).fillna(0.0)
    out["google_match_count"] = pd.to_numeric(
        out.get("google_match_count"), errors="coerce"
    ).fillna(0).astype(int)
    out["reviews_per_shop"] = out["google_total_reviews"] / out["google_match_count"].replace(0, 1)

    # google_density: Google Places生データの登録数 / 人口（ターゲット非依存）
    population = pd.to_numeric(out.get("population"), errors="coerce").fillna(0.0)
    if google_df is not None and not google_df.empty and "jis_mesh3" not in out.columns:
        out["google_density"] = 0.0
    elif google_df is not None and not google_df.empty:
        from src.preprocess.mesh_converter import lat_lon_to_mesh3

        gdf = google_df.copy()
        gdf["lat"] = pd.to_numeric(gdf["lat"], errors="coerce")
        gdf["lng"] = pd.to_numeric(gdf["lng"], errors="coerce")
        gdf = gdf.dropna(subset=["lat", "lng"])
        gdf["jis_mesh3"] = [lat_lon_to_mesh3(lat, lng) for lat, lng in zip(gdf["lat"], gdf["lng"])]
        mesh_gp_count = gdf.groupby("jis_mesh3").size().rename("google_place_count")
        out = out.merge(mesh_gp_count, on="jis_mesh3", how="left")
        out["google_place_count"] = out["google_place_count"].fillna(0).astype(int)
        out["google_density"] = out["google_place_count"] / (population / 10000 + 0.1)
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

        logger.info("Aggregating HotPepper rows to JIS mesh 3")
        mesh_agg_df = aggregate_hotpepper_by_mesh(google_matched_df)
        logger.info("HotPepper mesh rows: %d", len(mesh_agg_df))

        mesh1_codes = sorted({mesh3_to_mesh1(code) for code in mesh_agg_df["jis_mesh3"].dropna().astype(str)})
        logger.info("Unique mesh1 codes: %d", len(mesh1_codes))

        population_raw_df = load_or_fetch_population(
            tag=args.tag,
            mesh1_codes=mesh1_codes,
            skip_fetch=args.skip_fetch,
        )
        population_df = normalize_population_df(population_raw_df)
        logger.info("Normalized population mesh rows: %d", len(population_df))

        integrated_df = mesh_agg_df.merge(population_df, on="jis_mesh3", how="left")
        integrated_df["population"] = pd.to_numeric(
            integrated_df["population"], errors="coerce"
        ).fillna(0.0)

        logger.info("Adding station, passenger, and land-price features")
        station_df = load_station_cache(args.tag)
        price_df = load_land_price_cache(args.tag)
        passenger_df = load_passenger_cache(args.tag)
        if station_df is not None:
            logger.info("Loaded stations: %d", len(station_df))
        if price_df is not None:
            logger.info("Loaded land prices: %d", len(price_df))
        if passenger_df is not None:
            logger.info("Loaded station passengers: %d", len(passenger_df))

        featured_df = add_all_features(
            integrated_df, station_df=station_df, passenger_df=passenger_df, price_df=price_df
        )
        if google_path.exists():
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
                getattr(row, "jis_mesh3", ""),
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
