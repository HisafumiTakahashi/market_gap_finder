#!/usr/bin/env python3
"""Add a new area by running the existing single-area pipeline components."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config import settings
from scripts.integrate_estat import (
    _merge_google_ratings,
    aggregate_hotpepper_by_mesh,
    load_or_fetch_population,
    normalize_population_df,
    save_integrated,
)
from src.analyze.features import add_all_features
from src.analyze.ml_model import load_model, prepare_features
from src.analyze.scoring import compute_opportunity_score_v3b, generate_reason
from src.collect.collector import run_collection
from src.collect.land_price import download_land_price, save_land_price_cache
from src.collect.station import fetch_all_stations, save_station_cache
from src.collect.station_passengers import download_station_passengers, save_passenger_cache
from src.preprocess.google_matcher import match_google_to_hotpepper
from src.preprocess.mesh_converter import mesh_quarter_to_mesh1
from src.visualize.report import generate_report

logger = logging.getLogger(__name__)

PREFECTURE_CODES = {
    "北海道": "01",
    "青森県": "02",
    "岩手県": "03",
    "宮城県": "04",
    "秋田県": "05",
    "山形県": "06",
    "福島県": "07",
    "茨城県": "08",
    "栃木県": "09",
    "群馬県": "10",
    "埼玉県": "11",
    "千葉県": "12",
    "東京都": "13",
    "神奈川県": "14",
    "新潟県": "15",
    "富山県": "16",
    "石川県": "17",
    "福井県": "18",
    "山梨県": "19",
    "長野県": "20",
    "岐阜県": "21",
    "静岡県": "22",
    "愛知県": "23",
    "三重県": "24",
    "滋賀県": "25",
    "京都府": "26",
    "大阪府": "27",
    "兵庫県": "28",
    "奈良県": "29",
    "和歌山県": "30",
    "鳥取県": "31",
    "島根県": "32",
    "岡山県": "33",
    "広島県": "34",
    "山口県": "35",
    "徳島県": "36",
    "香川県": "37",
    "愛媛県": "38",
    "高知県": "39",
    "福岡県": "40",
    "佐賀県": "41",
    "長崎県": "42",
    "熊本県": "43",
    "大分県": "44",
    "宮崎県": "45",
    "鹿児島県": "46",
    "沖縄県": "47",
}

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_HEADERS = {"User-Agent": "market-gap-finder/1.0"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect, score, and report a newly specified area.")
    parser.add_argument("--tag", required=True, help="Area tag for saved files.")
    parser.add_argument("--lat", type=float, required=True, help="Center latitude.")
    parser.add_argument("--lng", type=float, required=True, help="Center longitude.")
    parser.add_argument("--radius", type=float, default=5.0, help="Collection radius in km.")
    parser.add_argument("--top-n", type=int, default=20, help="Top N rows for the HTML report.")
    return parser.parse_args()


def _bounding_box(lat: float, lng: float, radius_km: float) -> tuple[float, float, float, float]:
    lat_delta = radius_km / 111.0
    lng_delta = radius_km / max(111.0 * math.cos(math.radians(lat)), 1e-6)
    return lat - lat_delta, lat + lat_delta, lng - lng_delta, lng + lng_delta


def _resolve_prefecture(lat: float, lng: float) -> tuple[str, str]:
    response = requests.get(
        NOMINATIM_URL,
        params={
            "lat": lat,
            "lon": lng,
            "format": "json",
            "zoom": 10,
            "accept-language": "ja",
        },
        headers=NOMINATIM_HEADERS,
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    address = payload.get("address", {})
    prefecture = str(address.get("state") or address.get("province") or "").strip()
    if prefecture not in PREFECTURE_CODES:
        raise KeyError(f"Unsupported prefecture for external data: {prefecture or 'unknown'}")
    return prefecture, PREFECTURE_CODES[prefecture]


def _fetch_external_data(tag: str, prefecture: str, pref_code: str) -> None:
    logger.info("Fetching station data for %s", prefecture)
    station_df = fetch_all_stations(prefecture)
    save_station_cache(station_df, tag)

    logger.info("Fetching land price data for %s (%s)", prefecture, pref_code)
    land_price_df = download_land_price(pref_code)
    save_land_price_cache(land_price_df, tag)

    logger.info("Fetching station passenger data")
    passenger_df = download_station_passengers()
    save_passenger_cache(passenger_df, tag)


def _integrate_area(tag: str) -> pd.DataFrame:
    logger.info("Loading collected HotPepper data")
    hotpepper_path = settings.RAW_DATA_DIR / f"{tag}_hotpepper.csv"
    if not hotpepper_path.exists():
        raise FileNotFoundError(hotpepper_path)
    hotpepper_df = pd.read_csv(hotpepper_path)

    google_path = settings.RAW_DATA_DIR / f"{tag}_google_places.csv"
    google_df: pd.DataFrame | None = None
    google_matched_df = hotpepper_df
    if google_path.exists():
        google_df = pd.read_csv(google_path)
        google_matched_df = match_google_to_hotpepper(hotpepper_df, google_df)

    logger.info("Aggregating HotPepper rows to quarter mesh")
    mesh_agg_df = aggregate_hotpepper_by_mesh(google_matched_df)
    mesh1_codes = sorted({mesh_quarter_to_mesh1(code) for code in mesh_agg_df["jis_mesh"].dropna().astype(str)})

    logger.info("Fetching or loading e-Stat population")
    population_raw_df = load_or_fetch_population(tag=tag, mesh1_codes=mesh1_codes, skip_fetch=False)
    population_df, fallback_residuals = normalize_population_df(population_raw_df)

    integrated_df = mesh_agg_df.merge(population_df, on="jis_mesh", how="left")
    for column in population_df.columns:
        if column == "jis_mesh":
            continue
        integrated_df[column] = pd.to_numeric(integrated_df[column], errors="coerce")

    if fallback_residuals:
        mesh3_codes = integrated_df["jis_mesh"].astype(str).str[:8]
        for column, fallback_map in fallback_residuals.items():
            if column not in integrated_df.columns:
                continue
            missing_mask = integrated_df[column].isna()
            if missing_mask.any():
                integrated_df.loc[missing_mask, column] = mesh3_codes.loc[missing_mask].map(fallback_map).values

    logger.info("Adding external features")
    station_df = pd.read_csv(settings.EXTERNAL_DATA_DIR / f"{tag}_stations.csv")
    passenger_df = pd.read_csv(settings.EXTERNAL_DATA_DIR / f"{tag}_station_passengers.csv")
    land_price_df = pd.read_csv(settings.EXTERNAL_DATA_DIR / f"{tag}_land_price.csv")

    featured_df = add_all_features(
        integrated_df,
        station_df=station_df,
        passenger_df=passenger_df,
        price_df=land_price_df,
    )
    featured_df = _merge_google_ratings(featured_df, google_df)
    scored_df = compute_opportunity_score_v3b(featured_df).sort_values("opportunity_score", ascending=False)
    scored_df["reason"] = scored_df.apply(generate_reason, axis=1)
    save_integrated(scored_df, tag)
    return scored_df


def _predict_market_gap(tag: str, integrated_df: pd.DataFrame) -> Path:
    logger.info("Loading ML model: models/combined_lightgbm.txt")
    model = load_model("combined")
    if model is None:
        raise FileNotFoundError(settings.PROJECT_ROOT / "models" / "combined_lightgbm.txt")

    features, _ = prepare_features(integrated_df)
    predicted_log_count = model.predict(features)
    actual_log_count = np.log1p(pd.to_numeric(integrated_df["restaurant_count"], errors="coerce").fillna(0).to_numpy())

    gap_df = integrated_df.copy()
    gap_df["predicted_log_count"] = predicted_log_count
    gap_df["actual_log_count"] = actual_log_count
    gap_df["market_gap"] = predicted_log_count - actual_log_count
    gap_df["predicted_count"] = np.expm1(predicted_log_count)
    gap_df["gap_count"] = gap_df["predicted_count"] - pd.to_numeric(
        gap_df["restaurant_count"], errors="coerce"
    ).fillna(0).to_numpy()

    output_path = settings.PROCESSED_DATA_DIR / f"{tag}_ml_gap.csv"
    gap_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("Saved ML gap results: %s", output_path)
    return output_path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    try:
        logger.info("Step 1/5: collecting HotPepper data for %s", args.tag)
        lat_min, lat_max, lng_min, lng_max = _bounding_box(args.lat, args.lng, args.radius)
        run_collection(lat_min=lat_min, lat_max=lat_max, lng_min=lng_min, lng_max=lng_max, output_tag=args.tag)

        logger.info("Step 2/5: resolving prefecture and fetching external data")
        prefecture, pref_code = _resolve_prefecture(args.lat, args.lng)
        _fetch_external_data(args.tag, prefecture, pref_code)

        logger.info("Step 3/5: integrating e-Stat and feature data")
        integrated_df = _integrate_area(args.tag)

        logger.info("Step 4/5: scoring with the combined LightGBM model")
        _predict_market_gap(args.tag, integrated_df)

        logger.info("Step 5/5: generating HTML report")
        output_path = generate_report(tag=args.tag, top_n=args.top_n, ml_r2="combined")
        logger.info("Completed area pipeline: %s", output_path)
        return 0
    except Exception as exc:
        logger.exception("Area pipeline failed for tag=%s", args.tag)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
