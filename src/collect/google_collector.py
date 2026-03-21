"""Google Places APIによるメッシュベースの飲食店データ収集。"""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd

from config import settings
from src.collect.google_places import fetch_all_nearby, to_dataframe

logger = logging.getLogger(__name__)

_REQUEST_DELAY = 0.2


def generate_mesh(
    lat_min: float,
    lat_max: float,
    lng_min: float,
    lng_max: float,
) -> list[tuple[float, float]]:
    """メッシュグリッドの中心点を生成する。"""
    points: list[tuple[float, float]] = []
    lat = min(lat_min, lat_max)
    lat_end = max(lat_min, lat_max)
    lng_start = min(lng_min, lng_max)
    lng_end = max(lng_min, lng_max)

    while lat <= lat_end + 1e-12:
        lng = lng_start
        while lng <= lng_end + 1e-12:
            points.append((round(lat, 10), round(lng, 10)))
            lng += settings.MESH_LON_STEP
        lat += settings.MESH_LAT_STEP

    return points


def run_google_collection(
    lat_min: float,
    lat_max: float,
    lng_min: float,
    lng_max: float,
    output_tag: str = "result",
    radius: int = 1000,
) -> pd.DataFrame:
    """メッシュグリッド上でGoogle Places Nearby Searchを実行し、飲食店データを収集する。"""
    if not settings.GOOGLE_PLACES_API_KEY:
        raise ValueError(
            "GOOGLE_PLACES_API_KEY が設定されていません。環境変数を設定してください。"
        )

    mesh_points = generate_mesh(lat_min, lat_max, lng_min, lng_max)
    logger.info("Google Places collection: %d mesh points for tag=%s", len(mesh_points), output_tag)

    all_records: list[dict[str, Any]] = []
    seen_place_ids: set[str] = set()

    for i, (lat, lng) in enumerate(mesh_points):
        if i > 0:
            time.sleep(_REQUEST_DELAY)

        try:
            results = fetch_all_nearby(lat, lng, radius=radius, place_type="restaurant")
            if results is None:
                continue

            for record in results:
                place_id = record.get("place_id")
                if place_id and place_id not in seen_place_ids:
                    seen_place_ids.add(place_id)
                    all_records.append(record)
        except Exception:
            logger.warning("Google Places fetch failed at (%f, %f)", lat, lng, exc_info=True)

        if (i + 1) % 50 == 0:
            logger.info(
                "Progress: %d/%d mesh points, %d unique places",
                i + 1,
                len(mesh_points),
                len(all_records),
            )

    df = to_dataframe(all_records)

    settings.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = settings.RAW_DATA_DIR / f"{output_tag}_google_places.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("Google Places data saved: %s (%d records)", output_path, len(df))

    return df
