from __future__ import annotations

import logging

import pandas as pd

from config import settings
from src.collect import hotpepper

logger = logging.getLogger(__name__)


def generate_mesh(
    lat_min: float,
    lat_max: float,
    lng_min: float,
    lng_max: float,
) -> list[tuple[float, float]]:
    """指定範囲をメッシュ化し、格子点の座標を返す。"""
    if lat_min > lat_max:
        lat_min, lat_max = lat_max, lat_min
    if lng_min > lng_max:
        lng_min, lng_max = lng_max, lng_min

    mesh_points: list[tuple[float, float]] = []

    lat = lat_min
    while lat <= lat_max + 1e-12:
        lng = lng_min
        while lng <= lng_max + 1e-12:
            mesh_points.append((round(lat, 10), round(lng, 10)))
            lng += settings.MESH_LON_STEP
        lat += settings.MESH_LAT_STEP

    return mesh_points


def run_collection(
    lat_min: float,
    lat_max: float,
    lng_min: float,
    lng_max: float,
    output_tag: str = "result",
) -> pd.DataFrame:
    mesh_points = generate_mesh(
        lat_min=lat_min,
        lat_max=lat_max,
        lng_min=lng_min,
        lng_max=lng_max,
    )

    total_points = len(mesh_points)
    frames: list[pd.DataFrame] = []

    logger.info("Collection started: %s mesh points", total_points)

    for index, (lat, lng) in enumerate(mesh_points, start=1):
        logger.info(
            "Collecting mesh %s/%s: lat=%s, lng=%s",
            index,
            total_points,
            lat,
            lng,
        )

        records = hotpepper.fetch_all_pages(
            lat=lat,
            lng=lng,
            range_code=3,
        )
        if records is None:
            logger.warning(
                "Hotpepper collection failed at mesh %s/%s: lat=%s, lng=%s",
                index,
                total_points,
                lat,
                lng,
            )
            continue

        df = hotpepper.to_dataframe(records)
        if not df.empty:
            frames.append(df)

    result = (
        pd.concat(frames, ignore_index=True)
        if frames
        else hotpepper.to_dataframe([])
    )

    if "id" in result.columns:
        result = result.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)

    settings.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = settings.RAW_DATA_DIR / f"{output_tag}_hotpepper.csv"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    logger.info("CSV saved: %s", output_path)
    logger.info("Collection finished: %s records", len(result))

    return result
