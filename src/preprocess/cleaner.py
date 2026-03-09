from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


GENRE_MAPPING: dict[str, tuple[str, ...]] = {
    "izakaya": ("居酒屋",),
    "italian": ("イタリアン", "フレンチ"),
    "chinese": ("中華",),
    "yakiniku": ("焼肉", "韓国"),
    "cafe": ("カフェ",),
    "ramen": ("ラーメン",),
    "sushi": ("寿司", "和食"),
    "curry": ("カレー",),
}


def load_hotpepper(tag: str = "result") -> pd.DataFrame:
    file_path = settings.RAW_DATA_DIR / f"{tag}_hotpepper.csv"
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    logger.info("Loading Hotpepper data from %s", file_path)
    return pd.read_csv(file_path)


def map_genre(df: pd.DataFrame) -> pd.DataFrame:
    if "genre" not in df.columns:
        raise KeyError("genre column is required")

    def normalize_genre(value: object) -> str:
        text = str(value).strip().lower()
        if not text or text == "nan":
            return "other"

        for unified_genre, keywords in GENRE_MAPPING.items():
            if any(keyword.lower() in text for keyword in keywords):
                return unified_genre
        return "other"

    mapped = df.copy()
    mapped["unified_genre"] = mapped["genre"].map(normalize_genre)
    return mapped


def assign_mesh_code(df: pd.DataFrame) -> pd.DataFrame:
    if "lat" not in df.columns or "lng" not in df.columns:
        raise KeyError("lat and lng columns are required")

    assigned = df.copy()
    lat = pd.to_numeric(assigned["lat"], errors="coerce")
    lng = pd.to_numeric(assigned["lng"], errors="coerce")

    lat_bin = np.floor(lat / settings.MESH_LAT_STEP)
    lng_bin = np.floor(lng / settings.MESH_LON_STEP)
    valid_mask = lat.notna() & lng.notna()

    assigned["mesh_code"] = "unknown"
    assigned.loc[valid_mask, "mesh_code"] = (
        lat_bin[valid_mask].astype(int).astype(str)
        + "_"
        + lng_bin[valid_mask].astype(int).astype(str)
    )
    return assigned


def aggregate_by_mesh_genre(df: pd.DataFrame) -> pd.DataFrame:
    aggregated_input = assign_mesh_code(map_genre(df))
    aggregated_input["rating"] = pd.to_numeric(aggregated_input["rating"], errors="coerce")
    aggregated_input["review_count"] = (
        pd.to_numeric(aggregated_input["review_count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    aggregated_input["lat"] = pd.to_numeric(aggregated_input["lat"], errors="coerce")
    aggregated_input["lng"] = pd.to_numeric(aggregated_input["lng"], errors="coerce")

    aggregated = (
        aggregated_input.groupby(["mesh_code", "unified_genre"], dropna=False)
        .agg(
            restaurant_count=("id", "size"),
            avg_rating=("rating", "mean"),
            total_reviews=("review_count", "sum"),
            lat=("lat", "mean"),
            lng=("lng", "mean"),
        )
        .reset_index()
    )
    return aggregated


def save_processed(df: pd.DataFrame, filename: str) -> None:
    settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = settings.PROCESSED_DATA_DIR / filename
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("Saved processed data to %s", output_path)


def run_preprocess(tag: str = "result") -> pd.DataFrame:
    hotpepper_df = load_hotpepper(tag)
    aggregated_df = aggregate_by_mesh_genre(hotpepper_df)
    save_processed(aggregated_df, f"{tag}_aggregated.csv")
    return aggregated_df
