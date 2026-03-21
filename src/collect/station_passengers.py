"""駅別乗降客数データ収集ユーティリティ。"""

from __future__ import annotations

import io
import json
import logging
import re
import zipfile

import pandas as pd
import requests

from config import settings

logger = logging.getLogger(__name__)

_DIRECT_URL_TEMPLATE = "https://nlftp.mlit.go.jp/ksj/gml/data/S12/S12-{year_short}/S12-{year_short}_GML.zip"
_PASSENGER_KEY_PATTERN = re.compile(r"^S12_0(\d{2})$")


def _build_direct_url(fiscal_year: int) -> str:
    """年度下2桁を使って直接ダウンロードURLを組み立てる。"""
    year_short = str(fiscal_year)[-2:]
    return _DIRECT_URL_TEMPLATE.format(year_short=year_short)


def _extract_passenger_value(properties: dict) -> float | None:
    """feature の properties から最新の有効な乗降客数を抽出する。"""
    candidates: list[tuple[int, float]] = []

    for key, value in properties.items():
        match = _PASSENGER_KEY_PATTERN.match(str(key))
        if not match:
            continue

        key_num = int(match.group(1))
        if key_num < 9 or key_num in {6, 7, 8}:
            continue

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue

        candidates.append((key_num, numeric_value))

    if not candidates:
        return None

    # 最新年度から順に、正の値があればそれを返す
    candidates.sort(key=lambda item: item[0], reverse=True)
    for _, value in candidates:
        if value > 0:
            return value
    return None


def _parse_station_passengers_geojson(content: bytes) -> pd.DataFrame:
    """GeoJSON の駅別乗降客数データをパースする。"""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Shift-JIS エンコードの可能性
        try:
            data = json.loads(content.decode("cp932"))
        except Exception:
            logger.warning("GeoJSON decode failed")
            return pd.DataFrame(columns=["station_name", "operator", "line_name", "lat", "lng", "passengers"])

    records: list[dict] = []
    seen: set[tuple[str, float, float]] = set()

    for feature in data.get("features", []):
        properties = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        coordinates = geometry.get("coordinates", [])
        if not coordinates:
            continue

        passenger_value = _extract_passenger_value(properties)
        if passenger_value is None:
            continue

        station_name = str(properties.get("S12_001") or "").strip()
        operator = str(properties.get("S12_002") or "").strip()
        line_name = str(properties.get("S12_003") or "").strip()

        try:
            geom_type = geometry.get("type", "")
            if geom_type == "Point":
                lng = float(coordinates[0])
                lat = float(coordinates[1])
            elif geom_type in ("LineString", "MultiPoint"):
                # 中点を使用
                lngs = [float(c[0]) for c in coordinates]
                lats = [float(c[1]) for c in coordinates]
                lng = sum(lngs) / len(lngs)
                lat = sum(lats) / len(lats)
            else:
                # フォールバック: 最初の座標ペアを試す
                if isinstance(coordinates[0], (list, tuple)):
                    lng = float(coordinates[0][0])
                    lat = float(coordinates[0][1])
                else:
                    lng = float(coordinates[0])
                    lat = float(coordinates[1])
        except (TypeError, ValueError, IndexError):
            continue

        dedupe_key = (station_name, round(lat, 4), round(lng, 4))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        records.append(
            {
                "station_name": station_name,
                "operator": operator,
                "line_name": line_name,
                "lat": lat,
                "lng": lng,
                "passengers": passenger_value,
            }
        )

    logger.info("Parsed %d station passenger records from GeoJSON", len(records))
    return pd.DataFrame(records)


def download_station_passengers(fiscal_year: int = 2022) -> pd.DataFrame:
    """駅別乗降客数データをダウンロードしてパースする。"""
    for year in range(fiscal_year, fiscal_year - 3, -1):
        download_url = _build_direct_url(year)
        try:
            logger.info("Downloading station passenger ZIP: %s", download_url)
            response = requests.get(download_url, timeout=120)
            response.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                # UTF-8ディレクトリのGeoJSONを優先（Shift-JISエンコード問題回避）
                all_geojson = [name for name in zf.namelist() if name.endswith(".geojson")]
                utf8_geojson = [n for n in all_geojson if "UTF-8" in n or "utf-8" in n]
                geojson_files = utf8_geojson if utf8_geojson else all_geojson
                if not geojson_files:
                    logger.warning("No GeoJSON found in ZIP: %s", download_url)
                    continue

                for geojson_file in geojson_files:
                    parsed = _parse_station_passengers_geojson(zf.read(geojson_file))
                    if not parsed.empty:
                        logger.info("Downloaded %d station passenger records", len(parsed))
                        return parsed
        except Exception:
            logger.warning("Failed to download station passenger data: %s", download_url, exc_info=True)

    return pd.DataFrame(columns=["station_name", "operator", "line_name", "lat", "lng", "passengers"])


def save_passenger_cache(df: pd.DataFrame, tag: str) -> None:
    """駅別乗降客数キャッシュを保存する。"""
    settings.EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = settings.EXTERNAL_DATA_DIR / f"{tag}_station_passengers.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Saved station passenger cache: %s (%d rows)", path, len(df))


def load_passenger_cache(tag: str) -> pd.DataFrame | None:
    """駅別乗降客数キャッシュを読み込む。"""
    path = settings.EXTERNAL_DATA_DIR / f"{tag}_station_passengers.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    logger.info("Loaded station passenger cache: %s (%d rows)", path, len(df))
    return df
