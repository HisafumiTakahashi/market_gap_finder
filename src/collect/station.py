"""駅データ取得モジュール。

HeartRails Express API を利用して都道府県内の全鉄道駅の座標を取得し、
メッシュ単位の最寄り駅距離特徴量の計算に利用する。
"""

from __future__ import annotations

import logging
import math
import time

import pandas as pd
import requests

from config import settings

logger = logging.getLogger(__name__)

# HeartRails Express API
_HEARTRAILS_BASE = "http://express.heartrails.com/api/json"
_REQUEST_DELAY = 0.3  # APIリクエスト間隔（秒）


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """2点間のハーバーサイン距離をkm単位で返す。"""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def fetch_lines(prefecture: str) -> list[str]:
    """都道府県内の全路線名を取得する。"""
    resp = requests.get(
        _HEARTRAILS_BASE,
        params={"method": "getLines", "prefecture": prefecture},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    lines = data.get("response", {}).get("line", [])
    if isinstance(lines, str):
        lines = [lines]
    logger.info("%s の路線数: %d", prefecture, len(lines))
    return lines


def fetch_stations(line: str) -> list[dict]:
    """路線内の全駅情報を取得する。"""
    resp = requests.get(
        _HEARTRAILS_BASE,
        params={"method": "getStations", "line": line},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    stations = data.get("response", {}).get("station", [])
    if isinstance(stations, dict):
        stations = [stations]
    return stations


def fetch_all_stations(prefecture: str) -> pd.DataFrame:
    """都道府県内の全駅を取得し DataFrame で返す。

    Args:
        prefecture: 都道府県名（例: "東京都", "大阪府", "愛知県"）

    Returns:
        station_name, lat, lng, line, prefecture を含む DataFrame。
    """
    lines = fetch_lines(prefecture)
    all_stations = []
    seen = set()

    for i, line in enumerate(lines):
        try:
            time.sleep(_REQUEST_DELAY)
            stations = fetch_stations(line)
            for s in stations:
                name = s.get("name", "")
                lat = float(s.get("y", 0))
                lng = float(s.get("x", 0))
                key = (name, round(lat, 4), round(lng, 4))
                if key not in seen:
                    seen.add(key)
                    all_stations.append({
                        "station_name": name,
                        "lat": lat,
                        "lng": lng,
                        "line": line,
                        "prefecture": s.get("prefecture", prefecture),
                    })
        except Exception:
            logger.warning("路線 %s の駅取得に失敗。スキップします。", line)
            continue

        if (i + 1) % 20 == 0:
            logger.info("路線取得進捗: %d/%d (駅数: %d)", i + 1, len(lines), len(all_stations))

    logger.info("%s の駅取得完了: %d 駅", prefecture, len(all_stations))
    return pd.DataFrame(all_stations)


def save_station_cache(df: pd.DataFrame, tag: str) -> None:
    """駅データをキャッシュ保存する。"""
    settings.EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = settings.EXTERNAL_DATA_DIR / f"{tag}_stations.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("駅データキャッシュ保存: %s (%d 駅)", path, len(df))


def load_station_cache(tag: str) -> pd.DataFrame | None:
    """駅データキャッシュを読み込む。存在しなければ None。"""
    path = settings.EXTERNAL_DATA_DIR / f"{tag}_stations.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    logger.info("駅データキャッシュ読み込み: %s (%d 駅)", path, len(df))
    return df
