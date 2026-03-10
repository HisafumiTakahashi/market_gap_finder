"""メッシュ座標を地名へ逆ジオコーディングするユーティリティ。"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
_USER_AGENT = "market-gap-finder/1.0"
_PLACE_KEYS = ("quarter", "neighbourhood", "suburb", "city_district", "town", "city")


def _load_cache(cache_path: Path) -> dict[str, str]:
    """逆ジオコーディング結果の JSON キャッシュを読み込む。

    Args:
        cache_path: キャッシュファイルのパス。

    Returns:
        メッシュコードをキー、地名を値とする辞書。読込失敗時は空辞書。
    """
    if not cache_path.exists():
        return {}

    try:
        with cache_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logging.warning("Failed to load geocode cache %s: %s", cache_path, exc)
        return {}

    if not isinstance(data, dict):
        logging.warning("Invalid geocode cache format: %s", cache_path)
        return {}

    return {str(key): str(value) for key, value in data.items()}


def _save_cache(cache_path: Path, cache: dict[str, str]) -> None:
    """逆ジオコーディング結果のキャッシュを JSON として保存する。

    Args:
        cache_path: 保存先キャッシュファイルのパス。
        cache: 保存するキャッシュ辞書。
    """
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logging.warning("Failed to save geocode cache %s: %s", cache_path, exc)


def _extract_place_name(payload: dict[str, Any]) -> str:
    """Nominatim のレスポンスから代表的な地名を抽出する。

    Args:
        payload: 逆ジオコーディング API のレスポンス JSON。

    Returns:
        優先順位付きの住所キーから見つかった地名。見つからない場合は `unknown`。
    """
    address = payload.get("address", {})
    if not isinstance(address, dict):
        return "unknown"

    for key in _PLACE_KEYS:
        value = address.get(key)
        if value:
            return str(value)

    return "unknown"


def _reverse_geocode(lat: float, lng: float) -> str:
    """Nominatim API を使って単一点の地名を取得する。

    Args:
        lat: 対象地点の緯度。
        lng: 対象地点の経度。

    Returns:
        取得した地名文字列。

    Raises:
        requests.RequestException: HTTP 通信またはステータス検証に失敗した場合。
    """
    response = requests.get(
        _NOMINATIM_URL,
        params={
            "lat": lat,
            "lon": lng,
            "format": "json",
            "zoom": 14,
            "accept-language": "ja",
        },
        headers={"User-Agent": _USER_AGENT},
        timeout=10,
    )
    response.raise_for_status()
    return _extract_place_name(response.json())


def reverse_geocode_mesh(df: pd.DataFrame, cache_path: Path | None = None) -> pd.DataFrame:
    """DataFrame 内のメッシュコードに地名列を付与する。

    重複しないメッシュごとに逆ジオコーディングを実行し、結果を `place_name`
    列として元の DataFrame に結合する。キャッシュファイルが指定されている場合は
    既存結果を再利用し、新規結果を保存する。

    Args:
        df: `mesh_code`、`lat`、`lng` 列を含む DataFrame。
        cache_path: JSON キャッシュの保存先。`None` の場合はキャッシュしない。

    Returns:
        `place_name` 列を追加した DataFrame のコピー。

    Raises:
        KeyError: 必須列が不足している場合。
    """
    required_columns = {"mesh_code", "lat", "lng"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns: {sorted(missing_columns)}")

    result = df.copy()
    mesh_points = result.loc[:, ["mesh_code", "lat", "lng"]].copy()
    mesh_points["lat"] = pd.to_numeric(mesh_points["lat"], errors="coerce")
    mesh_points["lng"] = pd.to_numeric(mesh_points["lng"], errors="coerce")
    mesh_points = mesh_points.dropna(subset=["mesh_code", "lat", "lng"])
    mesh_points["mesh_code"] = mesh_points["mesh_code"].astype(str)
    mesh_points = mesh_points.drop_duplicates(subset=["mesh_code"], keep="first")

    cache: dict[str, str] = {}
    if cache_path is not None:
        cache = _load_cache(cache_path)

    place_name_map = {mesh_code: cache_value for mesh_code, cache_value in cache.items()}
    targets = mesh_points[~mesh_points["mesh_code"].isin(cache)].reset_index(drop=True)

    logging.info("Geocoding %d unique meshes", len(targets))

    for idx, row in targets.iterrows():
        mesh_code = row["mesh_code"]
        lat = float(row["lat"])
        lng = float(row["lng"])

        try:
            place_name = _reverse_geocode(lat, lng)
        except Exception as exc:
            logging.warning("Failed to geocode %s: %s", mesh_code, exc)
            place_name = "unknown"

        place_name_map[mesh_code] = place_name
        cache[mesh_code] = place_name
        logging.info("Geocoded %s -> %s", mesh_code, place_name)

        if idx < len(targets) - 1:
            time.sleep(1.1)

    if cache_path is not None:
        _save_cache(cache_path, cache)

    result["place_name"] = result["mesh_code"].astype(str).map(place_name_map).fillna("unknown")
    return result
