"""
Google Places API クライアント

Google Places API (New) を利用して、
指定エリアの飲食店情報・口コミ評価を取得する。

Reference
---------
https://developers.google.com/maps/documentation/places/web-service
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from config import settings

logger = logging.getLogger(__name__)


def nearby_search(
    lat: float,
    lng: float,
    radius: int = 1000,
    place_type: str = "restaurant",
    keyword: str | None = None,
    page_token: str | None = None,
) -> dict[str, Any] | None:
    """Nearby Search で周辺の飲食店を検索する。

    Parameters
    ----------
    lat : float
        検索中心の緯度
    lng : float
        検索中心の経度
    radius : int, optional
        検索半径（メートル）, デフォルト 1000
    place_type : str, optional
        場所タイプ, デフォルト "restaurant"
    keyword : str | None, optional
        追加キーワード（ジャンルなど）
    page_token : str | None, optional
        ページネーション用トークン

    Returns
    -------
    dict[str, Any] | None
        API レスポンス全体（results, next_page_token 等）。エラー時は None
    """
    if not settings.GOOGLE_PLACES_API_KEY:
        logger.error("GOOGLE_PLACES_API_KEY が設定されていません")
        return None

    # next_page_token 使用前に 2秒 wait
    if page_token:
        time.sleep(2)

    url = f"{settings.GOOGLE_PLACES_BASE_URL}/nearbysearch/json"
    params: dict[str, Any] = {
        "key": settings.GOOGLE_PLACES_API_KEY,
        "location": f"{lat},{lng}",
        "radius": int(radius),
        "type": place_type,
    }
    if keyword:
        params["keyword"] = keyword
    if page_token:
        params["pagetoken"] = page_token

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except Exception as e:
        logger.error("Google Places Nearby Search リクエスト失敗: %s", e)
        return None

    status = payload.get("status", "UNKNOWN")
    if status in {"OK", "ZERO_RESULTS"}:
        return payload

    logger.error("Google Places Nearby Search エラー: status=%s, %s",
                 status, payload.get("error_message", ""))
    return None


def fetch_all_nearby(
    lat: float,
    lng: float,
    radius: int = 1000,
    place_type: str = "restaurant",
    keyword: str | None = None,
) -> list[dict[str, Any]] | None:
    """next_page_token を辿って全件取得する。

    Parameters
    ----------
    lat : float
        検索中心の緯度
    lng : float
        検索中心の経度
    radius : int, optional
        検索半径（メートル）
    place_type : str, optional
        場所タイプ
    keyword : str | None, optional
        追加キーワード

    Returns
    -------
    list[dict[str, Any]] | None
        全ページ分の place 情報リスト。エラー時は None
    """
    all_results: list[dict[str, Any]] = []
    next_token: str | None = None
    pages = 0

    while True:
        payload = nearby_search(
            lat=lat,
            lng=lng,
            radius=radius,
            place_type=place_type,
            keyword=keyword,
            page_token=next_token,
        )
        if payload is None:
            return None

        results = payload.get("results", [])
        if isinstance(results, list):
            all_results.extend(results)

        next_token = payload.get("next_page_token")
        pages += 1

        if not next_token or pages >= 3:
            break

    return all_results


def get_place_details(place_id: str) -> dict[str, Any] | None:
    """Place Details を取得する。

    Parameters
    ----------
    place_id : str
        Google Place ID

    Returns
    -------
    dict[str, Any] | None
        詳細情報（評価、口コミ数、営業時間など）。エラー時は None
    """
    if not settings.GOOGLE_PLACES_API_KEY:
        logger.error("GOOGLE_PLACES_API_KEY が設定されていません")
        return None

    url = f"{settings.GOOGLE_PLACES_BASE_URL}/details/json"
    params = {
        "key": settings.GOOGLE_PLACES_API_KEY,
        "place_id": place_id,
        "fields": "place_id,name,formatted_address,geometry,rating,user_ratings_total,types,business_status",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except Exception as e:
        logger.error("Google Places Details リクエスト失敗: %s", e)
        return None

    status = payload.get("status", "UNKNOWN")
    if status not in {"OK", "ZERO_RESULTS"}:
        logger.error("Google Places Details エラー: status=%s, %s",
                     status, payload.get("error_message", ""))
        return None

    return payload.get("result", {})


def text_search(shop_name: str, location_hint: str | None = None) -> str | None:
    """Text Search API で店名から place_id を取得する。

    Parameters
    ----------
    shop_name : str
        検索する店舗名
    location_hint : str | None, optional
        絞り込み用のロケーション文字列（例: "新宿区"）

    Returns
    -------
    str | None
        最初にヒットした place_id。見つからない場合は None
    """
    if not settings.GOOGLE_PLACES_API_KEY:
        logger.error("GOOGLE_PLACES_API_KEY が設定されていません")
        return None

    query = f"{shop_name} {location_hint}" if location_hint else shop_name
    url = f"{settings.GOOGLE_PLACES_BASE_URL}/textsearch/json"
    params = {
        "key": settings.GOOGLE_PLACES_API_KEY,
        "query": query,
        "type": "restaurant",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except Exception as e:
        logger.error("Google Places Text Search リクエスト失敗: %s", e)
        return None

    status = payload.get("status", "UNKNOWN")
    if status not in {"OK", "ZERO_RESULTS"}:
        logger.error("Google Places Text Search エラー: status=%s", status)
        return None

    results = payload.get("results", [])
    if results:
        return results[0].get("place_id")
    return None


def to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """API レスポンスを DataFrame に変換する。

    Parameters
    ----------
    records : list[dict[str, Any]]
        fetch 系関数の戻り値

    Returns
    -------
    pd.DataFrame
        整形済みの DataFrame
    """
    if not records:
        return pd.DataFrame(
            columns=["place_id", "name", "genre", "types", "address",
                     "lat", "lng", "rating", "review_count", "source"]
        )

    rows: list[dict[str, Any]] = []
    for record in records:
        geometry = record.get("geometry") or {}
        location = geometry.get("location") or {}
        types = record.get("types") or []
        genre = types[0] if types else None

        rows.append({
            "place_id": record.get("place_id"),
            "name": record.get("name"),
            "genre": genre,
            "types": ",".join(types) if types else None,
            "address": record.get("formatted_address") or record.get("vicinity"),
            "lat": location.get("lat"),
            "lng": location.get("lng"),
            "rating": record.get("rating"),
            "review_count": record.get("user_ratings_total"),
            "source": "google_places",
        })

    df = pd.DataFrame(rows)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce")
    return df


def save_raw(df: pd.DataFrame, filename: str) -> None:
    """Raw データとして CSV 保存する。

    Parameters
    ----------
    df : pd.DataFrame
        保存対象
    filename : str
        保存ファイル名（data/raw/ 配下に保存）
    """
    settings.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    target = Path(filename)
    if target.suffix.lower() != ".csv":
        target = target.with_suffix(".csv")

    output_path = settings.RAW_DATA_DIR / target.name
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("Google Places raw データ保存: %s", output_path)
