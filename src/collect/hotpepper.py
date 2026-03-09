"""
ホットペッパーグルメサーチ API クライアント

リクルートが提供するホットペッパーグルメサーチ API を利用して、
指定エリア・ジャンルの飲食店データを取得する。

Reference
---------
https://webservice.recruit.co.jp/doc/hotpepper/reference.html
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from config import settings

logger = logging.getLogger(__name__)


def fetch_restaurants_by_area(
    lat: float,
    lng: float,
    range_code: int = 3,
    genre_code: str | None = None,
    start: int = 1,
    count: int = 100,
) -> list[dict[str, Any]] | None:
    """指定座標を中心にホットペッパーAPIで飲食店を検索する。

    Parameters
    ----------
    lat : float
        検索中心の緯度
    lng : float
        検索中心の経度
    range_code : int, optional
        検索範囲コード (1:300m, 2:500m, 3:1000m, 4:2000m, 5:3000m),
        デフォルトは 3
    genre_code : str | None, optional
        ジャンルコード（例: "G001"＝居酒屋）
    start : int, optional
        取得開始位置, デフォルト 1
    count : int, optional
        取得件数, デフォルト 100

    Returns
    -------
    list[dict[str, Any]] | None
        取得した店舗情報のリスト。エラー時は None
    """
    if not settings.HOTPEPPER_API_KEY:
        logger.error("HOTPEPPER_API_KEY が設定されていません")
        return None

    count = max(1, min(int(count), settings.HOTPEPPER_MAX_RESULTS))
    params: dict[str, Any] = {
        "key": settings.HOTPEPPER_API_KEY,
        "lat": lat,
        "lng": lng,
        "range": range_code,
        "start": max(1, int(start)),
        "count": count,
        "format": "json",
    }
    if genre_code:
        params["genre"] = genre_code

    try:
        response = requests.get(settings.HOTPEPPER_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except Exception as e:
        logger.error("ホットペッパーAPI リクエスト失敗: %s", e)
        return None

    results = payload.get("results", {})
    error = results.get("error")
    if error:
        logger.error("ホットペッパーAPI エラー: %s", error)
        return None

    shops = results.get("shop", [])
    return shops if isinstance(shops, list) else []


def fetch_all_pages(
    lat: float,
    lng: float,
    range_code: int = 3,
    genre_code: str | None = None,
) -> list[dict[str, Any]] | None:
    """ページネーションを処理し、全件を取得する。

    Parameters
    ----------
    lat : float
        検索中心の緯度
    lng : float
        検索中心の経度
    range_code : int, optional
        検索範囲コード
    genre_code : str | None, optional
        ジャンルコード

    Returns
    -------
    list[dict[str, Any]] | None
        全ページ分を結合した店舗情報リスト。エラー時は None
    """
    all_records: list[dict[str, Any]] = []
    start = 1
    per_page = settings.HOTPEPPER_MAX_RESULTS

    while True:
        page_records = fetch_restaurants_by_area(
            lat=lat,
            lng=lng,
            range_code=range_code,
            genre_code=genre_code,
            start=start,
            count=per_page,
        )
        if page_records is None:
            return None
        if not page_records:
            break

        all_records.extend(page_records)

        if len(page_records) < per_page:
            break

        start += len(page_records)
        if start > 10000:
            logger.warning("start=%s でページング停止（無限ループ防止）", start)
            break

    return all_records


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
            columns=["id", "name", "genre", "genre_code", "address", "lat", "lng",
                     "rating", "review_count", "source"]
        )

    rows: list[dict[str, Any]] = []
    for record in records:
        genre_data = record.get("genre")
        genre_data = genre_data if isinstance(genre_data, dict) else {}
        rows.append({
            "id": record.get("id"),
            "name": record.get("name"),
            "genre": genre_data.get("name"),
            "genre_code": genre_data.get("code"),
            "address": record.get("address"),
            "lat": record.get("lat"),
            "lng": record.get("lng"),
            "rating": pd.NA,
            "review_count": pd.NA,
            "source": "hotpepper",
        })

    df = pd.DataFrame(rows)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
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
    logger.info("ホットペッパー raw データ保存: %s", output_path)
