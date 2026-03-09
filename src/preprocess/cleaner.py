"""
データクレンジング・名寄せ・集約モジュール

ホットペッパー / Google Places から取得した Raw データを統合し、
分析に適した形式へ前処理する。

主な処理
--------
- 重複除去（名寄せ）
- ジャンルの統一カテゴリへのマッピング
- 緯度経度を使ったメッシュコードの付与
- メッシュ × ジャンル単位での集約
"""

from __future__ import annotations

import difflib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


def _coalesce_column(df: pd.DataFrame, candidates: tuple[str, ...], default: Any = None) -> pd.Series:
    """カラム候補を順番に探して最初に見つかったものを返す。"""
    for col in candidates:
        if col in df.columns:
            return df[col]
    return pd.Series([default] * len(df), index=df.index)


def _haversine_m(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Haversine 公式で距離（メートル）を計算する。"""
    earth_radius = 6_371_000.0
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return earth_radius * c


def _standardize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """各ソースの DataFrame を統一スキーマに変換する。"""
    out = pd.DataFrame({
        "id": _coalesce_column(df, ("id", "place_id", "shop_id")),
        "name": _coalesce_column(df, ("name", "shop_name")),
        "genre": _coalesce_column(df, ("genre", "genre_name", "types")),
        "rating": _coalesce_column(df, ("rating", "avg_rating", "score")),
        "review_count": _coalesce_column(df, ("review_count", "user_ratings_total", "reviews"), 0),
        "lat": _coalesce_column(df, ("lat", "latitude")),
        "lng": _coalesce_column(df, ("lng", "lon", "longitude")),
        "source": source,
    })
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lng"] = pd.to_numeric(out["lng"], errors="coerce")
    out["rating"] = pd.to_numeric(out["rating"], errors="coerce")
    out["review_count"] = pd.to_numeric(out["review_count"], errors="coerce").fillna(0.0)
    return out


def load_raw_data(source: str) -> pd.DataFrame:
    """Raw データを読み込む。

    Parameters
    ----------
    source : str
        "hotpepper" または "google_places"

    Returns
    -------
    pd.DataFrame
        Raw データ
    """
    source = source.strip().lower()
    if source not in {"hotpepper", "google_places"}:
        raise ValueError("source は 'hotpepper' または 'google_places' を指定してください")

    files = sorted(
        settings.RAW_DATA_DIR.glob(f"{source}*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(f"source='{source}' の raw CSV が見つかりません: {settings.RAW_DATA_DIR}")

    return pd.read_csv(files[0])


def deduplicate(
    hp_df: pd.DataFrame,
    gp_df: pd.DataFrame,
    distance_threshold_m: float = 50.0,
) -> pd.DataFrame:
    """ホットペッパーと Google Places のデータを名寄せ・統合する。

    Parameters
    ----------
    hp_df : pd.DataFrame
        ホットペッパー由来データ
    gp_df : pd.DataFrame
        Google Places 由来データ
    distance_threshold_m : float, optional
        同一店舗とみなす距離閾値（メートル）, デフォルト 50

    Returns
    -------
    pd.DataFrame
        重複除去済みの統合データ
    """
    hp = _standardize(hp_df, "hotpepper").dropna(subset=["lat", "lng"]).reset_index(drop=True)
    gp = _standardize(gp_df, "google_places").dropna(subset=["lat", "lng"]).reset_index(drop=True)

    if hp.empty:
        return gp.copy()
    if gp.empty:
        return hp.copy()

    merged = hp.copy()

    for _, gp_row in gp.iterrows():
        distances = _haversine_m(
            lat1=float(gp_row["lat"]),
            lon1=float(gp_row["lng"]),
            lat2=merged["lat"].to_numpy(dtype=float),
            lon2=merged["lng"].to_numpy(dtype=float),
        )
        nearest_idx = int(np.argmin(distances))
        nearest_dist = float(distances[nearest_idx])

        if nearest_dist <= distance_threshold_m:
            base = merged.iloc[nearest_idx].copy()
            base_name = str(base.get("name") or "").strip()
            gp_name = str(gp_row.get("name") or "").strip()
            similarity = difflib.SequenceMatcher(None, base_name, gp_name).ratio()

            if similarity >= 0.7:
                # Google の rating/review_count が優れていればマージ
                gp_rating = pd.to_numeric(pd.Series([gp_row.get("rating")]), errors="coerce").iloc[0]
                base_rating = pd.to_numeric(pd.Series([base.get("rating")]), errors="coerce").iloc[0]
                if pd.notna(gp_rating) and (pd.isna(base_rating) or gp_rating > base_rating):
                    base["rating"] = float(gp_rating)

                gp_reviews = pd.to_numeric(pd.Series([gp_row.get("review_count")]), errors="coerce").iloc[0]
                base_reviews = pd.to_numeric(pd.Series([base.get("review_count")]), errors="coerce").iloc[0]
                if pd.notna(gp_reviews) and (pd.isna(base_reviews) or gp_reviews > base_reviews):
                    base["review_count"] = float(gp_reviews)

                if not base.get("genre") and gp_row.get("genre"):
                    base["genre"] = gp_row["genre"]

                base["source"] = "hotpepper+google_places"
                merged.iloc[nearest_idx] = base
                continue

        merged = pd.concat([merged, pd.DataFrame([gp_row.to_dict()])], ignore_index=True)

    return merged.reset_index(drop=True)


def map_genre(df: pd.DataFrame) -> pd.DataFrame:
    """各ソース固有のジャンルを統一カテゴリにマッピングする。

    Parameters
    ----------
    df : pd.DataFrame
        統合済みデータ

    Returns
    -------
    pd.DataFrame
        unified_genre カラムが追加されたデータ
    """
    genre_map: dict[str, list[str]] = {
        "izakaya": ["居酒屋", "izakaya", "pub"],
        "italian": ["イタリアン", "italian", "pizza", "pasta", "フレンチ", "french"],
        "chinese": ["中華", "chinese", "dimsum"],
        "yakiniku": ["焼肉", "yakiniku", "bbq", "korean", "韓国"],
        "cafe": ["カフェ", "cafe", "coffee"],
        "ramen": ["ラーメン", "ramen", "noodle"],
        "sushi": ["寿司", "sushi", "和食", "washoku"],
        "curry": ["カレー", "curry"],
    }

    def _normalize(value: Any) -> str:
        text = str(value or "").strip().lower()
        for unified, keywords in genre_map.items():
            if any(kw.lower() in text for kw in keywords):
                return unified
        return "other"

    out = df.copy()
    source_col = _coalesce_column(out, ("unified_genre", "genre", "types", "category"), "")
    out["unified_genre"] = source_col.map(_normalize)
    return out


def assign_mesh_code(
    df: pd.DataFrame,
    lat_step: float | None = None,
    lon_step: float | None = None,
) -> pd.DataFrame:
    """緯度経度からメッシュコードを算出・付与する。

    Parameters
    ----------
    df : pd.DataFrame
        lat, lng カラムを持つデータ
    lat_step : float | None, optional
        メッシュの緯度刻み幅（Noneなら settings から取得）
    lon_step : float | None, optional
        メッシュの経度刻み幅

    Returns
    -------
    pd.DataFrame
        mesh_code カラムが追加されたデータ
    """
    lat_step = settings.MESH_LAT_STEP if lat_step is None else float(lat_step)
    lon_step = settings.MESH_LON_STEP if lon_step is None else float(lon_step)

    if "lat" not in df.columns or "lng" not in df.columns:
        raise KeyError("lat および lng カラムが必要です")

    out = df.copy()
    lat = pd.to_numeric(out["lat"], errors="coerce")
    lng = pd.to_numeric(out["lng"], errors="coerce")

    lat_bin = np.floor(lat / lat_step)
    lon_bin = np.floor(lng / lon_step)

    out["mesh_code"] = (
        lat_bin.fillna(-1).astype(int).astype(str)
        + "_"
        + lon_bin.fillna(-1).astype(int).astype(str)
    )
    return out


def aggregate_by_mesh_genre(df: pd.DataFrame) -> pd.DataFrame:
    """メッシュ × ジャンル単位で店舗数・平均評価等を集約する。

    Parameters
    ----------
    df : pd.DataFrame
        メッシュコード・ジャンル付きデータ

    Returns
    -------
    pd.DataFrame
        集約済みデータ（1行 = 1メッシュ×1ジャンル）
    """
    out = map_genre(df)
    if "mesh_code" not in out.columns:
        out = assign_mesh_code(out)

    out["_rating"] = pd.to_numeric(
        _coalesce_column(out, ("rating", "avg_rating"), 0), errors="coerce"
    ).fillna(0.0)
    out["_review_count"] = pd.to_numeric(
        _coalesce_column(out, ("review_count", "user_ratings_total", "reviews"), 0), errors="coerce"
    ).fillna(0.0)

    if "lat" not in out.columns:
        out["lat"] = np.nan
    if "lng" not in out.columns:
        out["lng"] = np.nan

    grouped = (
        out.groupby(["mesh_code", "unified_genre"], dropna=False)
        .agg(
            restaurant_count=("mesh_code", "size"),
            avg_rating=("_rating", "mean"),
            review_count=("_review_count", "sum"),
            lat=("lat", "mean"),
            lng=("lng", "mean"),
        )
        .reset_index()
    )
    return grouped


def save_processed(df: pd.DataFrame, filename: str) -> None:
    """加工済みデータを保存する。

    Parameters
    ----------
    df : pd.DataFrame
        保存対象
    filename : str
        保存ファイル名（data/processed/ 配下に保存）
    """
    settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    target = Path(filename)
    if target.suffix.lower() != ".csv":
        target = target.with_suffix(".csv")

    output_path = settings.PROCESSED_DATA_DIR / target.name
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("加工済みデータ保存: %s", output_path)
