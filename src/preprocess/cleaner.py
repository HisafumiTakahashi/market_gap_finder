"""収集済み店舗データの前処理を行うモジュール。

Hotpepper から取得した生データを読み込み、ジャンル正規化やメッシュコード付与、
メッシュ単位の集計を行って分析しやすい中間データを生成する。
"""

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
    """生データ CSV から Hotpepper 収集結果を読み込む。

    Args:
        tag: 入出力ファイル名に使用するタグ。

    Returns:
        読み込んだ店舗データの DataFrame。

    Raises:
        FileNotFoundError: 対象 CSV が存在しない場合。
    """
    file_path = settings.RAW_DATA_DIR / f"{tag}_hotpepper.csv"
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    logger.info("Loading Hotpepper data from %s", file_path)
    return pd.read_csv(file_path)


def map_genre(df: pd.DataFrame) -> pd.DataFrame:
    """`genre` 列を正規化して統一ジャンル列を追加する。

    `GENRE_MAPPING` に定義した部分一致ルールに従ってジャンル名を
    `unified_genre` 列へ変換し、該当しない値や欠損値は `other` とする。

    Args:
        df: `genre` 列を含む店舗データ。

    Returns:
        `unified_genre` 列を追加した DataFrame のコピー。

    Raises:
        KeyError: `genre` 列が存在しない場合。
    """
    if "genre" not in df.columns:
        raise KeyError("genre column is required")

    def normalize_genre(value: object) -> str:
        """単一のジャンル文字列を統一ジャンル名へ変換する。"""
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
    """緯度経度から独自メッシュコードを算出して付与する。

    レガシー関数。現パイプラインでは JIS 標準 4 分の 1 メッシュ（10 桁）を
    使用するため、``mesh_converter.assign_jis_mesh_quarter`` を推奨する。

    緯度・経度を数値化したうえで設定済みの収集グリッド刻み幅ごとにビン分けし、
    `lat_bin_lng_bin` 形式の文字列を `mesh_code` 列として追加する。
    緯度経度が欠損または数値化できない行には `unknown` を設定する。

    Args:
        df: `lat` 列と `lng` 列を含む DataFrame。

    Returns:
        `mesh_code` 列を追加した DataFrame のコピー。

    Raises:
        KeyError: `lat` または `lng` 列が存在しない場合。
    """
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
    """メッシュコードと統一ジャンル単位で店舗データを集計する。

    前処理としてジャンル正規化とメッシュコード付与を行い、その後
    店舗数・平均評価・レビュー総数・平均座標を算出する。

    Args:
        df: 生の店舗一覧 DataFrame。

    Returns:
        `mesh_code` と `unified_genre` ごとに集約した DataFrame。
    """
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
    """前処理済み DataFrame を CSV として保存する。

    Args:
        df: 保存対象の DataFrame。
        filename: `PROCESSED_DATA_DIR` 配下に保存するファイル名。
    """
    settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = settings.PROCESSED_DATA_DIR / filename
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("Saved processed data to %s", output_path)


def run_preprocess(tag: str = "result") -> pd.DataFrame:
    """前処理パイプライン全体を実行する。

    生データの読み込み、メッシュ・ジャンル単位の集計、集計結果の保存までを
    一括で行う。

    Args:
        tag: 対象ファイルのタグ。

    Returns:
        集計済み DataFrame。
    """
    hotpepper_df = load_hotpepper(tag)
    aggregated_df = aggregate_by_mesh_genre(hotpepper_df)
    save_processed(aggregated_df, f"{tag}_aggregated.csv")
    return aggregated_df
