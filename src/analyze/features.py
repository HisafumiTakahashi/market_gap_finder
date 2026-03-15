"""空間特徴量エンジニアリングモジュール。

メッシュ単位の集計データに対して、ジャンル多様性・隣接メッシュ競合・
ジャンル集中度（HHI）・人口飽和度などの空間特徴量を追加する。
"""

from __future__ import annotations

import logging

import pandas as pd

from src.preprocess.mesh_converter import mesh3_to_lat_lon

logger = logging.getLogger(__name__)


def _neighbor_mesh_codes(mesh3: str) -> list[str]:
    """3次メッシュコードの8近傍メッシュコードを返す。

    3次メッシュの末尾2桁 (s, t) を ±1 して隣接を算出する。
    s: 0-9 (緯度方向), t: 0-9 (経度方向)
    """
    mesh3 = str(mesh3)
    if len(mesh3) != 8 or not mesh3.isdigit():
        return []

    prefix = mesh3[:6]
    s = int(mesh3[6])
    t = int(mesh3[7])
    neighbors = []

    for ds in (-1, 0, 1):
        for dt in (-1, 0, 1):
            if ds == 0 and dt == 0:
                continue
            ns, nt = s + ds, t + dt
            if 0 <= ns <= 9 and 0 <= nt <= 9:
                neighbors.append(f"{prefix}{ns}{nt}")

    return neighbors


def add_genre_diversity(df: pd.DataFrame) -> pd.DataFrame:
    """メッシュ内のジャンル数を特徴量として追加する。

    商業集積が進んでいるエリアほどジャンルが多様になる傾向がある。
    """
    if df.empty or "jis_mesh3" not in df.columns:
        df = df.copy()
        df["genre_diversity"] = 0
        return df

    mesh_genre_count = (
        df.groupby("jis_mesh3")["unified_genre"]
        .nunique()
        .rename("genre_diversity")
    )
    return df.merge(mesh_genre_count, on="jis_mesh3", how="left")


def add_genre_hhi(df: pd.DataFrame) -> pd.DataFrame:
    """メッシュ内のジャンル集中度（HHI）を追加する。

    HHI が高い = 特定ジャンルに偏っている = 他ジャンル参入余地あり。
    HHI が低い = 多様なジャンルが均等に存在。
    """
    if df.empty or "jis_mesh3" not in df.columns:
        df = df.copy()
        df["genre_hhi"] = 0.0
        return df

    mesh_total = df.groupby("jis_mesh3")["restaurant_count"].transform("sum")
    share = df["restaurant_count"] / mesh_total.replace(0, 1)
    df = df.copy()
    df["_share_sq"] = share ** 2
    hhi = df.groupby("jis_mesh3")["_share_sq"].sum().rename("genre_hhi")
    df = df.drop(columns=["_share_sq"]).merge(hhi, on="jis_mesh3", how="left")
    return df


def add_neighbor_competition(df: pd.DataFrame) -> pd.DataFrame:
    """隣接メッシュの平均店舗数を特徴量として追加する。

    周辺に競合が多い = 需要が高いエリアだが飽和リスクもある。
    """
    if df.empty or "jis_mesh3" not in df.columns:
        df = df.copy()
        df["neighbor_avg_restaurants"] = 0.0
        return df

    mesh_total = (
        df.groupby("jis_mesh3")["restaurant_count"]
        .sum()
        .to_dict()
    )

    def _calc_neighbor_avg(mesh3: str) -> float:
        neighbors = _neighbor_mesh_codes(mesh3)
        if not neighbors:
            return 0.0
        counts = [mesh_total.get(n, 0) for n in neighbors]
        return sum(counts) / len(counts)

    unique_meshes = df["jis_mesh3"].dropna().unique()
    neighbor_map = {m: _calc_neighbor_avg(m) for m in unique_meshes}
    df = df.copy()
    df["neighbor_avg_restaurants"] = df["jis_mesh3"].map(neighbor_map).fillna(0.0)
    return df


def add_saturation_index(df: pd.DataFrame) -> pd.DataFrame:
    """人口あたり総店舗数（飽和度）をメッシュ単位で追加する。

    人口1万人あたりの全ジャンル合計店舗数。高いほど競争が激しい。
    """
    if df.empty or "jis_mesh3" not in df.columns:
        df = df.copy()
        df["saturation_index"] = 0.0
        return df

    mesh_agg = df.groupby("jis_mesh3").agg(
        total_restaurants=("restaurant_count", "sum"),
        population=("population", "first"),
    )
    pop = mesh_agg["population"].fillna(0).clip(lower=0)
    mesh_agg["saturation_index"] = mesh_agg["total_restaurants"] / (pop / 10000 + 0.1)
    sat = mesh_agg["saturation_index"]

    return df.merge(sat, on="jis_mesh3", how="left")


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """全空間特徴量を一括追加する。"""
    logger.info("空間特徴量を追加します: %d 行", len(df))
    out = df
    out = add_genre_diversity(out)
    out = add_genre_hhi(out)
    out = add_neighbor_competition(out)
    out = add_saturation_index(out)
    logger.info("特徴量追加完了: 列数 %d → %d", len(df.columns), len(out.columns))
    return out
