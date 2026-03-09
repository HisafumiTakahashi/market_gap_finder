"""
機会スコア算出モジュール

「需要が高いにもかかわらず競合が少ない」エリア×ジャンルの組み合わせに
高いスコアを付与し、出店候補をランキングする。

スコア設計
----------
demand_score       = normalize(口コミ数 × 平均評価 × 混雑度インデックス)
competitor_density = 競合店舗数 / 人口1万人
opportunity_score  = w_demand  * demand_score / (competitor_density + 0.01)
                   * w_population * population_density_norm
                   * w_land     * (1 / (land_price_norm + ε))
最終的に MinMaxScaler で 0-1 に正規化。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

_LAPLACE = 0.01   # 競合密度のラプラス補正
_EPSILON = 1e-6   # 地価ゼロ除算防止


# ──────────────────────────────────────
# ユーティリティ
# ──────────────────────────────────────

def _normalize(series: pd.Series) -> pd.Series:
    """MinMax 正規化（0–1）。全値が同一の場合は 0.0 で埋める。"""
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    min_v, max_v = float(values.min()), float(values.max())
    if max_v <= min_v:
        return pd.Series(0.0, index=series.index, dtype=float)
    return (values - min_v) / (max_v - min_v)


def _first_numeric(
    df: pd.DataFrame,
    candidates: tuple[str, ...],
    default: float = 0.0,
) -> pd.Series:
    """候補カラムを順番に探して数値 Series を返す。見つからなければ default で埋める。"""
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


# ──────────────────────────────────────
# 因子算出
# ──────────────────────────────────────

def compute_demand_score(df: pd.DataFrame) -> pd.Series:
    """需要スコアを算出する（0-1 正規化済み）。

    demand_score = normalize(口コミ数 × 平均評価 × 混雑度インデックス)

    Parameters
    ----------
    df : pd.DataFrame
        エリア×ジャンル集約済みデータ

    Returns
    -------
    pd.Series
        正規化された demand_score (0–1), 名前='demand_score'
    """
    if df.empty:
        return pd.Series(dtype=float, index=df.index, name="demand_score")

    review_count = _first_numeric(df, ("review_count", "user_ratings_total", "reviews"), 0.0)
    rating = _first_numeric(df, ("avg_rating", "rating", "score"), 0.0)
    congestion = _first_numeric(df, ("congestion_index", "congestion", "busy_level"), 1.0)

    raw = review_count * rating * congestion
    return _normalize(raw).rename("demand_score")


def compute_competitor_density(df: pd.DataFrame) -> pd.Series:
    """競合密度を算出する（競合店舗数 / 人口1万人）。

    Parameters
    ----------
    df : pd.DataFrame
        エリア×ジャンル集約済みデータ

    Returns
    -------
    pd.Series
        competitor_density（単位: 店/万人）, 名前='competitor_density'
    """
    if df.empty:
        return pd.Series(dtype=float, index=df.index, name="competitor_density")

    store_count = _first_numeric(
        df, ("restaurant_count", "store_count", "count", "n_restaurants"), 0.0
    )

    # population_10k があればそのまま使用、population（生値）の場合は1万人単位に変換
    if "population_10k" in df.columns:
        pop_10k = pd.to_numeric(df["population_10k"], errors="coerce").fillna(1.0).clip(lower=_EPSILON)
    elif "population" in df.columns:
        pop_10k = (pd.to_numeric(df["population"], errors="coerce").fillna(10_000.0) / 10_000.0).clip(lower=_EPSILON)
    else:
        pop_10k = pd.Series(1.0, index=df.index, dtype=float)

    return (store_count / pop_10k).rename("competitor_density")


# ──────────────────────────────────────
# 機会スコア
# ──────────────────────────────────────

def compute_opportunity_score(
    df: pd.DataFrame,
    w_demand: float | None = None,
    w_competitor: float | None = None,
    w_population: float | None = None,
    w_land_price: float | None = None,
) -> pd.DataFrame:
    """機会スコアを算出して DataFrame に付与する。

    opportunity_score_raw = w_demand * demand_score / (competitor_density + 0.01)
                          * w_population * population_density_norm
                          * w_land_price * (1 / (land_price_norm + ε))

    最終スコアは MinMaxScaler で 0-1 に正規化される。

    Parameters
    ----------
    df : pd.DataFrame
        エリア×ジャンル集約済みデータ
    w_demand : float | None
        需要スコアの線形重み（None なら settings.WEIGHT_DEMAND）
    w_competitor : float | None
        競合密度の線形重み（None なら settings.WEIGHT_COMPETITOR）
    w_population : float | None
        人口密度の線形重み（None なら settings.WEIGHT_POPULATION）
    w_land_price : float | None
        地価逆数の線形重み（None なら settings.WEIGHT_LAND_PRICE）

    Returns
    -------
    pd.DataFrame
        中間スコアカラムと opportunity_score カラムが追加されたデータ
    """
    w_demand = float(getattr(settings, "WEIGHT_DEMAND", 1.0)) if w_demand is None else float(w_demand)
    w_competitor = float(getattr(settings, "WEIGHT_COMPETITOR", 1.0)) if w_competitor is None else float(w_competitor)
    w_population = float(getattr(settings, "WEIGHT_POPULATION", 1.0)) if w_population is None else float(w_population)
    w_land_price = float(getattr(settings, "WEIGHT_LAND_PRICE", 1.0)) if w_land_price is None else float(w_land_price)

    out = df.copy()

    demand_score = compute_demand_score(out)
    competitor_density = compute_competitor_density(out)
    pop_density_norm = _normalize(
        _first_numeric(out, ("population_density", "pop_density", "population_10k", "population"), 0.0)
    )
    land_price_norm = _normalize(
        _first_numeric(out, ("land_price", "land_price_index", "地価"), 0.0)
    )

    out["demand_score"] = demand_score
    out["competitor_density"] = competitor_density
    out["population_density_norm"] = pop_density_norm
    out["land_price_norm"] = land_price_norm

    raw_score = (
        w_demand * demand_score
        / (competitor_density + _LAPLACE)
        * w_population * pop_density_norm
        * w_land_price * (1.0 / (land_price_norm + _EPSILON))
    )

    out["opportunity_score"] = _normalize(raw_score)
    return out


# ──────────────────────────────────────
# ランキング・推薦
# ──────────────────────────────────────

def rank_opportunities(
    df: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """機会スコア上位の出店候補を返す。

    Parameters
    ----------
    df : pd.DataFrame
        opportunity_score 付きデータ
    top_n : int
        上位何件を返すか（デフォルト 20）

    Returns
    -------
    pd.DataFrame
        上位 top_n 件のランキング（インデックスをリセット済み）
    """
    if "opportunity_score" not in df.columns:
        raise KeyError("opportunity_score カラムが必要です")
    if top_n <= 0:
        return df.iloc[0:0].copy()

    return (
        df.sort_values("opportunity_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def generate_reason(row: pd.Series) -> str:
    """1件の推薦行から根拠テキストを自動生成する。

    Parameters
    ----------
    row : pd.Series
        rank_opportunities の各行

    Returns
    -------
    str
        日本語の根拠テキスト
    """
    area = row.get("area", row.get("mesh_id", "対象エリア"))
    genre = row.get("genre", row.get("genre_code", "飲食"))
    score = row.get("opportunity_score", 0.0)

    parts = [f"【{area} × {genre}】 機会スコア {score:.3f}"]

    demand = row.get("demand_score")
    if demand is not None:
        level = "非常に高い" if demand >= 0.7 else "中程度" if demand >= 0.4 else "低い"
        parts.append(f"需要: {level}（{demand:.2f}）")

    density = row.get("competitor_density")
    if density is not None:
        level = "少ない" if density < 1.0 else "中程度" if density < 3.0 else "多い"
        parts.append(f"競合: {level}（{density:.2f}店/万人）")

    pop = row.get("population_density_norm")
    if pop is not None:
        level = "高密度" if pop >= 0.7 else "中密度" if pop >= 0.4 else "低密度"
        parts.append(f"人口: {level}（正規化値 {pop:.2f}）")

    land = row.get("land_price_norm")
    if land is not None:
        level = "低コスト" if land < 0.4 else "中程度" if land < 0.7 else "高コスト"
        parts.append(f"地価: {level}（正規化値 {land:.2f}）")

    return " / ".join(parts)


def get_top_recommendations(
    df: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """機会スコア上位 top_n 件に根拠テキストを付与して返す。

    Parameters
    ----------
    df : pd.DataFrame
        opportunity_score 付きデータ
    top_n : int
        上位何件を返すか（デフォルト 20）

    Returns
    -------
    pd.DataFrame
        上位 top_n 件 + reason カラム
    """
    ranked = rank_opportunities(df, top_n=top_n)
    if ranked.empty:
        ranked["reason"] = pd.Series(dtype=str)
        return ranked
    ranked["reason"] = ranked.apply(generate_reason, axis=1)
    return ranked
