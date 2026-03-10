"""市場機会スコアの算出と推薦生成を行うモジュール。

メッシュ単位に集計された店舗データから需要スコア、競合密度、機会スコアを計算し、
上位候補に対して説明文を付与する。
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_COMPETITOR_OFFSET = 0.1


def _normalize(series: pd.Series) -> pd.Series:
    """数値系列を 0 から 1 の範囲へ正規化する。

    欠損値や非数値は 0 として扱い、全値が同一の場合はすべて 0 を返す。

    Args:
        series: 正規化対象の系列。

    Returns:
        0 以上 1 以下に正規化された浮動小数点系列。
    """
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if values.empty:
        return pd.Series(dtype=float, index=series.index)

    min_value = float(values.min())
    max_value = float(values.max())
    if min_value == max_value:
        return pd.Series(0.0, index=series.index, dtype=float)

    return ((values - min_value) / (max_value - min_value)).astype(float)


def compute_demand_score(df: pd.DataFrame) -> pd.Series:
    """メッシュ全体の店舗数に基づく需要スコアを計算する。

    同一メッシュ内の `restaurant_count` を合計し、その値を正規化して
    `demand_score` として返す。

    Args:
        df: `mesh_code` と `restaurant_count` を含む集計済み DataFrame。

    Returns:
        元のインデックスに揃えた需要スコア系列。
    """
    logger.info("Computing demand score for %s rows", len(df))

    if df.empty:
        return pd.Series(dtype=float, index=df.index, name="demand_score")

    mesh_total_count = (
        pd.to_numeric(df["restaurant_count"], errors="coerce")
        .fillna(0.0)
        .groupby(df["mesh_code"])
        .transform("sum")
    )
    demand_raw = mesh_total_count

    return _normalize(demand_raw).rename("demand_score")


def compute_opportunity_score(df: pd.DataFrame) -> pd.DataFrame:
    """需要と競合密度から機会スコアを算出する。

    需要スコアを計算し、ジャンル別の競合店舗数で割ることで生の機会値を作成する。
    その後、0 から 1 の範囲に再正規化して `opportunity_score` 列へ格納する。

    Args:
        df: 集計済み店舗データ。

    Returns:
        `demand_score`、`competitor_density`、`opportunity_score` を追加した
        DataFrame のコピー。
    """
    logger.info("Computing opportunity score for %s rows", len(df))

    out = df.copy()
    if out.empty:
        out["demand_score"] = pd.Series(dtype=float, index=out.index)
        out["competitor_density"] = pd.Series(dtype=float, index=out.index)
        out["opportunity_score"] = pd.Series(dtype=float, index=out.index)
        return out

    demand_score = compute_demand_score(out)
    competitor_density = pd.to_numeric(
        out["restaurant_count"], errors="coerce"
    ).fillna(0.0)
    opportunity_raw = demand_score / (competitor_density + _COMPETITOR_OFFSET)

    out["demand_score"] = demand_score
    out["competitor_density"] = competitor_density
    out["opportunity_score"] = _normalize(opportunity_raw)
    return out


def rank_opportunities(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """機会スコアの高い候補を上位順に返す。

    Args:
        df: `opportunity_score` 列を含む DataFrame。
        top_n: 返却する上位件数。

    Returns:
        機会スコア降順に並べた先頭 `top_n` 件の DataFrame。

    Raises:
        KeyError: `opportunity_score` 列が存在しない場合。
    """
    if "opportunity_score" not in df.columns:
        raise KeyError("opportunity_score column is required")
    if top_n <= 0:
        return df.iloc[0:0].copy()

    logger.info("Ranking top %s opportunities from %s rows", top_n, len(df))
    return (
        df.sort_values("opportunity_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def generate_reason(row: pd.Series) -> str:
    """単一候補の機会スコア説明文を生成する。

    需要スコアと競合密度をしきい値でラベル化し、メッシュコードとジャンル名を
    含む人間向けの要約文字列を返す。

    Args:
        row: 推薦候補 1 行分の系列。

    Returns:
        推薦理由を表す日本語文字列。
    """
    mesh_code = str(row.get("mesh_code", "unknown_mesh"))
    unified_genre = str(row.get("unified_genre", "unknown_genre"))
    opportunity_score = float(
        pd.to_numeric(pd.Series([row.get("opportunity_score", 0.0)]), errors="coerce")
        .fillna(0.0)
        .iloc[0]
    )
    demand_score = float(
        pd.to_numeric(pd.Series([row.get("demand_score", 0.0)]), errors="coerce")
        .fillna(0.0)
        .iloc[0]
    )
    competitor_density = int(
        pd.to_numeric(
            pd.Series([row.get("competitor_density", row.get("restaurant_count", 0))]),
            errors="coerce",
        )
        .fillna(0)
        .iloc[0]
    )

    if demand_score >= 0.7:
        demand_label = "高い"
    elif demand_score >= 0.4:
        demand_label = "中程度"
    else:
        demand_label = "低い"

    if competitor_density <= 3:
        competitor_label = "少ない"
    elif competitor_density <= 10:
        competitor_label = "中程度"
    else:
        competitor_label = "多い"

    return (
        f"【{mesh_code} × {unified_genre}】"
        f"機会スコア {opportunity_score:.3f} / "
        f"需要: {demand_label}({demand_score:.2f}) / "
        f"競合: {competitor_label}({competitor_density}店)"
    )


def get_top_recommendations(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """上位候補を抽出し、推薦理由を付与して返す。

    Args:
        df: スコア計算済み DataFrame。
        top_n: 取得する上位件数。

    Returns:
        上位候補に `reason` 列を追加した DataFrame。
    """
    logger.info("Generating top %s recommendations", top_n)

    ranked = rank_opportunities(df, top_n=top_n)
    if ranked.empty:
        ranked["reason"] = pd.Series(dtype=str, index=ranked.index)
        return ranked

    ranked["reason"] = ranked.apply(generate_reason, axis=1)
    return ranked


def run_scoring(aggregated_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """スコアリングと推薦生成をまとめて実行する。

    Args:
        aggregated_df: メッシュ・ジャンル単位に集計済みの DataFrame。
        top_n: 返却する上位推薦件数。

    Returns:
        機会スコア計算済みかつ上位推薦に絞り込んだ DataFrame。
    """
    logger.info("Running scoring pipeline for %s rows", len(aggregated_df))
    scored_df = compute_opportunity_score(aggregated_df)
    return get_top_recommendations(scored_df, top_n=top_n)
