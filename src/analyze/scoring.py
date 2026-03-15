"""市場機会スコアの算出と推薦生成を行うモジュール。

メッシュ単位に集計された店舗データから需要スコア、競合密度、機会スコアを計算し、
上位候補に対して説明文を付与する。
"""

from __future__ import annotations

import logging

import pandas as pd

from config.settings import WEIGHT_COMPETITOR, WEIGHT_DEMAND, WEIGHT_POPULATION

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


def compute_demand_score_v2(df: pd.DataFrame) -> pd.Series:
    """e-Stat 統合後の人口系指標を優先して需要スコアを計算する。

    `daytime_population` があれば昼間人口を、なければ `population` を利用する。
    どちらも存在しない場合のみ、従来互換のため `restaurant_count` を代替需要指標として
    利用し、循環性が残ることを警告ログに出力する。

    Args:
        df: 集計済み DataFrame。

    Returns:
        `demand_score` という名前の正規化済み系列。
    """
    logger.info("Computing demand score v2 for %s rows", len(df))

    if df.empty:
        return pd.Series(dtype=float, index=df.index, name="demand_score")

    if "daytime_population" in df.columns:
        demand_raw = pd.to_numeric(df["daytime_population"], errors="coerce").fillna(0.0)
    elif "population" in df.columns:
        demand_raw = pd.to_numeric(df["population"], errors="coerce").fillna(0.0)
    else:
        logger.warning(
            "daytime_population and population are missing; "
            "falling back to restaurant_count for demand score v2"
        )
        demand_raw = pd.to_numeric(df["restaurant_count"], errors="coerce").fillna(0.0)

    return _normalize(demand_raw).rename("demand_score")


def compute_opportunity_score_v2(df: pd.DataFrame) -> pd.DataFrame:
    """人口補正済み競合密度を用いて機会スコアを算出する v2 実装。

    需要は `compute_demand_score_v2` で計算し、競合密度は人口 1 万人あたりの
    飲食店数として扱う。人口がない場合は従来互換のため `restaurant_count` をそのまま
    競合密度として利用する。

    Args:
        df: 集計済み店舗データ。

    Returns:
        `demand_score`、`competitor_density`、`opportunity_score` を追加した
        DataFrame のコピー。
    """
    logger.info("Computing opportunity score v2 for %s rows", len(df))

    out = df.copy()
    if out.empty:
        out["demand_score"] = pd.Series(dtype=float, index=out.index)
        out["competitor_density"] = pd.Series(dtype=float, index=out.index)
        out["opportunity_score"] = pd.Series(dtype=float, index=out.index)
        return out

    demand_score = compute_demand_score_v2(out)
    restaurant_count = pd.to_numeric(out["restaurant_count"], errors="coerce").fillna(0.0)

    if "population" in out.columns:
        population = pd.to_numeric(out["population"], errors="coerce").fillna(0.0)
        competitor_density = restaurant_count / (population / 10000.0 + _COMPETITOR_OFFSET)
        population_signal = _normalize(population)
    else:
        competitor_density = restaurant_count
        if "daytime_population" in out.columns:
            population_signal = _normalize(
                pd.to_numeric(out["daytime_population"], errors="coerce").fillna(0.0)
            )
        else:
            population_signal = pd.Series(1.0, index=out.index, dtype=float)

    opportunity_raw = (
        demand_score * WEIGHT_DEMAND + population_signal * WEIGHT_POPULATION
    ) / (competitor_density * WEIGHT_COMPETITOR + _COMPETITOR_OFFSET)

    out["demand_score"] = demand_score
    out["competitor_density"] = competitor_density.rename("competitor_density")
    out["opportunity_score"] = _normalize(opportunity_raw)
    return out


def run_scoring_v2(aggregated_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """v2 スコアリングと推薦生成をまとめて実行する。

    Args:
        aggregated_df: e-Stat 統合済みの集計 DataFrame。
        top_n: 返却する上位推薦件数。

    Returns:
        v2 機会スコア計算済みかつ上位推薦に絞り込んだ DataFrame。
    """
    logger.info("Running scoring v2 pipeline for %s rows", len(aggregated_df))
    scored_df = compute_opportunity_score_v2(aggregated_df)
    return get_top_recommendations(scored_df, top_n=top_n)


def compute_opportunity_score_v3(df: pd.DataFrame) -> pd.DataFrame:
    """空間特徴量を組み込んだ v3 機会スコアを算出する。

    v2 の人口ベーススコアに加えて、ジャンル多様性・隣接メッシュ競合・
    飽和度・ジャンル集中度を加味する。

    スコア = (需要 + 人口シグナル + 多様性ボーナス + ジャンル空白ボーナス)
             / (競合密度 + 隣接競合圧 + 飽和ペナルティ)

    Args:
        df: 空間特徴量追加済みの集計データ。

    Returns:
        `opportunity_score` を含む DataFrame。
    """
    logger.info("Computing opportunity score v3 for %d rows", len(df))

    out = df.copy()
    if out.empty:
        for col in ("demand_score", "competitor_density", "opportunity_score"):
            out[col] = pd.Series(dtype=float, index=out.index)
        return out

    # 需要スコア（v2と同じ）
    demand_score = compute_demand_score_v2(out)

    # 人口シグナル
    restaurant_count = pd.to_numeric(out["restaurant_count"], errors="coerce").fillna(0.0)
    if "population" in out.columns:
        population = pd.to_numeric(out["population"], errors="coerce").fillna(0.0)
        competitor_density = restaurant_count / (population / 10000.0 + _COMPETITOR_OFFSET)
        population_signal = _normalize(population)
    else:
        competitor_density = restaurant_count
        population_signal = pd.Series(1.0, index=out.index, dtype=float)

    # 空間特徴量ボーナス / ペナルティ
    diversity = _normalize(
        pd.to_numeric(out.get("genre_diversity"), errors="coerce").fillna(0.0)
    ) if "genre_diversity" in out.columns else pd.Series(0.0, index=out.index)

    genre_hhi = pd.to_numeric(out.get("genre_hhi"), errors="coerce").fillna(0.0) \
        if "genre_hhi" in out.columns else pd.Series(0.0, index=out.index)
    # HHI が高い = ジャンル偏り = 他ジャンルにとっては参入機会
    genre_gap_bonus = _normalize(genre_hhi)

    neighbor_pressure = _normalize(
        pd.to_numeric(out.get("neighbor_avg_restaurants"), errors="coerce").fillna(0.0)
    ) if "neighbor_avg_restaurants" in out.columns else pd.Series(0.0, index=out.index)

    saturation = _normalize(
        pd.to_numeric(out.get("saturation_index"), errors="coerce").fillna(0.0)
    ) if "saturation_index" in out.columns else pd.Series(0.0, index=out.index)

    # v3 スコア合成
    numerator = (
        demand_score * WEIGHT_DEMAND
        + population_signal * WEIGHT_POPULATION
        + diversity * 0.3
        + genre_gap_bonus * 0.2
    )
    denominator = (
        competitor_density * WEIGHT_COMPETITOR
        + neighbor_pressure * 0.2
        + saturation * 0.3
        + _COMPETITOR_OFFSET
    )
    opportunity_raw = numerator / denominator

    out["demand_score"] = demand_score
    out["competitor_density"] = competitor_density
    out["opportunity_score"] = _normalize(opportunity_raw)
    return out


def run_scoring_v3(aggregated_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """v3 スコアリング（空間特徴量込み）と推薦生成を実行する。

    Args:
        aggregated_df: 空間特徴量追加済みの集計 DataFrame。
        top_n: 返却する上位推薦件数。

    Returns:
        v3 機会スコア計算済みかつ上位推薦に絞り込んだ DataFrame。
    """
    logger.info("Running scoring v3 pipeline for %d rows", len(aggregated_df))
    scored_df = compute_opportunity_score_v3(aggregated_df)
    return get_top_recommendations(scored_df, top_n=top_n)


# ──────────────────────────────────────────────────────────────────────
# フランチャイズ出店判断モデルへの拡張ロードマップ
# ──────────────────────────────────────────────────────────────────────
# 1. 外部データ: 国勢調査・経済センサス・商業動態統計、駅別乗降客数、携帯位置情報、
#    クレジット決済推計、観光流動、求人件数、地価公示、オフィス就業人口などを統合する。
# 2. 空間特徴量: 最寄り駅距離、駅ランク、商業施設距離、競合同業のクラスター強度、
#    導線上の視認性、道路幅員、昼夜人口比、商圏内 POI 密度を加える。
# 3. 時系列要素: 人口増減、昼間人口の季節変動、開店・閉店純増、レビュー件数推移、
#    賃料上昇率、再開発計画の進捗を時系列特徴として持たせる。
# 4. モデル候補: ロジスティック回帰を基準線に、勾配ブースティング、ランダムフォレスト、
#    XGBoost/LightGBM、空間回帰、時系列を含むなら Temporal Fusion Transformer などを比較する。
# 5. 評価指標: 既存店の売上・継続率・投資回収期間との相関、上位候補の hit rate、
#    地域外汎化性能、時系列ホールドアウトでの再現率を確認する。
