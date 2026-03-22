#!/usr/bin/env python3
"""スコアリング結果の検証・比較スクリプト。

v1/v2/v3 のスコアを並列計算し、順位相関・特徴量相関・上位候補の妥当性を分析する。
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from config import settings
from src.analyze.features import add_all_features
from src.analyze.scoring import (
    compute_opportunity_score,
    compute_opportunity_score_v2,
    compute_opportunity_score_v3,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="スコア検証・比較")
    parser.add_argument("--tag", type=str, default="tokyo")
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def load_integrated(tag: str) -> pd.DataFrame:
    """統合済みデータを読み込む。"""
    path = settings.PROCESSED_DATA_DIR / f"{tag}_integrated.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} が見つかりません。先に integrate_estat.py を実行してください。")
    return pd.read_csv(path)


def compute_all_scores(df: pd.DataFrame) -> pd.DataFrame:
    """v1/v2/v3 のスコアを並列計算して1つの DataFrame にまとめる。

    統合済みCSVに既に空間特徴量が含まれている場合はそのまま利用する。
    """
    # v1 用のベースカラム
    base_cols = ["jis_mesh", "jis_mesh3", "unified_genre", "restaurant_count", "population",
                 "avg_rating", "total_reviews", "lat", "lng", "mesh_code"]
    clean = df[[c for c in base_cols if c in df.columns]].copy()

    # v1
    v1 = compute_opportunity_score(clean.copy())
    result = clean.copy()
    result["score_v1"] = v1["opportunity_score"].values

    # v2
    v2 = compute_opportunity_score_v2(clean.copy())
    result["score_v2"] = v2["opportunity_score"].values

    # v3: 統合済みCSVに空間特徴量があればそのまま利用
    feature_cols = ["genre_diversity", "genre_hhi", "neighbor_avg_restaurants",
                    "saturation_index", "nearest_station_distance", "land_price"]
    has_features = all(c in df.columns for c in feature_cols[:4])

    if has_features:
        featured = df.copy()
    else:
        featured = add_all_features(clean.copy())

    v3 = compute_opportunity_score_v3(featured)
    result["score_v3"] = v3["opportunity_score"].values

    # 特徴量もコピー
    for col in feature_cols:
        if col in v3.columns:
            result[col] = v3[col].values

    return result


def rank_correlation(s1: pd.Series, s2: pd.Series) -> float:
    """スピアマン順位相関係数を計算する。"""
    corr = s1.corr(s2, method="spearman")
    return 0.0 if pd.isna(corr) else float(corr)


def analyze_version_comparison(df: pd.DataFrame, top_n: int) -> None:
    """バージョン間のスコア比較を出力する。"""
    print("=" * 60)
    print("1. バージョン間スコア相関（スピアマン順位相関）")
    print("=" * 60)
    pairs = [("v1", "v2"), ("v2", "v3"), ("v1", "v3")]
    for a, b in pairs:
        rho = rank_correlation(df[f"score_{a}"], df[f"score_{b}"])
        print(f"  {a} vs {b}: ρ = {rho:.4f}")
    print()

    print("=" * 60)
    print("2. 各バージョンの上位候補（Top-N）")
    print("=" * 60)
    for ver in ["v1", "v2", "v3"]:
        col = f"score_{ver}"
        mesh_col = "jis_mesh" if "jis_mesh" in df.columns else "jis_mesh3"
        top = df.nlargest(top_n, col)[[mesh_col, "unified_genre", "restaurant_count", "population", col]]
        print(f"\n--- {ver} Top {top_n} ---")
        for rank, (_, row) in enumerate(top.iterrows(), 1):
            pop = int(row.get("population", 0))
            rest = int(row["restaurant_count"])
            print(f"  {rank:2d}. {row[mesh_col]} x {row['unified_genre']:10s} | "
                  f"score={row[col]:.4f} | pop={pop:,} | shops={rest}")
    print()


def analyze_feature_correlation(df: pd.DataFrame) -> None:
    """特徴量とスコアの相関を分析する。"""
    print("=" * 60)
    print("3. 特徴量 × スコア相関（ピアソン）")
    print("=" * 60)
    features = ["population", "restaurant_count", "genre_diversity",
                "genre_hhi", "neighbor_avg_restaurants", "saturation_index"]
    scores = ["score_v1", "score_v2", "score_v3"]

    header = f"{'feature':>28s} | " + " | ".join(f"{s:>10s}" for s in scores)
    print(header)
    print("-" * len(header))
    for feat in features:
        if feat not in df.columns:
            continue
        vals = []
        for sc in scores:
            corr = df[feat].corr(df[sc])
            vals.append(f"{corr:>10.4f}" if not pd.isna(corr) else f"{'N/A':>10s}")
        print(f"{feat:>28s} | " + " | ".join(vals))
    print()


def analyze_top_candidate_quality(df: pd.DataFrame, top_n: int) -> None:
    """上位候補の妥当性チェック（低人口・高飽和の異常候補を検出）。"""
    print("=" * 60)
    print("4. v3 上位候補の妥当性チェック")
    print("=" * 60)
    top = df.nlargest(top_n, "score_v3").copy()

    # 低人口チェック
    low_pop = top[top["population"] < 10000]
    if not low_pop.empty:
        print(f"  ⚠ 人口1万未満のメッシュが上位に {len(low_pop)} 件:")
        for _, row in low_pop.iterrows():
            mesh_col = "jis_mesh" if "jis_mesh" in row.index else "jis_mesh3"
            print(f"    {row[mesh_col]} x {row['unified_genre']} (pop={int(row['population']):,})")
    else:
        print(f"  OK: 上位 {top_n} 件すべて人口1万以上")

    # 高飽和チェック
    if "saturation_index" in top.columns:
        median_sat = df["saturation_index"].median()
        high_sat = top[top["saturation_index"] > median_sat * 3]
        if not high_sat.empty:
            print(f"  ⚠ 飽和度が中央値の3倍超のメッシュが上位に {len(high_sat)} 件:")
            for _, row in high_sat.iterrows():
                mesh_col = "jis_mesh" if "jis_mesh" in row.index else "jis_mesh3"
                print(f"    {row[mesh_col]} x {row['unified_genre']} (sat={row['saturation_index']:.1f})")
        else:
            print(f"  OK: 上位 {top_n} 件すべて飽和度が適正範囲")

    print()


def analyze_rank_stability(df: pd.DataFrame, top_n: int) -> None:
    """v2→v3 で上位候補がどう変動したかを分析する。"""
    print("=" * 60)
    print("5. v2 → v3 順位変動分析")
    print("=" * 60)
    df = df.copy()
    df["rank_v2"] = df["score_v2"].rank(ascending=False, method="min").astype(int)
    df["rank_v3"] = df["score_v3"].rank(ascending=False, method="min").astype(int)
    df["rank_change"] = df["rank_v2"] - df["rank_v3"]

    top_v3 = df.nsmallest(top_n, "rank_v3")
    moved_up = top_v3[top_v3["rank_change"] > 5].sort_values("rank_change", ascending=False)
    moved_down = top_v3[top_v3["rank_change"] < -5].sort_values("rank_change")

    if not moved_up.empty:
        print(f"  ↑ v3で大幅上昇（5位以上）: {len(moved_up)} 件")
        for _, row in moved_up.head(5).iterrows():
            mesh_col = "jis_mesh" if "jis_mesh" in row.index else "jis_mesh3"
            print(f"    {row[mesh_col]} x {row['unified_genre']:10s} | "
                  f"v2={int(row['rank_v2'])}位 → v3={int(row['rank_v3'])}位 (+{int(row['rank_change'])})")

    if not moved_down.empty:
        print(f"  ↓ v3で大幅下降（5位以上）: {len(moved_down)} 件")
        for _, row in moved_down.head(5).iterrows():
            mesh_col = "jis_mesh" if "jis_mesh" in row.index else "jis_mesh3"
            print(f"    {row[mesh_col]} x {row['unified_genre']:10s} | "
                  f"v2={int(row['rank_v2'])}位 → v3={int(row['rank_v3'])}位 ({int(row['rank_change'])})")

    if moved_up.empty and moved_down.empty:
        print("  上位候補の順位変動は軽微です（±5位以内）")

    print()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    try:
        raw_df = load_integrated(args.tag)
        logger.info("統合データ読み込み: %d 行", len(raw_df))

        scored = compute_all_scores(raw_df)
        logger.info("全バージョンスコア計算完了")
        print()

        analyze_version_comparison(scored, args.top_n)
        analyze_feature_correlation(scored)
        analyze_top_candidate_quality(scored, args.top_n)
        analyze_rank_stability(scored, args.top_n)

        # 検証結果を保存
        output_path = settings.PROCESSED_DATA_DIR / f"{args.tag}_score_comparison.csv"
        scored.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info("スコア比較結果を保存: %s", output_path)

        return 0
    except Exception:
        logger.exception("スコア検証に失敗しました")
        return 1


if __name__ == "__main__":
    sys.exit(main())
