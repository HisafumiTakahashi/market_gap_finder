#!/usr/bin/env python3
"""LightGBM市場ギャップ予測モデルの学習・評価スクリプト。"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from config import settings
from src.analyze.ml_model import (
    compare_with_v3,
    compute_market_gap,
    compute_shap_values,
    save_model,
    train_cv,
    train_full_model,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LightGBM市場ギャップ予測モデル学習")
    parser.add_argument("--tag", type=str, default="tokyo", help="対象データタグ")
    parser.add_argument("--combined", action="store_true", help="tokyo, osaka, nagoya を結合して学習")
    parser.add_argument("--top-n", type=int, default=20, help="表示する上位件数")
    parser.add_argument("--folds", type=int, default=5, help="CV分割数")
    return parser.parse_args()


def load_integrated(tag: str) -> pd.DataFrame:
    path = settings.PROCESSED_DATA_DIR / f"{tag}_integrated.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} が見つかりません。先に integrate_estat.py を実行してください。")
    return pd.read_csv(path)


def load_combined_integrated(tags: list[str]) -> pd.DataFrame:
    return pd.concat([load_integrated(tag) for tag in tags], ignore_index=True)


def print_cv_results(cv_results: dict) -> None:
    print("=" * 60)
    print("1. Cross-Validation 結果")
    print("=" * 60)
    for m in cv_results["fold_metrics"]:
        print(f"  Fold {m['fold']}: RMSE={m['rmse']:.4f}, R2={m['r2']:.4f}")
    print(f"  平均: RMSE={cv_results['avg_rmse']:.4f}, R2={cv_results['avg_r2']:.4f}")
    print()


def print_feature_importance(importance_df: pd.DataFrame) -> None:
    print("=" * 60)
    print("2. 特徴量重要度 (gain)")
    print("=" * 60)
    total = importance_df["importance"].sum()
    for _, row in importance_df.iterrows():
        pct = row["importance"] / total * 100 if total > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"  {row['feature']:>30s}: {row['importance']:>10.1f} ({pct:5.1f}%) {bar}")
    print()


def print_top_gaps(gap_df: pd.DataFrame, top_n: int) -> None:
    print("=" * 60)
    print(f"3. 市場ギャップ Top {top_n}（ML予測 - 実数）")
    print("=" * 60)
    top = gap_df.nlargest(top_n, "market_gap")
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        mesh = row.get("jis_mesh3", "")
        genre = row.get("unified_genre", "")
        gap = row["market_gap"]
        predicted = row["predicted_count"]
        actual = row["restaurant_count"]
        pop = int(row.get("population", 0))
        dist = row.get("nearest_station_distance", 0)
        print(
            f"  {rank:2d}. {mesh} x {genre:10s} | "
            f"gap={gap:+.3f} | pred={predicted:.1f} actual={actual:.0f} | "
            f"pop={pop:,} | station={dist:.2f}km"
        )
    print()


def print_v3_comparison(comparison_df: pd.DataFrame, top_n: int) -> None:
    print("=" * 60)
    print(f"4. ML vs v3 ランキング比較 (Top {top_n})")
    print("=" * 60)

    # ML上位
    ml_top = comparison_df.nsmallest(top_n, "rank_ml")
    v3_top = comparison_df.nsmallest(top_n, "rank_v3")

    # 重複率
    ml_set = set(ml_top.index)
    v3_set = set(v3_top.index)
    overlap = len(ml_set & v3_set)
    print(f"  上位{top_n}件の重複: {overlap}/{top_n} ({overlap/top_n*100:.0f}%)")

    # スピアマン相関
    corr = comparison_df["rank_ml"].corr(comparison_df["rank_v3"], method="spearman")
    print(f"  全体順位相関 (スピアマン): {corr:.4f}")
    print()

    # ML上位のうちv3と大きく異なるもの
    big_diff = ml_top[abs(ml_top["rank_diff"]) > 10].sort_values("rank_diff", ascending=False)
    if not big_diff.empty:
        print(f"  ML上位でv3と10位以上乖離: {len(big_diff)}件")
        for _, row in big_diff.head(5).iterrows():
            print(
                f"    {row.get('jis_mesh3', '')} x {row.get('unified_genre', ''):10s} | "
                f"ML={int(row['rank_ml'])}位 v3={int(row['rank_v3'])}位 (差={int(row['rank_diff'])})"
            )
    else:
        print("  ML上位とv3の乖離は軽微（10位以内）")
    print()


def print_shap_summary(shap_values: np.ndarray, features: pd.DataFrame) -> None:
    print("=" * 60)
    print("5. SHAP 特徴量寄与（平均絶対値）")
    print("=" * 60)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": features.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    for _, row in shap_df.iterrows():
        bar = "#" * int(row["mean_abs_shap"] * 20)
        print(f"  {row['feature']:>30s}: {row['mean_abs_shap']:.4f} {bar}")
    print()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    try:
        if args.combined:
            model_tag = "combined"
            df = load_combined_integrated(["tokyo", "osaka", "nagoya"])
        else:
            model_tag = args.tag
            df = load_integrated(args.tag)
        logger.info("統合データ読み込み: %d 行", len(df))
        print()

        # 1. Cross-Validation
        cv_results = train_cv(df, n_splits=args.folds)
        print_cv_results(cv_results)

        # 2. 特徴量重要度
        print_feature_importance(cv_results["feature_importance"])

        # 3. 市場ギャップ
        gap_df = compute_market_gap(df, cv_results["oof_predictions"])
        print_top_gaps(gap_df, args.top_n)

        # 4. v3比較
        if "opportunity_score" in gap_df.columns:
            comparison_df = compare_with_v3(gap_df)
            print_v3_comparison(comparison_df, args.top_n)

        # 5. SHAP（全データモデル）
        full_model = train_full_model(df)
        shap_values, features = compute_shap_values(full_model, df)
        print_shap_summary(shap_values, features)

        # モデル保存
        save_model(full_model, model_tag)

        # ギャップ結果保存
        output_path = settings.PROCESSED_DATA_DIR / f"{model_tag}_ml_gap.csv"
        gap_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info("MLギャップ結果を保存: %s", output_path)

        return 0
    except Exception:
        logger.exception("モデル学習に失敗しました")
        return 1


if __name__ == "__main__":
    sys.exit(main())
