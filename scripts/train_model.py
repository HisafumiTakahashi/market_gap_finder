#!/usr/bin/env python3
"""LightGBM市場ギャップ予測モデルの学習スクリプト。"""

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
    train_leave_one_area_out,
    tune_hyperparams,
)

logger = logging.getLogger(__name__)
ALL_TAGS = ["tokyo", "osaka", "nagoya", "fukuoka"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LightGBM市場ギャップ予測モデル学習")
    parser.add_argument("--tag", type=str, default="tokyo", help="対象エリアタグ")
    parser.add_argument("--combined", action="store_true", help="tokyo, osaka, nagoya を結合して学習")
    parser.add_argument("--top-n", type=int, default=20, help="表示する上位候補数")
    parser.add_argument("--folds", type=int, default=5, help="CV分割数")
    parser.add_argument("--tune", action="store_true", help="Optunaでハイパーパラメータを探索")
    parser.add_argument("--loao", action="store_true", help="Leave-One-Area-Out CVで汎化性能を測定")
    parser.add_argument("--two-stage", action="store_true", help="2段階モデル（残差ターゲット）で学習")
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
    for metric in cv_results["fold_metrics"]:
        print(f"  Fold {metric['fold']}: RMSE={metric['rmse']:.4f}, R2={metric['r2']:.4f}")
    print(f"  平均   RMSE={cv_results['avg_rmse']:.4f}, R2={cv_results['avg_r2']:.4f}")
    print()


def print_tuning_results(tuning_results: dict) -> None:
    print("=" * 60)
    print("2. Optuna 最適化結果")
    print("=" * 60)
    print(f"  best_rmse={tuning_results['best_rmse']:.4f}, best_r2={tuning_results['best_r2']:.4f}")
    print(f"  num_boost_round={tuning_results['best_num_rounds']}")
    for key, value in tuning_results["best_params"].items():
        print(f"  {key}={value}")
    print()


def print_feature_importance(importance_df: pd.DataFrame, section_no: int = 2) -> None:
    print("=" * 60)
    print(f"{section_no}. 特徴量重要度 (gain)")
    print("=" * 60)
    total = importance_df["importance"].sum()
    for _, row in importance_df.iterrows():
        pct = row["importance"] / total * 100 if total > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"  {row['feature']:>30s}: {row['importance']:>10.1f} ({pct:5.1f}%) {bar}")
    print()


def print_top_gaps(gap_df: pd.DataFrame, top_n: int, section_no: int = 3) -> None:
    print("=" * 60)
    print(f"{section_no}. 市場ギャップ Top {top_n}")
    print("=" * 60)
    top = gap_df.nlargest(top_n, "market_gap")
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        print(
            f"  {rank:2d}. {row.get('jis_mesh', row.get('jis_mesh3', ''))} x {row.get('unified_genre', ''):10s} | "
            f"gap={row['market_gap']:+.3f} | pred={row['predicted_count']:.1f} actual={row['restaurant_count']:.0f} | "
            f"pop={int(row.get('population', 0)):,} | station={row.get('nearest_station_distance', 0):.2f}km"
        )
    print()


def print_v3_comparison(comparison_df: pd.DataFrame, top_n: int, section_no: int = 4) -> None:
    print("=" * 60)
    print(f"{section_no}. ML vs v3 ランキング比較 (Top {top_n})")
    print("=" * 60)

    ml_top = comparison_df.nsmallest(top_n, "rank_ml")
    v3_top = comparison_df.nsmallest(top_n, "rank_v3")
    overlap = len(set(ml_top.index) & set(v3_top.index))
    print(f"  Top {top_n} の重複: {overlap}/{top_n} ({overlap / top_n * 100:.0f}%)")

    corr = comparison_df["rank_ml"].corr(comparison_df["rank_v3"], method="spearman")
    print(f"  順位相関 (spearman): {corr:.4f}")
    print()

    big_diff = ml_top[abs(ml_top["rank_diff"]) > 10].sort_values("rank_diff", ascending=False)
    if big_diff.empty:
        print("  ML上位候補と v3 の順位差は概ね 10 位以内です。")
        print()
        return

    print(f"  ML上位候補で v3 と 10 位超差があるもの: {len(big_diff)}件")
    for _, row in big_diff.head(5).iterrows():
        print(
            f"    {row.get('jis_mesh', row.get('jis_mesh3', ''))} x {row.get('unified_genre', ''):10s} | "
            f"ML={int(row['rank_ml'])}位 v3={int(row['rank_v3'])}位 (差={int(row['rank_diff'])})"
        )
    print()


def print_shap_summary(shap_values: np.ndarray, features: pd.DataFrame, section_no: int = 5) -> None:
    print("=" * 60)
    print(f"{section_no}. SHAP 特徴量寄与")
    print("=" * 60)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({"feature": features.columns, "mean_abs_shap": mean_abs_shap})
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)

    for _, row in shap_df.iterrows():
        bar = "#" * int(row["mean_abs_shap"] * 20)
        print(f"  {row['feature']:>30s}: {row['mean_abs_shap']:.4f} {bar}")
    print()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    try:
        available_tags: list[str] = []
        if args.combined:
            model_tag = "combined"
            available_tags = [t for t in ALL_TAGS if (settings.PROCESSED_DATA_DIR / f"{t}_integrated.csv").exists()]
            if not available_tags:
                raise FileNotFoundError("No integrated CSV files found for combined training")
            df = load_combined_integrated(available_tags)
        else:
            model_tag = args.tag
            df = load_integrated(args.tag)

        logger.info("統合データ読み込み: %d rows", len(df))
        print()

        best_params: dict | None = None
        best_num_rounds: int | None = None

        if args.tune:
            tuning_results = tune_hyperparams(df, n_trials=50, n_splits=args.folds)
            print_tuning_results(tuning_results)
            best_params = tuning_results["best_params"]
            best_num_rounds = tuning_results["best_num_rounds"]

        target_mode = "residual" if args.two_stage else "raw"
        cv_results = train_cv(
            df,
            n_splits=args.folds,
            params=best_params,
            num_rounds=best_num_rounds or 300,
            target_mode=target_mode,
        )
        print_cv_results(cv_results)
        if args.loao and args.combined and len(available_tags) >= 2:
            loao_results = train_leave_one_area_out(available_tags, params=best_params, num_rounds=best_num_rounds or 300)
            print("=" * 60)
            print("Leave-One-Area-Out CV 結果")
            print("=" * 60)
            for r in loao_results["area_results"]:
                print(f"  {r['test_area']:10s}: RMSE={r['rmse']:.4f}, R2={r['r2']:.4f}")
            print(f"  {'平均':10s}: RMSE={loao_results['avg_rmse']:.4f}, R2={loao_results['avg_r2']:.4f}")
            print()
        print_feature_importance(cv_results["feature_importance"], section_no=3 if args.tune else 2)

        gap_df = compute_market_gap(df, cv_results["oof_predictions"], target_mode=target_mode)
        print_top_gaps(gap_df, args.top_n, section_no=4 if args.tune else 3)

        next_section = 5 if args.tune else 4
        if "opportunity_score" in gap_df.columns:
            comparison_df = compare_with_v3(gap_df)
            print_v3_comparison(comparison_df, args.top_n, section_no=next_section)
            next_section += 1

        full_model = train_full_model(
            df,
            params=best_params,
            num_rounds=best_num_rounds or 300,
            target_mode=target_mode,
        )
        shap_values, features = compute_shap_values(full_model, df)
        print_shap_summary(shap_values, features, section_no=next_section)

        save_model(full_model, model_tag)

        output_path = settings.PROCESSED_DATA_DIR / f"{model_tag}_ml_gap.csv"
        gap_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info("MLギャップ結果を保存: %s", output_path)
        return 0
    except Exception:
        logger.exception("モデル学習に失敗しました")
        return 1


if __name__ == "__main__":
    sys.exit(main())
