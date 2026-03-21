#!/usr/bin/env python3
"""e-Stat 統合済みエリアデータを作成する CLI スクリプト。"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from config import settings
from src.analyze.features import add_all_features
from src.analyze.scoring import (
    compute_opportunity_score_v2,
    compute_opportunity_score_v3,
    generate_reason,
    run_scoring_v3,
)
from src.collect.estat import fetch_mesh_population, save_raw
from src.collect.land_price import load_land_price_cache
from src.collect.station import load_station_cache
from src.preprocess.cleaner import load_hotpepper, map_genre
from src.preprocess.mesh_converter import assign_jis_mesh, mesh3_to_mesh1


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """CLI 引数を解釈する。"""
    parser = argparse.ArgumentParser(description="e-Stat mesh population integration pipeline.")
    parser.add_argument("--tag", type=str, default="tokyo", help="対象データのタグ。")
    parser.add_argument("--top-n", type=int, default=20, help="表示する上位件数。")
    parser.add_argument(
        "--stats-id",
        type=str,
        default="0003412636",
        help="取得対象の国勢調査統計表 ID。",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="e-Stat API 取得をスキップして既存 CSV を利用する。",
    )
    return parser.parse_args()


def aggregate_hotpepper_by_mesh(df: pd.DataFrame) -> pd.DataFrame:
    """HotPepper データを JIS 3 次メッシュとジャンル単位で集計する。"""
    prepared = assign_jis_mesh(map_genre(df))
    prepared["rating"] = pd.to_numeric(prepared.get("rating"), errors="coerce")
    prepared["review_count"] = (
        pd.to_numeric(prepared.get("review_count"), errors="coerce").fillna(0).astype(int)
    )
    prepared["lat"] = pd.to_numeric(prepared.get("lat"), errors="coerce")
    prepared["lng"] = pd.to_numeric(prepared.get("lng"), errors="coerce")
    prepared = prepared.dropna(subset=["jis_mesh3"]).copy()

    aggregated = (
        prepared.groupby(["jis_mesh3", "unified_genre"], dropna=False)
        .agg(
            restaurant_count=("id", "size"),
            avg_rating=("rating", "mean"),
            total_reviews=("review_count", "sum"),
            lat=("lat", "mean"),
            lng=("lng", "mean"),
        )
        .reset_index()
    )
    aggregated["mesh_code"] = aggregated["jis_mesh3"]
    return aggregated


def normalize_population_df(df: pd.DataFrame) -> pd.DataFrame:
    """e-Stat メッシュ人口データを 3 次メッシュ単位へ正規化する。"""
    if df.empty:
        return pd.DataFrame(columns=["jis_mesh3", "population"])

    normalized = df.copy()

    if "mesh_code" not in normalized.columns:
        raise KeyError("mesh_code column is required in e-Stat data")

    # 8桁のメッシュコードを抽出（3次メッシュ）
    normalized["jis_mesh3"] = normalized["mesh_code"].astype(str).str.extract(r"(\d{8})", expand=False)
    normalized["population"] = (
        normalized["population"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )

    # cat01 フィルタは fetch_mesh_population 側で実施済みだが、念のためフォールバック
    if "cat01" in normalized.columns:
        cat01_str = normalized["cat01"].astype(str)
        total_mask = cat01_str.isin(["0010"]) | cat01_str.str.contains("総数|総人口|人口（総数）", na=False)
        if total_mask.any():
            normalized = normalized.loc[total_mask].copy()

    normalized = normalized.dropna(subset=["jis_mesh3", "population"])
    if normalized.empty:
        return pd.DataFrame(columns=["jis_mesh3", "population"])

    return (
        normalized.groupby("jis_mesh3", as_index=False)
        .agg(population=("population", "sum"))
        .sort_values("jis_mesh3")
        .reset_index(drop=True)
    )


def load_or_fetch_population(tag: str, mesh1_codes: list[str], skip_fetch: bool) -> pd.DataFrame:
    """e-Stat 人口データを取得または既存 CSV から読み込む。"""
    cache_name = f"{tag}_estat_population.csv"
    cache_path = settings.RAW_DATA_DIR / cache_name

    if skip_fetch:
        if not cache_path.exists():
            raise FileNotFoundError(cache_path)
        logger.info("既存の e-Stat 人口データを読み込みます: %s", cache_path)
        return pd.read_csv(cache_path)

    logger.info("e-Stat API から %d 件の 1 次メッシュ人口データを取得します", len(mesh1_codes))
    population_df = fetch_mesh_population(mesh1_codes=mesh1_codes)
    save_raw(population_df, cache_name)
    return population_df


def save_integrated(df: pd.DataFrame, tag: str) -> None:
    """統合済みデータを保存する。"""
    settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = settings.PROCESSED_DATA_DIR / f"{tag}_integrated.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("統合結果を保存しました: %s", output_path)


def main() -> int:
    """e-Stat 統合パイプラインを実行する。"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    try:
        logger.info("HotPepper データを読み込みます: tag=%s", args.tag)
        hotpepper_df = load_hotpepper(args.tag)

        logger.info("ジャンル正規化と JIS 3 次メッシュ付与を実行します")
        mesh_agg_df = aggregate_hotpepper_by_mesh(hotpepper_df)
        logger.info("HotPepper メッシュ集計件数: %d", len(mesh_agg_df))

        mesh1_codes = sorted({mesh3_to_mesh1(code) for code in mesh_agg_df["jis_mesh3"].dropna().astype(str)})
        logger.info("必要な 1 次メッシュ数: %d", len(mesh1_codes))

        population_raw_df = load_or_fetch_population(
            tag=args.tag,
            mesh1_codes=mesh1_codes,
            skip_fetch=args.skip_fetch,
        )
        population_df = normalize_population_df(population_raw_df)
        logger.info("正規化後の人口メッシュ件数: %d", len(population_df))

        integrated_df = mesh_agg_df.merge(population_df, on="jis_mesh3", how="left")
        integrated_df["population"] = pd.to_numeric(
            integrated_df["population"], errors="coerce"
        ).fillna(0.0)

        logger.info("空間特徴量を追加します")
        # 外部データ（駅・地価）があればロード
        station_df = load_station_cache(args.tag)
        price_df = load_land_price_cache(args.tag)
        if station_df is not None:
            logger.info("駅データを読み込みました: %d 駅", len(station_df))
        if price_df is not None:
            logger.info("地価データを読み込みました: %d 件", len(price_df))

        featured_df = add_all_features(integrated_df, station_df=station_df, price_df=price_df)

        full_scored_df = compute_opportunity_score_v3(featured_df).sort_values(
            "opportunity_score", ascending=False
        )
        full_scored_df["reason"] = full_scored_df.apply(generate_reason, axis=1)
        save_integrated(full_scored_df, args.tag)

        top_candidates = run_scoring_v3(featured_df, top_n=args.top_n)
        logger.info("上位 %d 件を表示します", min(args.top_n, len(top_candidates)))
        for rank, row in enumerate(top_candidates.itertuples(index=False), start=1):
            logger.info(
                "%d位: mesh=%s genre=%s score=%.4f population=%.0f",
                rank,
                getattr(row, "jis_mesh3", ""),
                getattr(row, "unified_genre", ""),
                getattr(row, "opportunity_score", 0.0),
                getattr(row, "population", 0.0),
            )

        return 0
    except Exception:
        logger.exception("e-Stat 統合パイプラインの実行に失敗しました")
        return 1


if __name__ == "__main__":
    sys.exit(main())
