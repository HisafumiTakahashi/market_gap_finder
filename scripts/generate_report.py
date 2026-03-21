#!/usr/bin/env python3
"""市場ギャップ分析レポートを生成するCLIスクリプト。"""

from __future__ import annotations

import argparse
import logging
import sys

from config import settings
from src.visualize.report import generate_report

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="市場ギャップ分析HTMLレポートを生成する。")
    parser.add_argument("--tag", type=str, default="tokyo", help="対象エリアタグ")
    parser.add_argument("--top-n", type=int, default=20, help="表示する上位件数")
    parser.add_argument("--all", action="store_true", help="全エリア（tokyo/osaka/nagoya）を生成")
    return parser.parse_args()


def get_ml_r2(tag: str) -> str:
    """MLギャップCSVからR2を推定する（train_model.pyで保存済みならそちらを使う）。"""
    import pandas as pd
    ml_path = settings.PROCESSED_DATA_DIR / f"{tag}_ml_gap.csv"
    if not ml_path.exists():
        return "-"
    # R2はCSVに保存していないが、存在すれば学習済みと判断
    return "学習済み"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    tags = ["tokyo", "osaka", "nagoya"] if args.all else [args.tag]

    try:
        for tag in tags:
            integrated_path = settings.PROCESSED_DATA_DIR / f"{tag}_integrated.csv"
            if not integrated_path.exists():
                logger.warning("%s の統合データが見つかりません。スキップします。", tag)
                continue

            ml_r2 = get_ml_r2(tag)
            output_path = generate_report(tag=tag, top_n=args.top_n, ml_r2=ml_r2)
            print(f"レポート生成完了: {output_path}")

        return 0
    except Exception:
        logger.exception("レポート生成に失敗しました")
        return 1


if __name__ == "__main__":
    sys.exit(main())
