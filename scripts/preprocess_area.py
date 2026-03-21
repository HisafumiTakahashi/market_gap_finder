#!/usr/bin/env python3
"""収集済みエリアデータの前処理を CLI から実行するスクリプト。"""

from __future__ import annotations

import argparse
import logging
import sys

from src.preprocess.cleaner import run_preprocess


def parse_args() -> argparse.Namespace:
    """前処理対象の入出力タグに関する引数を解析する。

    Returns:
        前処理対象タグを保持した名前空間。
    """
    parser = argparse.ArgumentParser(description="Preprocess aggregated area data.")
    parser.add_argument("--tag", type=str, default="result", help="Input/output file tag.")
    return parser.parse_args()


def main() -> int:
    """前処理パイプラインを実行し、終了コードを返す。

    Returns:
        正常終了時は `0`、例外発生時は `1`。
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    try:
        result = run_preprocess(tag=args.tag)
        logging.info("Preprocessed %d rows", len(result))
        return 0
    except Exception:
        logging.exception("Failed to preprocess area data")
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
