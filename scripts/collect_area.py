#!/usr/bin/env python3

"""指定エリアの店舗データ収集を CLI から実行するスクリプト。"""

import argparse
import logging
import sys

from src.collect.collector import run_collection


def parse_args() -> argparse.Namespace:
    """収集対象エリアを表すコマンドライン引数を解析する。

    Returns:
        緯度経度範囲と出力タグを保持した名前空間。
    """
    parser = argparse.ArgumentParser(
        description="Collect data for a specified geographic area."
    )
    parser.add_argument("--lat-min", type=float, required=True)
    parser.add_argument("--lat-max", type=float, required=True)
    parser.add_argument("--lng-min", type=float, required=True)
    parser.add_argument("--lng-max", type=float, required=True)
    parser.add_argument("--tag", type=str, default="result")
    return parser.parse_args()


def main() -> int:
    """収集処理を実行し、終了コードを返す。

    Returns:
        正常終了時は `0`、例外発生時は `1`。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()

    try:
        result = run_collection(
            lat_min=args.lat_min,
            lat_max=args.lat_max,
            lng_min=args.lng_min,
            lng_max=args.lng_max,
            output_tag=args.tag,
        )
        logging.info("Collected %d rows", len(result))
        return 0
    except Exception:
        logging.exception("Collection failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
