#!/usr/bin/env python3
"""外部データを取得してキャッシュ保存するスクリプト。"""

from __future__ import annotations

import argparse
import logging
import sys

from src.collect.land_price import download_land_price, load_land_price_cache, save_land_price_cache
from src.collect.station import fetch_all_stations, load_station_cache, save_station_cache
from src.collect.station_passengers import (
    download_station_passengers,
    load_passenger_cache,
    save_passenger_cache,
)

logger = logging.getLogger(__name__)

AREA_MAP = {
    "tokyo": ("東京都", "13"),
    "osaka": ("大阪府", "27"),
    "nagoya": ("愛知県", "23"),
}


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    parser = argparse.ArgumentParser(description="外部データを取得してキャッシュ保存する。")
    parser.add_argument("--tag", type=str, required=True, choices=list(AREA_MAP.keys()), help="対象エリアのタグ")
    parser.add_argument("--skip-stations", action="store_true", help="駅データ取得をスキップ")
    parser.add_argument("--skip-land-price", action="store_true", help="地価データ取得をスキップ")
    parser.add_argument("--skip-passengers", action="store_true", help="駅乗降客数データ取得をスキップ")
    parser.add_argument("--force", action="store_true", help="キャッシュを無視して再取得")
    return parser.parse_args()


def main() -> int:
    """外部データ取得処理を実行する。"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    tag = args.tag
    pref_name, pref_code = AREA_MAP[tag]

    try:
        if not args.skip_stations:
            cached = None if args.force else load_station_cache(tag)
            if cached is not None:
                logger.info("駅データはキャッシュ済みです (%d 件)", len(cached))
            else:
                logger.info("駅データを取得します: %s", pref_name)
                station_df = fetch_all_stations(pref_name)
                save_station_cache(station_df, tag)

        if not args.skip_land_price:
            cached = None if args.force else load_land_price_cache(tag)
            if cached is not None:
                logger.info("地価データはキャッシュ済みです (%d 件)", len(cached))
            else:
                logger.info("地価データを取得します: %s (pref_code=%s)", pref_name, pref_code)
                price_df = download_land_price(pref_code)
                save_land_price_cache(price_df, tag)

        if not args.skip_passengers:
            cached = None if args.force else load_passenger_cache(tag)
            if cached is not None:
                logger.info("駅乗降客数データはキャッシュ済みです (%d 件)", len(cached))
            else:
                logger.info("駅乗降客数データを取得します")
                passenger_df = download_station_passengers()
                save_passenger_cache(passenger_df, tag)

        return 0
    except Exception:
        logger.exception("外部データ取得に失敗しました")
        return 1


if __name__ == "__main__":
    sys.exit(main())
