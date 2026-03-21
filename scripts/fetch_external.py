#!/usr/bin/env python3
"""外部データ（駅・地価）を事前取得するCLIスクリプト。"""

from __future__ import annotations

import argparse
import logging
import sys

from src.collect.land_price import (
    PREFECTURE_CODES,
    download_land_price,
    load_land_price_cache,
    save_land_price_cache,
)
from src.collect.station import (
    fetch_all_stations,
    load_station_cache,
    save_station_cache,
)

logger = logging.getLogger(__name__)

# タグ → (都道府県名, 都道府県コード)
AREA_MAP = {
    "tokyo": ("東京都", "13"),
    "osaka": ("大阪府", "27"),
    "nagoya": ("愛知県", "23"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="外部データ（駅・地価）を取得・キャッシュする。")
    parser.add_argument("--tag", type=str, required=True, choices=list(AREA_MAP.keys()),
                        help="対象エリアのタグ")
    parser.add_argument("--skip-stations", action="store_true", help="駅データ取得をスキップ")
    parser.add_argument("--skip-land-price", action="store_true", help="地価データ取得をスキップ")
    parser.add_argument("--force", action="store_true", help="キャッシュを無視して再取得")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    tag = args.tag
    pref_name, pref_code = AREA_MAP[tag]

    try:
        # 駅データ
        if not args.skip_stations:
            cached = None if args.force else load_station_cache(tag)
            if cached is not None:
                logger.info("駅データはキャッシュ済みです (%d 駅)", len(cached))
            else:
                logger.info("駅データを取得します: %s", pref_name)
                station_df = fetch_all_stations(pref_name)
                save_station_cache(station_df, tag)

        # 地価データ
        if not args.skip_land_price:
            cached = None if args.force else load_land_price_cache(tag)
            if cached is not None:
                logger.info("地価データはキャッシュ済みです (%d 件)", len(cached))
            else:
                logger.info("地価データを取得します: %s (pref_code=%s)", pref_name, pref_code)
                price_df = download_land_price(pref_code)
                save_land_price_cache(price_df, tag)

        return 0
    except Exception:
        logger.exception("外部データ取得に失敗しました")
        return 1


if __name__ == "__main__":
    sys.exit(main())
