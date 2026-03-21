#!/usr/bin/env python3
"""Google Places APIでエリアの飲食店データを収集するCLIスクリプト。"""

from __future__ import annotations

import argparse
import logging
import sys

from src.collect.google_collector import run_google_collection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Google Placesで飲食店データを収集")
    parser.add_argument("--lat-min", type=float, required=True)
    parser.add_argument("--lat-max", type=float, required=True)
    parser.add_argument("--lng-min", type=float, required=True)
    parser.add_argument("--lng-max", type=float, required=True)
    parser.add_argument("--tag", type=str, default="result")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    try:
        result = run_google_collection(
            lat_min=args.lat_min,
            lat_max=args.lat_max,
            lng_min=args.lng_min,
            lng_max=args.lng_max,
            output_tag=args.tag,
        )
        logging.info("Collected %d records", len(result))
        return 0
    except Exception:
        logging.exception("Google Places collection failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
